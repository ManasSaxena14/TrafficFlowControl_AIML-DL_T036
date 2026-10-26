"""Phase 2: one leak-free preprocessing pipeline that feeds every model.

Output per protocol (data/processed/<protocol>/): train.npz, val.npz, test.npz, scaler.json.
Shared:                                         data/processed/adj.npz

Each split holds
    X          (S, 12, N, 7)  float32  speed_z, mask, sin/cos time-of-day, sin/cos day-of-week, weekend
    Y          (S, 12, N)     float32  future speed in mph; NaN where the sensor was down (never 0)
    M_y        (S, 12, N)     bool     True = real reading (the target mask for loss and metrics)
    origin_idx (S,)           int64    index of the last input timestamp
    target_time(S, 12)        datetime64[ns]  timestamps being predicted (peak / off-peak is judged on these)

Run: python -m src.data.preprocess [--protocol strict|dcrnn]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.load import SensorGraph, load_all, load_distances

ROOT = Path(__file__).resolve().parents[2]
SLOTS_PER_DAY = 288
N_FEATURES = 7
FEATURE_NAMES = ["speed_z", "mask", "sin_tod", "cos_tod", "sin_dow", "cos_dow", "is_weekend"]
CHUNK = 1024


def load_config(path: str | Path = ROOT / "configs" / "data" / "metr_la.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------------- 1. Chronological split
def split_bounds(n: int, split: dict) -> dict[str, tuple[int, int]]:
    """Half-open [start, end) blocks over n items: train first, then val, then test."""
    n_train = int(round(n * split["train"]))
    n_test = int(round(n * split["test"]))
    n_val = n - n_train - n_test
    return {"train": (0, n_train), "val": (n_train, n_train + n_val), "test": (n_train + n_val, n)}


def plan_windows(n_steps: int, cfg: dict, protocol: str) -> tuple[dict[str, np.ndarray], int]:
    """Choose the forecast origins (index of the last input step) of every split.

    strict: split the *timestamps* 70/10/20 first, then keep a window only if all its targets fall
            inside one block. Windows whose targets straddle a boundary are purged, so no target
            timestamp appears in two splits. Inputs may look back into earlier data (the past is
            not a label), exactly as at deployment.
    dcrnn : the reference script. Build all windows, then split the *windows* 70/10/20. The last
            train targets overlap the first val inputs and targets. Kept only for comparison.

    Returns (origins per split, train_end): train_end = exclusive end of the timestamps the scaler
    and the imputation table may use.
    """
    h, hor = cfg["window"]["history"], cfg["window"]["horizon"]
    if protocol == "dcrnn":
        origins = np.arange(h - 1, n_steps - hor)
        b = split_bounds(len(origins), cfg["split"])
        out = {k: origins[a:z] for k, (a, z) in b.items()}
        return out, int(out["train"][-1]) + 1
    if protocol != "strict":
        raise ValueError(f"unknown protocol {protocol!r}")
    b = split_bounds(n_steps, cfg["split"])
    out = {}
    for k, (a, z) in b.items():
        lo = max(a - 1, h - 1)          # first target = origin + 1 >= block start
        hi = z - 1 - hor                # last target  = origin + hor <= block end - 1
        out[k] = np.arange(lo, hi + 1)
    return out, b["train"][1]


# --------------------------------------------------------------------------- 2-3. Masking, scaling
def fit_scaler(speed: np.ndarray, train_end: int, include_zeros: bool = False) -> dict:
    """Mean/std over TRAIN timestamps only. Outages are NaN, so they do not bias it.

    include_zeros=True reproduces the DCRNN reference, which averages the 0-mph outages in.
    """
    tr = speed[:train_end]
    if include_zeros:
        return {"mu": float(np.nan_to_num(tr).mean()), "sd": float(np.nan_to_num(tr).std()), "includes_zeros": True}
    return {"mu": float(np.nanmean(tr)), "sd": float(np.nanstd(tr)), "includes_zeros": False}


def slot_mean_table(speed: np.ndarray, index: pd.DatetimeIndex, train_end: int) -> np.ndarray:
    """(288, N) mean speed per (time-of-day slot, sensor) over train timestamps; the imputation fallback."""
    slot = (index.hour * 12 + index.minute // 5).to_numpy()
    tab = pd.DataFrame(speed[:train_end]).groupby(slot[:train_end]).mean().reindex(range(SLOTS_PER_DAY)).to_numpy()
    sensor_mean = np.nanmean(speed[:train_end], axis=0)
    tab = np.where(np.isnan(tab), sensor_mean[None, :], tab)
    return np.where(np.isnan(tab), np.nanmean(speed[:train_end]), tab).astype(np.float32)


# --------------------------------------------------------------------------- 4. Cyclic time features
def time_features(index: pd.DatetimeIndex) -> np.ndarray:
    """(T, 5): sin/cos time-of-day, sin/cos day-of-week, weekend flag. 23:55 and 00:05 end up neighbours."""
    tod = (index.hour * 60 + index.minute) / 1440
    dow = index.dayofweek / 7
    return np.stack([
        np.sin(2 * np.pi * tod), np.cos(2 * np.pi * tod),
        np.sin(2 * np.pi * dow), np.cos(2 * np.pi * dow),
        (index.dayofweek >= 5).astype("float32"),
    ], axis=1).astype("float32")


# --------------------------------------------------------------------------- 5. Adjacency
def rebuild_adjacency(distances: pd.DataFrame, sensor_ids: list[str], kappa: float) -> np.ndarray:
    """W_ij = exp(-(d_ij / sigma)^2), sigma = std of all finite road distances, zeroed below kappa. Directed."""
    idx = {s: i for i, s in enumerate(sensor_ids)}
    dist = np.full((len(sensor_ids), len(sensor_ids)), np.inf)
    for f, t, c in zip(distances["from"], distances["to"], distances["cost"]):
        if f in idx and t in idx:
            dist[idx[f], idx[t]] = c
    sigma = dist[~np.isinf(dist)].std()
    adj = np.exp(-np.square(dist / sigma))
    adj[adj < kappa] = 0
    return adj.astype(np.float32)


def adjacency_arrays(adj: np.ndarray, sensor_ids: list[str]) -> dict[str, np.ndarray]:
    """Dense matrix plus the sparse edge list a PyG model wants (row = source i, col = target j)."""
    src, dst = np.nonzero(adj)
    return {"adj": adj, "edge_index": np.stack([src, dst]).astype(np.int64),
            "edge_weight": adj[src, dst].astype(np.float32), "sensor_ids": np.array(sensor_ids)}


# --------------------------------------------------------------------------- 6. Windowing
def build_split(origins: np.ndarray, raw: np.ndarray, last_valid: np.ndarray, tfeat: np.ndarray, slot_tab: np.ndarray,
                slots: np.ndarray, index: pd.DatetimeIndex, scaler: dict, cfg: dict) -> dict[str, np.ndarray]:
    h, hor = cfg["window"]["history"], cfg["window"]["horizon"]
    limit = cfg["preprocess"]["impute_ffill_limit"]
    n_sens = raw.shape[1]
    cols = np.arange(n_sens)
    raw0 = np.nan_to_num(raw)
    S = len(origins)
    X = np.empty((S, h, n_sens, N_FEATURES), np.float32)
    Y = np.empty((S, hor, n_sens), np.float32)
    for a in range(0, S, CHUNK):
        o = origins[a:a + CHUNK]
        t_in = o[:, None] + np.arange(-h + 1, 1)                      # (s, h)
        t_out = o[:, None] + np.arange(1, hor + 1)                    # (s, hor)
        lv = last_valid[t_in]                                         # (s, h, N): last real reading <= t
        usable = (lv >= (o - h + 1)[:, None, None]) & ((t_in[:, :, None] - lv) <= limit)   # inside THIS window, past only
        filled = raw0[np.clip(lv, 0, None), cols]
        value = np.where(usable, filled, slot_tab[slots[t_in]])      # else train (slot, sensor) mean
        real = ~np.isnan(raw[t_in])
        X[a:a + CHUNK, :, :, 0] = (value - scaler["mu"]) / scaler["sd"]
        X[a:a + CHUNK, :, :, 1] = real
        X[a:a + CHUNK, :, :, 2:] = tfeat[t_in][:, :, None, :]
        Y[a:a + CHUNK] = raw[t_out]
    t_out_all = origins[:, None] + np.arange(1, hor + 1)
    return {"X": X, "Y": Y, "M_y": ~np.isnan(Y), "origin_idx": origins.astype(np.int64),
            "target_time": index.to_numpy()[t_out_all]}


def build_dataset(speed: pd.DataFrame, cfg: dict, protocol: str | None = None) -> dict:
    """Run the whole pipeline in memory. Returns {'splits', 'scaler', 'plan', 'dropped_train', 'train_end'}."""
    protocol = protocol or cfg["preprocess"]["protocol"]
    plan, train_end = plan_windows(len(speed), cfg, protocol)
    values = speed.values.astype("float32")
    raw = np.where(values > cfg["masking"]["missing_value"], values, np.nan)   # outage -> NaN, never "0 traffic"
    scaler = fit_scaler(raw, train_end, include_zeros=(protocol == "dcrnn"))
    scaler["protocol"] = protocol
    slot_tab = slot_mean_table(raw, speed.index, train_end)
    slots = (speed.index.hour * 12 + speed.index.minute // 5).to_numpy()
    last_valid = np.maximum.accumulate(np.where(~np.isnan(raw), np.arange(len(raw))[:, None], -1), axis=0)
    tfeat = time_features(speed.index)

    dropped = 0
    if protocol == "strict":
        thr = cfg["preprocess"]["drop_train_target_masked_frac"]
        hor = cfg["window"]["horizon"]
        real = ~np.isnan(raw)
        keep = np.array([1 - real[o + 1:o + hor + 1].mean() <= thr for o in plan["train"]])
        dropped = int((~keep).sum())
        plan["train"] = plan["train"][keep]
    splits = {k: build_split(o, raw, last_valid, tfeat, slot_tab, slots, speed.index, scaler, cfg) for k, o in plan.items()}
    return {"splits": splits, "scaler": scaler, "plan": plan, "dropped_train": dropped, "train_end": train_end}


# --------------------------------------------------------------------------- I/O + CLI
def save_dataset(ds: dict, graph: SensorGraph, out_root: Path, protocol: str) -> Path:
    out = out_root / protocol
    out.mkdir(parents=True, exist_ok=True)
    for name, arrays in ds["splits"].items():
        np.savez(out / f"{name}.npz", **arrays)
    (out / "scaler.json").write_text(json.dumps(ds["scaler"], indent=2))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--protocol", choices=["strict", "dcrnn"], default=None)
    args = ap.parse_args()
    cfg = load_config()
    protocol = args.protocol or cfg["preprocess"]["protocol"]
    out_root = ROOT / cfg["preprocess"]["processed_dir"]

    speed, graph, _ = load_all()
    rebuilt = rebuild_adjacency(load_distances(), graph.sensor_ids, cfg["adjacency"]["kernel_threshold"])
    if not np.allclose(rebuilt, graph.adj, atol=1e-5):
        raise ValueError("Adjacency rebuilt from distances does not match adj_mx.pkl")
    out_root.mkdir(parents=True, exist_ok=True)
    np.savez(out_root / "adj.npz", **adjacency_arrays(graph.adj, graph.sensor_ids))

    ds = build_dataset(speed, cfg, protocol)
    out = save_dataset(ds, graph, out_root, protocol)
    sc = ds["scaler"]
    print(f"protocol={protocol}  scaler mu={sc['mu']:.3f} sd={sc['sd']:.3f}  "
          f"dropped train windows (targets >90% masked)={ds['dropped_train']}")
    for k, s in ds["splits"].items():
        t = s["target_time"]
        print(f"{k:5s} X{s['X'].shape} Y{s['Y'].shape}  targets {t.min()} -> {t.max()}  "
              f"masked targets {100 * (1 - s['M_y'].mean()):.2f}%")
    print("saved to", out)


if __name__ == "__main__":
    main()
