"""Phase 3 feature table for the per-segment ML baselines.

One row = one sensor at one forecast origin t (row index = window * 207 + sensor). Everything comes from
the Phase 2 input windows, so every model sees the same 12-step (60 min) history and the same imputation.
METR-LA is speed only: flow and occupancy are not available (PeMS, Phase 9) and are stated as a limitation.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.dataset import PROCESSED

HIST = 12
HORIZON = 12
LAGS = [f"speed_t-{k}" if k else "speed_t" for k in range(HIST)]
ROLL = [f"{stat}_{w}" for w in (3, 6, 12) for stat in ("mean", "std", "min")]
CHANGE = ["delta_1", "delta_3"]
MISSING = [f"real_t-{k}" if k else "real_t" for k in range(HIST)] + ["n_imputed"]
TIME = ["sin_tod", "cos_tod", "sin_dow", "cos_dow", "is_weekend"]
SEGMENT = ["sensor_id", "sensor_mean", "free_flow_p85", "in_degree", "out_degree"]
FEATURE_NAMES = LAGS + ROLL + CHANGE + MISSING + TIME + SEGMENT
DYNAMIC = FEATURE_NAMES[: len(FEATURE_NAMES) - len(SEGMENT)]   # everything that varies in time for one sensor


def static_features(speed: pd.DataFrame, adj: np.ndarray, train_end: int) -> np.ndarray:
    """(N, 5) per-sensor constants computed on TRAIN timestamps only: id, mean speed, free-flow (85th pct), in/out degree."""
    v = speed.values[:train_end].astype("float32")
    v = np.where(v > 0, v, np.nan)
    off = (adj > 0) & ~np.eye(len(adj), dtype=bool)
    return np.stack([np.arange(adj.shape[0]), np.nanmean(v, axis=0), np.nanpercentile(v, 85, axis=0),
                     off.sum(axis=0), off.sum(axis=1)], axis=1).astype("float32")


def build_features(X: np.ndarray, scaler: dict, static: np.ndarray, chunk: int = 512) -> np.ndarray:
    """X (S, 12, N, 7) from Phase 2 -> (S * N, len(FEATURE_NAMES)) float32."""
    S, _, N, _ = X.shape
    out = np.empty((S, N, len(FEATURE_NAMES)), np.float32)
    for a in range(0, S, chunk):
        x = X[a:a + chunk]
        mph = x[..., 0] * scaler["sd"] + scaler["mu"]                       # (s, 12, N), imputed, in mph
        lags = mph[:, ::-1].transpose(0, 2, 1)                              # (s, N, 12): index k = speed at t-k
        real = x[:, ::-1, :, 1].transpose(0, 2, 1)                          # (s, N, 12): 1 = real reading
        cols = [lags]
        for w in (3, 6, 12):
            win = lags[..., :w]
            cols += [win.mean(-1, keepdims=True), win.std(-1, keepdims=True), win.min(-1, keepdims=True)]
        cols += [lags[..., :1] - lags[..., 1:2], lags[..., :1] - lags[..., 3:4]]
        cols += [real, (HIST - real.sum(-1, keepdims=True))]
        cols += [np.broadcast_to(x[:, -1, :1, 2:7], (x.shape[0], N, 5))]    # time features at t (same for every sensor)
        cols += [np.broadcast_to(static[None], (x.shape[0], N, static.shape[1]))]
        out[a:a + chunk] = np.concatenate(cols, axis=-1)
    return out.reshape(S * N, -1)


def flatten_targets(Y: np.ndarray, M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(S, 12, N) -> (S * N, 12) in the same row order as build_features. Masked targets stay NaN."""
    S, H, N = Y.shape
    return Y.transpose(0, 2, 1).reshape(S * N, H), M.transpose(0, 2, 1).reshape(S * N, H)


def _paths(split: str, protocol: str) -> dict[str, Path]:
    d = PROCESSED / protocol / "features"
    return {k: d / f"{k}_{split}.npy" for k in ("F", "Y", "M")}


def build_and_cache(protocol: str = "strict") -> None:
    """Build the feature tables once and store them as .npy so models can memory-map them (8 GB machines)."""
    from src.data.dataset import load_adjacency, load_scaler
    from src.data.load import load_speed
    from src.data.preprocess import load_config, split_bounds

    cfg = load_config()
    speed = load_speed()
    train_end = split_bounds(len(speed), cfg["split"])["train"][1]
    static = static_features(speed, load_adjacency()["adj"], train_end)
    scaler = load_scaler(protocol)
    for split in ("train", "val", "test"):
        with np.load(PROCESSED / protocol / f"{split}.npz") as z:
            F = build_features(z["X"], scaler, static)
            Y, M = flatten_targets(z["Y"], z["M_y"])
        p = _paths(split, protocol)
        p["F"].parent.mkdir(parents=True, exist_ok=True)
        np.save(p["F"], F)
        np.save(p["Y"], Y)
        np.save(p["M"], M)
        print(f"{split}: F{F.shape} Y{Y.shape}")
        del F, Y, M


def load_features(split: str, protocol: str = "strict") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Memory-mapped (F, Y, M): rows = windows * 207 sensors."""
    p = _paths(split, protocol)
    return tuple(np.load(p[k], mmap_mode="r") for k in ("F", "Y", "M"))


if __name__ == "__main__":
    build_and_cache()
