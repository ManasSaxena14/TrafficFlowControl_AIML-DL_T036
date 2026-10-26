"""Phase 4 runner: the four pieces of evidence for the gap, written to results/gap/ and results/phase4_findings.json.

    python -m src.train.phase4 onset         # error at congestion onset vs normal periods (needs Phase 3 xgb)
    python -m src.train.phase4 residuals     # residual correlation along road links (needs Phase 3 xgb)
    python -m src.train.phase4 neighbours    # XGB + upstream / downstream / both neighbour features (trains models)
    python -m src.train.phase4 table         # gap_results.csv: XGB vs XGB + neighbours at 15 / 30 / 60 min
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.analysis import gap
from src.data.dataset import load_adjacency, load_split
from src.data.load import load_speed
from src.eval.evaluate import REPORT_HORIZONS, evaluate_rows
from src.features.build import FEATURE_NAMES, load_features
from src.models import trees
from src.train.baselines import PRED_DIR, RESULTS, ROOT

OUT = ROOT / "results" / "gap"
VARIANTS = {"upstream": ["up"], "downstream": ["down"], "both": ["up", "down"]}
HORIZONS = sorted(REPORT_HORIZONS)             # 3, 6, 12  = 15 / 30 / 60 min


def _findings(update: dict) -> None:
    f = ROOT / "results" / "phase4_findings.json"
    cur = json.loads(f.read_text()) if f.exists() else {}
    cur.update(update)
    f.write_text(json.dumps(cur, indent=2, default=float))


def _test_origins() -> np.ndarray:
    return load_split("test", keys=["origin_idx"])["origin_idx"]


def run_onset() -> pd.DataFrame:
    """Evidence 2: does the per-sensor model fail right after congestion starts?"""
    OUT.mkdir(parents=True, exist_ok=True)
    speed = load_speed()
    grid = gap.post_onset_grid(speed)
    labels = gap.onset_labels(_test_origins(), grid)
    _, Y, M = load_features("test")
    frames = []
    for model in [m for m in ("persistence", "xgb", "xgb_both") if (PRED_DIR / f"{m}_test.npy").exists()]:
        P = np.load(PRED_DIR / f"{model}_test.npy", mmap_mode="r")
        hs = [h for h in range(1, 13) if not np.isnan(np.asarray(P[:1000, h - 1])).all()]
        df = gap.onset_vs_normal(Y, P, M, labels, hs)
        df.insert(0, "model", model)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(OUT / "onset_vs_normal.csv", index=False)
    test_start = speed.index[int(_test_origins()[0]) + 1]
    n_onsets = sum(int((gap.eda.congestion_onsets(speed[c]) >= test_start).sum()) for c in speed.columns)
    on_share = 100 * (labels & np.asarray(M)).sum() / np.asarray(M).sum()
    _findings({"test_onsets": n_onsets, "onset_share_of_targets_pct": float(on_share)})
    x = out[out.model == "xgb"].pivot(index="horizon", columns="group", values="mae")
    print("XGB MAE at onset vs normal:\n", x.round(2).assign(ratio=(x.onset / x.normal).round(2)).loc[HORIZONS])
    return out


def run_residuals() -> pd.DataFrame:
    """Evidence 4: are one sensor's errors correlated with its neighbours' errors?"""
    adj = load_adjacency()["adj"]
    _, Y, M = load_features("test")
    frames = []
    for model in [m for m in ("xgb", "xgb_both") if (PRED_DIR / f"{m}_test.npy").exists()]:
        P = np.load(PRED_DIR / f"{model}_test.npy", mmap_mode="r")
        for h in HORIZONS:
            y, p, m = (np.asarray(a[:, h - 1]).reshape(-1, gap.N_SENSORS) for a in (Y, P, M))
            df = gap.residual_correlation_by_hop(np.nan_to_num(y) - p, m, adj)
            df.insert(0, "horizon", h)
            df.insert(0, "model", model)
            frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(OUT / "residual_correlation.csv", index=False)
    print(out[out.model == "xgb"].pivot(index="group", columns="horizon", values="mean_corr").round(3))
    return out


def run_neighbours() -> pd.DataFrame:
    """Evidence 3 (the key experiment): add hand-made neighbour features to XGBoost.

    Same hyper-parameters as the Phase 3 XGBoost (max depth taken from its validation tuning; no extra tuning for the
    neighbour models), same row thinning, same early stopping on val. Trained only for 15 / 30 / 60 min (cost).
    """
    OUT.mkdir(parents=True, exist_ok=True)
    adj = load_adjacency()["adj"]
    base = json.loads((RESULTS / "xgb.json").read_text())
    train, val, test = (load_features(s) for s in ("train", "val", "test"))
    idx = trees.window_stride_rows(len(train[0]), base["stride"])
    frames, importances = [], {}
    for name, dirs in VARIANTS.items():
        cols = gap.neighbour_column_names(dirs)
        extra = {s: gap.add_neighbour_columns(d[0], adj, dirs) for s, d in zip(("train", "val", "test"), (train, val, test))}
        Ftr = np.hstack([np.asarray(train[0][idx]), extra["train"][idx]])
        Fva = np.hstack([np.asarray(val[0]), extra["val"]])
        Fte = np.hstack([np.asarray(test[0]), extra["test"]])
        model = trees.XGBPerHorizon(stride=1)
        model.params["max_depth"] = base["params"]["max_depth"]
        print(f"[xgb + {name}] features {Ftr.shape[1]} (+{len(cols)})")
        model.fit((Ftr, np.asarray(train[1][idx]), np.asarray(train[2][idx])), (Fva, val[1], val[2]), horizons=HORIZONS)
        importances[name] = model.importance(FEATURE_NAMES + cols)[HORIZONS]
        for split, F, d in (("val", Fva, val), ("test", Fte, test)):
            P = model.predict(F)
            np.save(PRED_DIR / f"xgb_{name}_{split}.npy", P)
            if name == "both":
                np.save(PRED_DIR / f"xgb_both_{split}.npy", P)
            df = evaluate_rows(d[1], P, d[2], HORIZONS)
            df.insert(0, "split", split)
            df.insert(0, "model", f"xgb + {name}")
            frames.append(df)
        del Ftr, Fva, Fte, extra
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(OUT / "xgb_neighbours.csv", index=False)
    pd.concat(importances, axis=1).to_csv(OUT / "xgb_neighbours_importance.csv")
    return out


def make_table() -> pd.DataFrame:
    """results/gap_results.csv: XGBoost vs XGBoost + neighbours at 15 / 30 / 60 min (the bridge between baselines and GNNs)."""
    base = pd.read_csv(RESULTS / "xgb.csv", dtype={"horizon": str}).assign(model="xgb (no neighbours)")
    nb = pd.read_csv(OUT / "xgb_neighbours.csv", dtype={"horizon": str})
    df = pd.concat([base[base.horizon.isin([str(h) for h in HORIZONS])], nb[nb.horizon.isin([str(h) for h in HORIZONS])]])
    rows = []
    for (model, split), d in df.groupby(["model", "split"], sort=False):
        d = d.set_index("horizon")
        r = {"model": model, "split": split}
        for h, minutes in REPORT_HORIZONS.items():
            for m in ("mae", "rmse", "mape"):
                r[f"{m.upper()}_{minutes}min"] = round(float(d.loc[str(h), m]), 3)
        rows.append(r)
    t = pd.DataFrame(rows)
    ref = t[(t.model == "xgb (no neighbours)")].set_index("split")
    for minutes in REPORT_HORIZONS.values():
        col = f"MAE_{minutes}min"
        t[f"MAE_gain_{minutes}min_%"] = (100 * (1 - t[col] / t.split.map(ref[col]))).round(1)
    t.to_csv(ROOT / "results" / "gap_results.csv", index=False)
    print(t.to_string(index=False))
    return t


RUNNERS = {"onset": run_onset, "residuals": run_residuals, "neighbours": run_neighbours, "table": make_table}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("steps", nargs="+", choices=list(RUNNERS))
    for s in ap.parse_args().steps:
        RUNNERS[s]()


if __name__ == "__main__":
    main()
