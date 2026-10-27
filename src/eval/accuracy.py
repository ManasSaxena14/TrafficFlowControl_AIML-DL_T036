"""Accuracy-style summaries for the speed forecasts (a regression task has no single "accuracy").

Two plain-language scores per model and horizon on the test block, over real readings only:
  * within_5mph_pct : share of forecasts within 5 mph of the true speed
  * mape_accuracy_pct: 100 - MAPE (the usual shorthand for "percent accuracy" of a regression forecast)

Run: python -m src.eval.accuracy   (needs the predictions saved by `make baselines` and `make gap`)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.features.build import load_features
from src.train.baselines import LADDER, PRED_DIR

ROOT = Path(__file__).resolve().parents[2]
HORIZONS = {3: 15, 6: 30, 12: 60}
MODELS = {k: label for k, (_, label) in LADDER.items()} | {"xgb_both": "XGBoost + neighbours (both directions)"}


def within_tolerance(y: np.ndarray, p: np.ndarray, m: np.ndarray, tol: float = 5.0) -> float:
    m = m.astype(bool)
    return float(100 * (np.abs(np.where(m, y, 0) - np.where(m, p, 0)) <= tol)[m].mean())


def main() -> None:
    _, Y, M = load_features("test")
    rows = []
    for key, label in MODELS.items():
        f = PRED_DIR / f"{key}_test.npy"
        if not f.exists():
            continue
        P = np.load(f, mmap_mode="r")
        for h, minutes in HORIZONS.items():
            y, p, m = np.asarray(Y[:, h - 1]), np.asarray(P[:, h - 1]), np.asarray(M[:, h - 1])
            mape = 100 * float((np.abs(np.where(m, p, 0) - np.where(m, y, 0)) / np.maximum(np.abs(np.where(m, y, 0)), 1.0))[m].mean())
            rows.append({"model": label, "minutes": minutes, "within_5mph_pct": round(within_tolerance(y, p, m), 1),
                         "mape_accuracy_pct": round(100 - mape, 1), "targets": int(m.sum())})
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "accuracy.csv", index=False)
    print(out.pivot(index="model", columns="minutes", values=["within_5mph_pct", "mape_accuracy_pct"]).to_string())


if __name__ == "__main__":
    main()
