"""Rungs 3-4: XGBoost (one model per horizon, all sensors stacked) and a multi-output Random Forest.

Both are *independent* models: a row only contains one sensor's own history, time and static descriptors.
They never see a neighbour's speed, which is exactly the gap Phase 4 measures.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

from src.eval.metrics import masked_mae
from src.features.build import FEATURE_NAMES, HORIZON

N_SENSORS = 207
MAX_BIN = 128


def window_stride_rows(n_rows: int, stride: int, n_sensors: int = N_SENSORS) -> np.ndarray:
    """Row indices of every `stride`-th window (all 207 sensors of that window). Windows are 5 min apart, so
    neighbours are near-duplicates; thinning them costs little accuracy and keeps an 8 GB machine happy."""
    return (np.arange(0, n_rows // n_sensors, stride)[:, None] * n_sensors + np.arange(n_sensors)).ravel()


class XGBPerHorizon:
    """One XGBoost regressor per horizon h = 1..12 trained on all sensors stacked (sensor id is a feature).

    Masked targets get weight 0. Tuning and early stopping use the validation block only.
    """

    def __init__(self, stride: int = 2, n_estimators: int = 600, seed: int = 42):
        self.stride, self.n_estimators, self.seed = stride, n_estimators, seed
        self.params = {"max_depth": 8, "eta": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 1,
                       "tree_method": "hist", "max_bin": MAX_BIN, "eval_metric": "mae", "nthread": 8, "seed": seed}
        self.boosters: dict[int, xgb.Booster] = {}
        self.best_rounds: dict[int, int] = {}

    def _mats(self, train, val):
        (F, Y, M), (Fv, Yv, Mv) = train, val
        idx = window_stride_rows(len(F), self.stride)
        self.Ytr, self.Mtr = np.asarray(Y[idx]), np.asarray(M[idx])
        self.dtr = xgb.QuantileDMatrix(np.asarray(F[idx]), label=np.zeros(len(idx), np.float32), max_bin=MAX_BIN)
        self.Yv, self.Mv = np.asarray(Yv), np.asarray(Mv)
        self.dv = xgb.QuantileDMatrix(np.asarray(Fv), label=np.zeros(len(Fv), np.float32), ref=self.dtr, max_bin=MAX_BIN)

    def _set_horizon(self, h: int) -> None:
        for d, Y, M in ((self.dtr, self.Ytr, self.Mtr), (self.dv, self.Yv, self.Mv)):
            d.set_label(np.nan_to_num(Y[:, h - 1]))
            d.set_weight(M[:, h - 1].astype(np.float32))

    def tune(self, train, val, horizon: int = 6, grid=({"max_depth": 6}, {"max_depth": 8}, {"max_depth": 10})) -> pd.DataFrame:
        """Small honest search on ONE representative horizon, scored on val. Same budget is stated in the report."""
        self._mats(train, val)
        self._set_horizon(horizon)
        rows = []
        for g in grid:
            booster = xgb.train({**self.params, **g}, self.dtr, self.n_estimators, evals=[(self.dv, "val")],
                                early_stopping_rounds=50, verbose_eval=False)
            rows.append({**g, "val_mae": booster.best_score, "rounds": booster.best_iteration + 1})
        res = pd.DataFrame(rows).sort_values("val_mae")
        self.params["max_depth"] = int(res.iloc[0]["max_depth"])
        return res

    def fit(self, train, val, log=print, horizons=range(1, HORIZON + 1)) -> XGBPerHorizon:
        if not hasattr(self, "dtr"):
            self._mats(train, val)
        self.n_features = self.dtr.num_col()
        for h in horizons:
            t0 = time.time()
            self._set_horizon(h)
            b = xgb.train(self.params, self.dtr, self.n_estimators, evals=[(self.dv, "val")],
                          early_stopping_rounds=50, verbose_eval=False)
            self.boosters[h], self.best_rounds[h] = b, b.best_iteration + 1
            log(f"  xgb h={h:2d} ({5 * h:2d} min): {self.best_rounds[h]} trees, val MAE {b.best_score:.3f}, {time.time() - t0:.0f}s")
        return self

    def predict(self, F: np.ndarray, chunk: int = 700_000) -> np.ndarray:
        """Rows (len(F), 12); horizons that were not trained stay NaN."""
        out = np.full((len(F), HORIZON), np.nan, np.float32)
        for a in range(0, len(F), chunk):
            d = xgb.DMatrix(np.asarray(F[a:a + chunk]))
            for h, b in self.boosters.items():
                out[a:a + chunk, h - 1] = b.predict(d, iteration_range=(0, self.best_rounds[h]))
        return out

    def importance(self, names: list[str] | None = None) -> pd.DataFrame:
        """Gain-based importance per horizon, normalised to sum to 1 for each horizon."""
        names = names or FEATURE_NAMES
        cols = {}
        for h, b in self.boosters.items():
            g = b.get_score(importance_type="total_gain")
            v = np.array([g.get(f"f{i}", 0.0) for i in range(len(names))])
            cols[h] = v / v.sum()
        return pd.DataFrame(cols, index=names).rename_axis("feature")

    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        for h, b in self.boosters.items():
            b.save_model(folder / f"h{h:02d}.json")


class RandomForestMulti:
    """One forest predicting all 12 horizons at once. Trees cannot take NaN targets, so training rows with any masked
    target are dropped; evaluation still uses every test row with the mask."""

    def __init__(self, stride: int = 6, **kw):
        self.stride = stride
        self.params = {"n_estimators": 100, "max_depth": 20, "min_samples_leaf": 10, "max_samples": 0.3, "max_features": 0.5,
                       "n_jobs": -1, "random_state": 42, **kw}

    def fit(self, train) -> RandomForestMulti:
        F, Y, M = train
        idx = window_stride_rows(len(F), self.stride)
        keep = np.asarray(M[idx]).all(axis=1)
        idx = idx[keep]
        self.n_train_rows = len(idx)
        self.model = RandomForestRegressor(**self.params).fit(np.asarray(F[idx]), np.asarray(Y[idx]))
        return self

    def predict(self, F: np.ndarray, chunk: int = 200_000) -> np.ndarray:
        return np.concatenate([self.model.predict(np.asarray(F[a:a + chunk])) for a in range(0, len(F), chunk)]).astype(np.float32)


def val_mae(model, val) -> float:
    Fv, Yv, Mv = val
    return masked_mae(np.asarray(Yv), model.predict(Fv), np.asarray(Mv))
