"""Rungs 0-2 of the baseline ladder: persistence, historical average, Ridge per sensor."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.eval.metrics import masked_mae
from src.features.build import DYNAMIC, HORIZON


def persistence(F: np.ndarray) -> np.ndarray:
    """Rung 0: the last (imputed) speed, repeated for all 12 horizons. The floor everything must beat."""
    return np.repeat(np.asarray(F[:, :1]), HORIZON, axis=1)


class HistoricalAverage:
    """Rung 1: mean speed per (weekday/weekend, 5-min slot, sensor), learned on train timestamps only."""

    def fit(self, speed: pd.DataFrame, train_end: int) -> HistoricalAverage:
        tr = speed.iloc[:train_end]
        v = tr.values.astype("float32")
        v = np.where(v > 0, v, np.nan)                                   # outages are not traffic
        slot = (tr.index.hour * 12 + tr.index.minute // 5).to_numpy()
        weekend = (tr.index.dayofweek >= 5).astype(int)
        self.table = np.empty((2, 288, v.shape[1]), np.float32)
        overall = np.nanmean(v, axis=0)
        for w in (0, 1):
            t = pd.DataFrame(v[weekend == w]).groupby(slot[weekend == w]).mean().reindex(range(288)).to_numpy()
            self.table[w] = np.where(np.isnan(t), overall[None], t)
        return self

    def predict(self, target_time: np.ndarray) -> np.ndarray:
        """target_time (S, 12) datetime64 -> rows (S * N, 12)."""
        idx = pd.DatetimeIndex(target_time.ravel())
        slot = (idx.hour * 12 + idx.minute // 5).to_numpy()
        weekend = (idx.dayofweek >= 5).astype(int)
        S, H = target_time.shape
        p = self.table[weekend, slot].reshape(S, H, -1)                  # (S, 12, N)
        return p.transpose(0, 2, 1).reshape(S * self.table.shape[2], H)


class RidgePerSensor:
    """Rung 2: one linear model per sensor and horizon on that sensor's own time-varying features."""

    def __init__(self, n_sensors: int = 207, alphas: tuple[float, ...] = (1.0, 10.0, 100.0, 1000.0)):
        self.N, self.alphas = n_sensors, alphas

    def _dyn(self, F: np.ndarray) -> np.ndarray:
        return F.reshape(-1, self.N, F.shape[1])[:, :, : len(DYNAMIC)]

    def _fit_alpha(self, F3, Y3, M3, alpha):
        d = F3.shape[2]
        coef = np.zeros((self.N, HORIZON, d), np.float32)
        icpt = np.zeros((self.N, HORIZON), np.float32)
        for n in range(self.N):
            Xn = np.asarray(F3[:, n, :])
            Xs = (Xn - self.mu[n]) / self.sd[n]
            for h in range(HORIZON):
                ok = np.asarray(M3[:, n, h])
                r = Ridge(alpha=alpha).fit(Xs[ok], np.asarray(Y3[:, n, h])[ok])
                coef[n, h], icpt[n, h] = r.coef_, r.intercept_
        return coef, icpt

    def fit(self, F, Y, M, Fv, Yv, Mv) -> RidgePerSensor:
        """Scaling uses train rows only; alpha is picked on the validation block only."""
        F3 = self._dyn(F)
        Y3, M3 = Y.reshape(-1, self.N, HORIZON), M.reshape(-1, self.N, HORIZON)
        self.mu = np.stack([np.asarray(F3[:, n, :]).mean(0) for n in range(self.N)])
        self.sd = np.stack([np.asarray(F3[:, n, :]).std(0) for n in range(self.N)]) + 1e-6
        best = (np.inf, None)
        self.val_mae = {}
        for a in self.alphas:
            self.coef, self.icpt = self._fit_alpha(F3, Y3, M3, a)
            self.val_mae[a] = masked_mae(np.asarray(Yv), self.predict(Fv), np.asarray(Mv))
            if self.val_mae[a] < best[0]:
                best = (self.val_mae[a], (a, self.coef, self.icpt))
        self.alpha, self.coef, self.icpt = best[1]
        return self

    def predict(self, F: np.ndarray) -> np.ndarray:
        Xs = (self._dyn(F) - self.mu[None]) / self.sd[None]              # (S, N, d)
        P = np.einsum("snd,nhd->snh", Xs, self.coef) + self.icpt[None]
        return P.reshape(-1, HORIZON).astype(np.float32)
