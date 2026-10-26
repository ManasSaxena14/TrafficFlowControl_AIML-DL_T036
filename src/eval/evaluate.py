"""One evaluation path for every model: masked MAE / RMSE / MAPE in mph, per forecast horizon.

Predictions and targets are in the Phase 3 "row" layout (rows = windows * sensors, columns = horizons 1..12).
Peak / off-peak and significance tests are added in Phase 6 on top of this.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.eval.metrics import masked_mae, masked_mape, masked_rmse

HORIZONS = list(range(1, 13))
REPORT_HORIZONS = {3: 15, 6: 30, 12: 60}     # horizon step -> minutes


def evaluate_rows(Y: np.ndarray, P: np.ndarray, M: np.ndarray, horizons: list[int] = HORIZONS) -> pd.DataFrame:
    """Rows 'horizon' = '1'..'12' (5 min each) plus 'avg' (pooled over the evaluated horizons). Masked targets never count."""
    rows = []
    cols = [h - 1 for h in horizons]
    for h in horizons:
        y, p, m = np.asarray(Y[:, h - 1]), np.asarray(P[:, h - 1]), np.asarray(M[:, h - 1])
        rows.append((str(h), masked_mae(y, p, m), masked_rmse(y, p, m), masked_mape(y, p, m)))
    y, p, m = np.asarray(Y)[:, cols], np.asarray(P)[:, cols], np.asarray(M)[:, cols]
    rows.append(("avg", masked_mae(y, p, m), masked_rmse(y, p, m), masked_mape(y, p, m)))
    return pd.DataFrame(rows, columns=["horizon", "mae", "rmse", "mape"])


def period_labels(target_time: np.ndarray, peaks: dict, n_sensors: int = 207) -> np.ndarray:
    """(S, 12) target timestamps -> (S * N, 12) labels in row layout: 'peak' (weekday AM/PM peak, judged on the
    timestamp being predicted), 'offpeak' (other weekday times) or 'weekend'. Peak windows come from the config."""
    idx = pd.DatetimeIndex(target_time.ravel())
    minutes = idx.hour * 60 + idx.minute

    def within(window):
        (h0, m0), (h1, m1) = (map(int, t.split(":")) for t in window)
        return (minutes >= h0 * 60 + m0) & (minutes <= h1 * 60 + m1)

    weekend = idx.dayofweek >= 5
    peak = ~weekend & (within(peaks["am"]) | within(peaks["pm"]))
    lab = np.where(weekend, "weekend", np.where(peak, "peak", "offpeak")).reshape(target_time.shape)
    return np.repeat(lab[:, None, :], n_sensors, axis=1).reshape(-1, target_time.shape[1])


def evaluate_by_period(Y: np.ndarray, P: np.ndarray, M: np.ndarray, labels: np.ndarray,
                       horizons: list[int] = HORIZONS) -> pd.DataFrame:
    """Masked MAE / RMSE / MAPE per period ('peak', 'offpeak', 'weekend') and horizon."""
    rows = []
    for h in horizons:
        y, p, m, lab = (np.asarray(a[:, h - 1]) for a in (Y, P, M, labels))
        for period in ("peak", "offpeak", "weekend"):
            mm = m & (lab == period)
            rows.append((period, str(h), masked_mae(y, p, mm), masked_rmse(y, p, mm), masked_mape(y, p, mm), int(mm.sum())))
    return pd.DataFrame(rows, columns=["period", "horizon", "mae", "rmse", "mape", "n"])
