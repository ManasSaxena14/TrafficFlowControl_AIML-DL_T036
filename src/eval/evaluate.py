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


def evaluate_rows(Y: np.ndarray, P: np.ndarray, M: np.ndarray) -> pd.DataFrame:
    """Rows 'horizon' = '1'..'12' (5 min each) plus 'avg' (pooled over all 12). Masked targets never count."""
    rows = []
    for h in HORIZONS:
        y, p, m = np.asarray(Y[:, h - 1]), np.asarray(P[:, h - 1]), np.asarray(M[:, h - 1])
        rows.append((str(h), masked_mae(y, p, m), masked_rmse(y, p, m), masked_mape(y, p, m)))
    y, p, m = np.asarray(Y), np.asarray(P), np.asarray(M)
    rows.append(("avg", masked_mae(y, p, m), masked_rmse(y, p, m), masked_mape(y, p, m)))
    return pd.DataFrame(rows, columns=["horizon", "mae", "rmse", "mape"])
