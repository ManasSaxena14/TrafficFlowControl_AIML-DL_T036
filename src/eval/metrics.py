"""Masked metrics, in mph. The same functions score every model (extended by horizon/peak in Phase 6).

Outage points (mask False) contribute exactly nothing: not to the numerator, not to the count.
"""
from __future__ import annotations

import numpy as np


def _prep(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray):
    m = mask.astype(bool)
    return np.where(m, y_true, 0.0), np.where(m, y_pred, 0.0), m


def masked_mae(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> float:
    yt, yp, m = _prep(y_true, y_pred, mask)
    return float(np.abs(yp - yt)[m].sum() / m.sum())


def masked_rmse(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> float:
    yt, yp, m = _prep(y_true, y_pred, mask)
    return float(np.sqrt(np.square(yp - yt)[m].sum() / m.sum()))


def masked_mape(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray, eps: float = 1.0) -> float:
    """In percent. Targets are real speeds (> 0 by construction); eps only guards a pathological near-zero reading."""
    yt, yp, m = _prep(y_true, y_pred, mask)
    return float(100 * (np.abs(yp - yt) / np.maximum(np.abs(yt), eps))[m].sum() / m.sum())
