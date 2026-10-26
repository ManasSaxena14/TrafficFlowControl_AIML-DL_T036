"""Phase 4: measure, don't assert, that independent per-sensor models miss spatial propagation.

Pure functions. Rows follow the Phase 3 layout (row = window * 207 + sensor, columns = horizons 1..12).
Direction convention (Phase 1): adj[i, j] > 0 means a road path from sensor i to sensor j, so j is DOWNSTREAM of i.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import shortest_path

from src.analysis import eda
from src.eval.metrics import masked_mae

N_SENSORS = 207
NEIGHBOUR_FEATURES = ["mean_t", "mean_t-1", "mean_t-3", "mean_t-6", "min_t", "worst_drop_3"]


# --------------------------------------------------------------------------- 3. Neighbour features
def _offdiag(adj: np.ndarray) -> np.ndarray:
    return np.where(~np.eye(len(adj), dtype=bool), adj, 0.0)


def neighbour_weights(adj: np.ndarray, direction: str) -> np.ndarray:
    """(N, N) matrix whose row i holds sensor i's neighbours in the given direction (edge weights, self excluded)."""
    a = _offdiag(adj)
    if direction == "down":      # j is downstream of i  <=>  adj[i, j] > 0
        return a
    if direction == "up":        # j is upstream of i    <=>  adj[j, i] > 0
        return a.T
    raise ValueError(direction)


def neighbour_index(w: np.ndarray) -> np.ndarray:
    """(N, K) neighbour ids per sensor, -1 padded."""
    k = int((w > 0).sum(axis=1).max())
    idx = np.full((len(w), k), -1, np.int64)
    for i in range(len(w)):
        nb = np.flatnonzero(w[i] > 0)
        idx[i, :len(nb)] = nb
    return idx


def neighbour_features(lags: np.ndarray, w: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """lags (s, N, 12) with index k = speed at t-k  ->  (s, N, 6) summaries of each sensor's neighbours in one direction.

    weighted mean speed at t, t-1, t-3, t-6; slowest neighbour at t; biggest 15-min drop among neighbours.
    Sensors with no neighbour in that direction get NaN (XGBoost routes NaN natively).
    """
    deg = w.sum(axis=1)
    wn = np.divide(w, deg[:, None], out=np.zeros_like(w), where=deg[:, None] > 0)
    out = np.empty(lags.shape[:2] + (6,), np.float32)
    for c, k in enumerate((0, 1, 3, 6)):
        out[..., c] = lags[..., k] @ wn.T
    valid = idx >= 0
    g = np.where(valid[None], lags[:, np.clip(idx, 0, None), 0], np.nan)                       # (s, N, K) speed at t
    drop = g - np.where(valid[None], lags[:, np.clip(idx, 0, None), 3], np.nan)                # speed_t - speed_(t-3)
    has = valid.any(axis=1)
    out[..., 4] = np.where(has, np.nanmin(np.where(valid[None], g, np.inf), axis=2), np.nan)
    out[..., 5] = np.where(has, np.nanmin(np.where(valid[None], drop, np.inf), axis=2), np.nan)
    out[:, ~has, :] = np.nan
    return out


def add_neighbour_columns(F: np.ndarray, adj: np.ndarray, directions: list[str], chunk: int = 256) -> np.ndarray:
    """Rows (S*N, 12 lag columns first) -> (S*N, 6 * len(directions)) neighbour features, same row order."""
    S = len(F) // N_SENSORS
    mats = {d: neighbour_weights(adj, d) for d in directions}
    idxs = {d: neighbour_index(m) for d, m in mats.items()}
    out = np.empty((S, N_SENSORS, 6 * len(directions)), np.float32)
    for a in range(0, S, chunk):
        lags = np.asarray(F[a * N_SENSORS:min(a + chunk, S) * N_SENSORS, :12]).reshape(-1, N_SENSORS, 12)
        out[a:a + chunk] = np.concatenate([neighbour_features(lags, mats[d], idxs[d]) for d in directions], axis=-1)
    return out.reshape(S * N_SENSORS, -1)


def neighbour_column_names(directions: list[str]) -> list[str]:
    return [f"{d}stream_{n}" for d in directions for n in NEIGHBOUR_FEATURES]


# --------------------------------------------------------------------------- 2. Error at congestion onset
def post_onset_grid(speed: pd.DataFrame, minutes: int = 30) -> np.ndarray:
    """(T, N) bool: True where the timestamp lies within `minutes` after a congestion onset at that sensor.

    Onsets use the Phase 1 definition (`eda.congestion_onsets`: > 20 mph drop within 15 min from >= 50 mph).
    The grid only LABELS evaluation points; it is never a model input.
    """
    steps = minutes // 5
    grid = np.zeros(speed.shape, bool)
    pos = {t: i for i, t in enumerate(speed.index)}
    for j, col in enumerate(speed.columns):
        for t in eda.congestion_onsets(speed[col]):
            i = pos[t]
            grid[i:i + steps + 1, j] = True
    return grid


def onset_labels(origin_idx: np.ndarray, grid: np.ndarray, horizon: int = 12) -> np.ndarray:
    """(S*N, 12) bool in row layout: is the TARGET (origin + h) within 30 min after an onset at that sensor?"""
    t = origin_idx[:, None] + np.arange(1, horizon + 1)                      # (S, 12)
    lab = grid[t]                                                           # (S, 12, N)
    return lab.transpose(0, 2, 1).reshape(-1, horizon)


def onset_vs_normal(Y, P, M, labels: np.ndarray, horizons=range(1, 13)) -> pd.DataFrame:
    """Masked MAE of one model at onset-affected targets vs all other targets, per horizon."""
    rows = []
    for h in horizons:
        y, p, m, lab = (np.asarray(a[:, h - 1]) for a in (Y, P, M, labels))
        for name, sel in (("onset", lab), ("normal", ~lab)):
            mm = m & sel
            rows.append({"horizon": h, "group": name, "mae": masked_mae(y, p, mm), "n": int(mm.sum())})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- 4. Residual correlation
def masked_corr_matrix(E: np.ndarray, M: np.ndarray) -> np.ndarray:
    """(N, N) Pearson correlation of residual series using only the timestamps where BOTH sensors are real."""
    m = M.astype(np.float64)
    e = np.where(M, E, 0.0).astype(np.float64)
    n = m.T @ m
    sx, sxy, sxx = e.T @ m, e.T @ e, (e ** 2).T @ m
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = sxy / n - sx * sx.T / n ** 2
        var = sxx / n - (sx / n) ** 2
        return cov / np.sqrt(var * var.T)


def hop_matrix(adj: np.ndarray) -> np.ndarray:
    """Undirected hop count between sensors on the road graph (inf = not connected)."""
    return shortest_path(_offdiag(adj) > 0, directed=False, unweighted=True)


def residual_correlation_by_hop(E: np.ndarray, M: np.ndarray, adj: np.ndarray) -> pd.DataFrame:
    """Mean residual correlation of sensor pairs grouped by how many road links separate them."""
    C, H = masked_corr_matrix(E, M), hop_matrix(adj)
    iu = np.triu_indices(len(adj), 1)
    c, h = C[iu], H[iu]
    ok = np.isfinite(c)
    groups = {"1 link": h == 1, "2 links": h == 2, "3 links": h == 3,
              "4+ links": (h >= 4) & np.isfinite(h), "not connected": ~np.isfinite(h)}
    rows = [{"group": k, "pairs": int((g & ok).sum()), "mean_corr": float(c[g & ok].mean())} for k, g in groups.items()]
    return pd.DataFrame(rows)
