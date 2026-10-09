"""Phase 1 exploratory analysis. Pure functions: data in, tables/arrays out. Plots live in src/viz/eda_plots.py.

Anything that could steer modelling choices (profiles, peaks, propagation lags) is computed on the
training period only (first 70% of timestamps), so the test period stays unseen until Phase 6.
"""
from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd

TRAIN_FRAC = 0.7
STEPS_PER_DAY = 288


def train_slice(speed: pd.DataFrame, frac: float = TRAIN_FRAC) -> pd.DataFrame:
    return speed.iloc[: int(round(len(speed) * frac))]


def masked(speed: pd.DataFrame) -> pd.DataFrame:
    """0 mph means 'sensor down', not 'traffic stopped': turn it into NaN."""
    return speed.where(speed > 0)


def slot_of_day(index: pd.DatetimeIndex) -> np.ndarray:
    return (index.hour * 12 + index.minute // 5).to_numpy()


# --------------------------------------------------------------------------- 1. Missingness
@dataclass
class MissingnessSummary:
    zero_pct: float                 # share of all readings that are 0
    network_outage_steps: int       # timestamps where every sensor reads 0
    network_outage_pct: float       # share of readings explained by those timestamps
    sensor_outage_pct: float        # share of readings that are isolated sensor outages
    worst_sensors: pd.Series        # top sensors by zero share
    sensors_over_20pct: int
    longest_network_outage_steps: int


def missingness(speed: pd.DataFrame) -> tuple[MissingnessSummary, pd.DataFrame]:
    """Return summary numbers and a (sensor x day) table of zero share."""
    zero = speed.values == 0
    all_down = zero.all(axis=1)
    runs = pd.Series(all_down).ne(pd.Series(all_down).shift()).cumsum()
    longest = int(pd.Series(all_down).groupby(runs).sum().max())
    per_sensor = pd.Series(zero.mean(axis=0), index=speed.columns)
    summary = MissingnessSummary(
        zero_pct=100 * zero.mean(),
        network_outage_steps=int(all_down.sum()),
        network_outage_pct=100 * all_down.mean(),
        sensor_outage_pct=100 * zero[~all_down].sum() / zero.size,
        worst_sensors=(100 * per_sensor).sort_values(ascending=False).head(10),
        sensors_over_20pct=int((per_sensor > 0.2).sum()),
        longest_network_outage_steps=longest,
    )
    by_day = pd.DataFrame(zero, index=speed.index, columns=speed.columns).groupby(speed.index.date).mean().T
    return summary, by_day


# --------------------------------------------------------------------------- 2. Daily profile
def daily_profile(speed: pd.DataFrame) -> pd.DataFrame:
    """Network-mean speed per 5-min slot, weekday vs weekend, ignoring outages. Index = hour (float)."""
    m = masked(speed)
    net = m.mean(axis=1)
    weekend = speed.index.dayofweek >= 5
    slot = slot_of_day(speed.index)
    out = pd.DataFrame({
        "weekday": net[~weekend].groupby(slot[~weekend]).mean(),
        "weekend": net[weekend].groupby(slot[weekend]).mean(),
        "weekday_p25": net[~weekend].groupby(slot[~weekend]).quantile(0.25),
        "weekday_p75": net[~weekend].groupby(slot[~weekend]).quantile(0.75),
    })
    out.index = out.index / 12.0
    return out


def peak_windows(profile: pd.Series, rel: float = 0.9) -> list[tuple[float, float, float]]:
    """Contiguous spans where weekday speed < rel x free-flow (95th pct of the profile).

    Returns (start_hour, end_hour, min_speed) for each span longer than 30 minutes.
    """
    thresh = rel * profile.quantile(0.95)
    below = (profile < thresh).to_numpy()
    spans, start = [], None
    for k, b in enumerate(np.append(below, False)):
        if b and start is None:
            start = k
        elif not b and start is not None:
            if k - start >= 6:
                seg = profile.iloc[start:k]
                spans.append((float(profile.index[start]), float(profile.index[k - 1] + 1 / 12), float(seg.min())))
            start = None
    return spans


# --------------------------------------------------------------------------- 3. Distribution
def speed_distribution(speed: pd.DataFrame) -> dict:
    v = speed.values[speed.values > 0]
    return {
        "values": v,
        "median": float(np.median(v)),
        "pct_ge_60": float(100 * (v >= 60).mean()),
        "pct_lt_40": float(100 * (v < 40).mean()),
        "pct_lt_20": float(100 * (v < 20).mean()),
    }


# --------------------------------------------------------------------------- 4. Graph
def build_digraph(adj: np.ndarray, sensor_ids: list[str]) -> nx.DiGraph:
    """Directed graph without self-loops; edge i -> j means traffic can drive from sensor i to sensor j."""
    g = nx.DiGraph()
    g.add_nodes_from(range(len(sensor_ids)))
    for i, j in np.argwhere(adj > 0):
        if i != j:
            g.add_edge(int(i), int(j), weight=float(adj[i, j]))
    return g


def graph_stats(g: nx.DiGraph) -> dict:
    ind = np.array([d for _, d in g.in_degree()])
    outd = np.array([d for _, d in g.out_degree()])
    reciprocal = sum(1 for u, v in g.edges if g.has_edge(v, u))
    return {
        "nodes": g.number_of_nodes(),
        "edges": g.number_of_edges(),
        "reciprocal_edge_pct": 100 * reciprocal / g.number_of_edges(),
        "in_degree": ind,
        "out_degree": outd,
        "mean_degree": float(ind.mean()),
        "isolated_nodes": int(((ind == 0) & (outd == 0)).sum()),
        "weak_components": nx.number_weakly_connected_components(g),
        "largest_weak_component": len(max(nx.weakly_connected_components(g), key=len)),
    }


# --------------------------------------------------------------------------- 5. Spatial correlation
def deseasonalised(speed: pd.DataFrame) -> np.ndarray:
    """Masked speed minus each sensor's (weekday/weekend x 5-min slot) mean.

    Raw speeds share the same daily rhythm, so every pair correlates at every lag. Removing that rhythm
    leaves the day-specific disturbances (incidents, jams), which is what actually propagates.
    """
    m = masked(speed)
    key = [(speed.index.dayofweek >= 5).astype(int), slot_of_day(speed.index)]
    return (m - m.groupby(key).transform("mean")).to_numpy(dtype=np.float64)


def lagged_corr(x: np.ndarray, y: np.ndarray, k: int) -> float:
    """corr(x[t-k], y[t]), ignoring NaNs. k > 0 means x leads y by k steps."""
    if k > 0:
        a, b = x[:-k], y[k:]
    elif k < 0:
        a, b = x[-k:], y[:k]
    else:
        a, b = x, y
    ok = ~np.isnan(a) & ~np.isnan(b)
    if ok.sum() < 100:
        return np.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def edge_lag_table(resid: np.ndarray, g: nx.DiGraph, max_lag: int = 12, min_corr: float = 0.3,
                   distances: pd.DataFrame | None = None, sensor_ids: list[str] | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    """For every directed edge (upstream u -> downstream d) compute corr(u[t-k], d[t]) for k in [-max_lag, max_lag].

    best_lag > 0: upstream moves first (disturbance travels with traffic).
    best_lag < 0: downstream moves first (congestion shockwave travelling backwards, against traffic).
    'coupled' edges co-move (peak corr >= min_corr); among them a lag is 'clear' if its peak is inside the
    window and either is lag 0 or beats lag 0 by >= 0.01. Weakly coupled edges have no meaningful lag.
    """
    lags = np.arange(-max_lag, max_lag + 1)
    edges = list(g.edges)
    curves = np.array([[lagged_corr(resid[:, u], resid[:, d], k) for k in lags] for u, d in edges])
    best = lags[np.nanargmax(curves, axis=1)]
    peak = np.nanmax(curves, axis=1)
    at0 = curves[:, max_lag]
    interior = np.abs(best) < max_lag
    tab = pd.DataFrame({"up": [u for u, _ in edges], "down": [d for _, d in edges],
                        "weight": [g[u][d]["weight"] for u, d in edges],
                        "corr_lag0": at0, "corr_peak": peak, "best_lag": best,
                        "coupled": peak >= min_corr})
    tab["clear"] = tab.coupled & interior & ((best == 0) | (peak - at0 >= 0.01))
    if distances is not None and sensor_ids is not None:
        dmap = {(f, t): c for f, t, c in distances[["from", "to", "cost"]].itertuples(index=False)}
        tab["dist_km"] = [dmap.get((sensor_ids[u], sensor_ids[d]), np.nan) / 1000 for u, d in edges]
    return tab, curves


def propagation_summary(tab: pd.DataFrame, step_min: int = 5) -> dict:
    """Headline numbers for the report: who moves first, how long it takes, implied shockwave speed."""
    c = tab[tab.clear]
    lagged = c[c.best_lag != 0]
    down_first = lagged[lagged.best_lag < 0]
    speed_kmh = (down_first.dist_km.sum() / (down_first.best_lag.abs().sum() * step_min / 60)
                 if "dist_km" in tab and len(down_first) else np.nan)
    return {
        "edges": len(tab),
        "weak_edge_pct": float(100 * (~tab.coupled).mean()),
        "clear_edges": len(c),
        "same_step_pct": float(100 * (c.best_lag == 0).mean()),
        "downstream_first": int((lagged.best_lag < 0).sum()),
        "upstream_first": int((lagged.best_lag > 0).sum()),
        "median_lag_min": float(lagged.best_lag.abs().median() * step_min),
        "shockwave_kmh": float(speed_kmh),
    }


def congestion_onsets(x: pd.Series, drop: float = 20, within: int = 3, before_min: float = 50) -> pd.DatetimeIndex:
    """Onset = speed falls by more than `drop` mph within `within` steps (15 min), starting from free-ish flow.

    Uses a 3-step rolling median so a single noisy reading cannot trigger an onset. Same definition as Phase 4.
    """
    sm = x.where(x > 0).rolling(3, center=True).median()
    hit = (sm.shift(within) >= before_min) & (sm <= sm.shift(within) - drop)
    first = hit & ~hit.shift(1, fill_value=False)
    return sm.index[first.to_numpy()]


def pick_example(tab: pd.DataFrame, speed: pd.DataFrame) -> tuple[pd.Series, pd.Timestamp, pd.Timestamp]:
    """Pick one clean, illustrative event for the worked-example figure.

    Among clearly downstream-first links, find congestion onsets at the downstream sensor that are followed,
    5-30 min later, by an onset at the upstream sensor (with no outages around the event). Score by the size
    of both drops. This is an illustration only; the evidence is the all-links statistics in figure 5b/5c.
    """
    cand = tab[tab.clear & (tab.best_lag < 0)].sort_values("corr_peak", ascending=False).head(40)
    best = (-np.inf, None, None, None)
    for _, r in cand.iterrows():
        u, d = int(r.up), int(r.down)
        su, sd = speed.iloc[:, u], speed.iloc[:, d]
        up_onsets = congestion_onsets(su)
        for t in congestion_onsets(sd):
            if t.dayofweek >= 5:
                continue
            nxt = up_onsets[(up_onsets > t) & (up_onsets <= t + pd.Timedelta(minutes=30))]
            if len(nxt) == 0:
                continue
            win = slice(t - pd.Timedelta(minutes=60), t + pd.Timedelta(minutes=90))
            if (su[win] == 0).any() or (sd[win] == 0).any():
                continue
            pre = slice(t - pd.Timedelta(minutes=45), t - pd.Timedelta(minutes=10))
            if min(su[pre].mean(), sd[pre].mean()) < 55:
                continue  # both sensors must be flowing freely before the event
            score = (sd[pre].mean() - sd[win].min()) + (su[pre].mean() - su[win].min())
            if score > best[0]:
                best = (score, r, t, nxt[0])
    return best[1], best[2], best[3]


def pick_congested_sensor(speed: pd.DataFrame, g: nx.DiGraph) -> int:
    """Sensor with the lowest weekday PM-peak speed relative to its own free flow, among sensors with >= 2 in- and out-neighbours."""
    m = masked(speed)
    wd = speed.index.dayofweek < 5
    pm = wd & (speed.index.hour >= 16) & (speed.index.hour < 20)
    ratio = m[pm].mean() / m.quantile(0.85)
    ok = [i for i in range(speed.shape[1]) if g.in_degree(i) >= 2 and g.out_degree(i) >= 2]
    return ok[int(np.argmin(ratio.iloc[ok].to_numpy()))]
