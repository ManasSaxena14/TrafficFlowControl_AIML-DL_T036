"""Phase 1 figures. Each function draws one report figure and returns the matplotlib Figure."""
from __future__ import annotations

from pathlib import Path

import folium
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from src.viz import style as S

style = S
FIG_DIR = Path(__file__).resolve().parents[2] / "reports" / "figures"


def save(fig: plt.Figure, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path)
    return path


def _hour_axis(ax: plt.Axes) -> None:
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 3))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 3)])


# --------------------------------------------------------------------------- 1
def plot_missingness(speed: pd.DataFrame, by_day: pd.DataFrame, summary) -> plt.Figure:
    zero = speed.values == 0
    all_down = zero.all(axis=1)
    days = pd.Index(speed.index.date)
    net = pd.Series(np.where(all_down[:, None], zero, False).mean(axis=1), index=speed.index).groupby(days).mean() * 100
    iso = pd.Series(np.where(~all_down[:, None], zero, False).mean(axis=1), index=speed.index).groupby(days).mean() * 100

    order = by_day.mean(axis=1).sort_values(ascending=False).index
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 7), height_ratios=[3, 1.1], sharex=True,
                                  gridspec_kw={"hspace": 0.08})
    im = ax.imshow(by_day.loc[order].values * 100, aspect="auto", cmap=S.SEQ_BLUE, vmin=0, vmax=100,
                   interpolation="nearest", extent=[-0.5, by_day.shape[1] - 0.5, len(order), 0])
    ax.grid(False)
    ax.set_ylabel("Sensor (sorted by outage share)")
    ax.set_title(f"1 · Sensor outages: {summary.zero_pct:.1f}% of readings are 0 mph, "
                 f"and {summary.network_outage_pct:.1f} points of that are whole-network outages")
    cb = fig.colorbar(im, ax=[ax, ax2], fraction=0.025, pad=0.01)
    cb.set_label("% of the day at 0 mph", color=S.INK_2)
    cb.outline.set_visible(False)

    x = np.arange(len(net))
    ax2.bar(x, iso.values, width=0.9, color=S.BLUE, label="Single-sensor outage")
    ax2.bar(x, net.values, width=0.9, bottom=iso.values, color=S.ORANGE, label="Whole network down")
    ax2.set_ylabel("% readings = 0")
    ax2.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    ticks = np.arange(0, len(net), 14)
    ax2.set_xticks(ticks)
    ax2.set_xticklabels([pd.Timestamp(by_day.columns[t]).strftime("%b %d") for t in ticks])
    return fig


# --------------------------------------------------------------------------- 2
def plot_daily_profile(profile: pd.DataFrame, peaks: list[tuple[float, float, float]]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 4.6))
    for a, b, _ in peaks:
        ax.axvspan(a, b, color=S.GRID, alpha=0.7, lw=0)
        label = "AM peak" if a < 12 else "PM peak"
        ax.text((a + b) / 2, 68.6, f"{label} {_hhmm(a)}–{_hhmm(b)}", ha="center", va="top", color=S.INK_2, fontsize=9)
    ax.fill_between(profile.index, profile.weekday_p25, profile.weekday_p75, color=S.BLUE_LIGHT, alpha=0.6,
                    lw=0, label="Weekday, middle 50% of days")
    ax.plot(profile.index, profile.weekday, color=S.BLUE, label="Weekday mean")
    ax.plot(profile.index, profile.weekend, color=S.ORANGE, label="Weekend mean")
    _hour_axis(ax)
    ax.set_ylim(40, 69)
    ax.set_ylabel("Network mean speed (mph)")
    ax.set_title("2 · Daily profile: two weekday dips (AM and PM rush), weekends stay near free flow")
    ax.legend(loc="lower left", ncol=3)
    return fig


def _hhmm(h: float) -> str:
    m = int(round(h * 60))
    return f"{m // 60:02d}:{m % 60:02d}"


# --------------------------------------------------------------------------- 3
def plot_distribution(dist: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 4.2))
    bins = np.arange(0, 72, 1)
    ax.hist(dist["values"], bins=bins, color=S.BLUE, edgecolor=S.SURFACE, linewidth=0.6)
    ax.axvline(40, color=S.INK_2, lw=1, ls="--")
    ax.text(39, ax.get_ylim()[1] * 0.92, f"{dist['pct_lt_40']:.1f}% of readings below 40 mph\n"
            f"({dist['pct_lt_20']:.1f}% below 20 mph): the congestion tail",
            ha="right", va="top", color=S.INK_2, fontsize=9)
    ax.text(58, ax.get_ylim()[1] * 0.92, f"{dist['pct_ge_60']:.0f}% at or above 60 mph",
            ha="right", va="top", color=S.INK_2, fontsize=9)
    ax.set_xlim(0, 71)
    ax.set_xlabel("Speed (mph), outages removed")
    ax.set_ylabel("Readings")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v / 1e3:.0f}k"))
    ax.set_title(f"3 · Speed distribution: most mass near free flow (median {dist['median']:.0f} mph), long low-speed tail")
    return fig


# --------------------------------------------------------------------------- 4
def plot_graph(g: nx.DiGraph, loc: pd.DataFrame, stats: dict) -> plt.Figure:
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 5.2), width_ratios=[1.5, 1])
    lon, lat = loc.longitude.to_numpy(), loc.latitude.to_numpy()
    for u, v, w in g.edges(data="weight"):
        ax.plot([lon[u], lon[v]], [lat[u], lat[v]], color=S.MUTED, lw=0.4 + 1.2 * w, alpha=0.25 + 0.5 * w,
                solid_capstyle="round", zorder=1)
    deg = stats["in_degree"] + stats["out_degree"]
    ax.scatter(lon, lat, s=14 + deg * 2.2, c=S.BLUE, edgecolor=S.SURFACE, linewidth=1.2, zorder=2)
    ax.set_aspect(1 / np.cos(np.deg2rad(lat.mean())))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(False)
    ax.set_title(f"4a · {stats['nodes']} sensors, {stats['edges']} directed road links\n"
                 f"(marker size = degree; line weight = adjacency weight)")

    bins = np.arange(0, max(stats["in_degree"].max(), stats["out_degree"].max()) + 2) - 0.5
    hi, _ = np.histogram(stats["in_degree"], bins)
    ho, _ = np.histogram(stats["out_degree"], bins)
    x = (bins[:-1] + 0.5)
    ax2.bar(x - 0.2, hi, width=0.38, color=S.BLUE, label="In-degree (upstream neighbours)")
    ax2.bar(x + 0.2, ho, width=0.38, color=S.ORANGE, label="Out-degree (downstream neighbours)")
    ax2.set_xticks(range(0, int(x.max()) + 1, 2))
    ax2.set_xlabel("Neighbours per sensor (self-loops excluded)")
    ax2.set_ylabel("Sensors")
    ax2.legend(loc="upper right")
    ax2.set_title(f"4b · Degree distribution: mean {stats['mean_degree']:.1f}; "
                  f"only {stats['reciprocal_edge_pct']:.0f}% of links are two-way")
    fig.tight_layout()
    return fig


def graph_map_html(g: nx.DiGraph, loc: pd.DataFrame, sensor_ids: list[str], path: Path) -> Path:
    """Interactive folium map: sensors + directed edges. Opens in any browser."""
    m = folium.Map(location=[loc.latitude.mean(), loc.longitude.mean()], zoom_start=11, tiles="CartoDB positron")
    lat, lon = loc.latitude.to_numpy(), loc.longitude.to_numpy()
    for u, v, w in g.edges(data="weight"):
        folium.PolyLine([[lat[u], lon[u]], [lat[v], lon[v]]], weight=1 + 3 * w, color=S.MUTED, opacity=0.35 + 0.4 * w,
                        tooltip=f"{sensor_ids[u]} → {sensor_ids[v]} (w={w:.2f})").add_to(m)
    for i, sid in enumerate(sensor_ids):
        folium.CircleMarker([lat[i], lon[i]], radius=4, color=S.SURFACE, weight=1.5, fill=True, fill_color=S.BLUE,
                            fill_opacity=1, tooltip=f"sensor {sid} · in {g.in_degree(i)} · out {g.out_degree(i)}").add_to(m)
    m.save(str(path))
    return path


# --------------------------------------------------------------------------- 5
def plot_propagation(tab: pd.DataFrame, curves: np.ndarray, links: list[int], sensor_ids: list[str],
                     summary: dict, max_lag: int = 12) -> plt.Figure:
    """5a: lag curves for a few example links; 5b: best-lag histogram over all links; 5c: lag vs distance."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), width_ratios=[1.2, 1.1, 1])
    lags_min = np.arange(-max_lag, max_lag + 1) * 5

    ax = axes[0]
    for idx, color in zip(links, [S.ORANGE, S.BLUE, S.AQUA]):
        r = tab.loc[idx]
        # Stored curve is corr(up[t-k], down[t]); flip so k > 0 means the DOWNSTREAM sensor moves first.
        c = curves[idx][::-1]
        k = lags_min[np.nanargmax(c)]
        ax.plot(lags_min, c, color=color,
                label=f"{sensor_ids[int(r.up)]} to {sensor_ids[int(r.down)]} ({r.dist_km:.1f} km): peak at {k:+d} min")
        ax.plot([k], [np.nanmax(c)], "o", color=color, ms=7, mec=S.SURFACE, mew=1.5)
    ax.axvline(0, color=S.AXIS, lw=1)
    ax.set_xlabel("Lag k (min): corr(downstream[t − k], upstream[t]); k > 0 means downstream moves first")
    ax.set_ylabel("Correlation of de-seasonalised speed")
    ax.set_title("5a · Three strongly linked road pairs: the peak sits\nat k > 0, so the downstream sensor leads")
    ax.legend(loc="lower center", fontsize=8)

    ax = axes[1]
    c = tab[tab.clear]
    lead = -c.best_lag  # same convention as 5a: > 0 means the downstream sensor moves first
    counts = lead.value_counts().reindex(range(-max_lag + 1, max_lag), fill_value=0)
    colors = [S.ORANGE if k > 0 else S.BLUE if k < 0 else S.MUTED for k in counts.index]
    ax.bar(counts.index * 5, counts.values, width=4, color=colors)
    ax.set_xlabel("Best lag (min): > 0 downstream moves first, < 0 upstream moves first")
    ax.set_ylabel("Road links")
    ax.set_title(f"5b · Who slows first? Downstream on {summary['downstream_first']} links,\n"
                 f"upstream on {summary['upstream_first']}; {summary['same_step_pct']:.0f}% within the same 5 min")

    ax = axes[2]
    lagged = c[c.best_lag != 0].copy()
    bins = [0, 1.5, 2.5, 3.2, 4.0]
    lagged["bin"] = pd.cut(lagged.dist_km, bins)
    g = lagged.groupby("bin", observed=True).best_lag.agg(lambda x: x.abs().mean() * 5)
    n = lagged.groupby("bin", observed=True).size()
    labels = [f"{b.left:g}–{b.right:g} km" for b in g.index]
    ax.bar(labels, g.values, width=0.6, color=S.BLUE)
    for i, (v, cnt) in enumerate(zip(g.values, n.values)):
        ax.text(i, v + 0.3, f"{v:.0f} min\n(n={cnt})", ha="center", va="bottom", fontsize=9, color=S.INK_2)
    ax.set_ylim(0, g.max() * 1.35)
    ax.set_xlabel("Road distance between the two sensors")
    ax.set_ylabel("Mean |lag| (min)")
    ax.set_title(f"5c · Lag grows with distance:\nimplied backward wave ≈ {summary['shockwave_kmh']:.0f} km/h")
    fig.tight_layout()
    return fig


def plot_propagation_example(speed: pd.DataFrame, up: int, down: int, t_down: pd.Timestamp, t_up: pd.Timestamp,
                             sensor_ids: list[str], dist_km: float) -> plt.Figure:
    """Worked example: a jam appears at the downstream sensor first and reaches the upstream one later."""
    sl = speed.loc[t_down - pd.Timedelta(minutes=60): t_down + pd.Timedelta(minutes=90)].where(lambda x: x > 0)
    lag = int((t_up - t_down).total_seconds() // 60)
    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.plot(sl.index, sl.iloc[:, down], color=S.ORANGE, marker="o", ms=4, label=f"Downstream sensor {sensor_ids[down]}")
    ax.plot(sl.index, sl.iloc[:, up], color=S.BLUE, marker="o", ms=4,
            label=f"Upstream sensor {sensor_ids[up]} ({dist_km:.1f} km before it on the road)")
    for t, c, name, ha in [(t_down, S.ORANGE, "jam reaches downstream", "right"),
                           (t_up, S.BLUE, "jam reaches upstream", "left")]:
        ax.axvline(t, color=c, lw=1, ls="--")
        pad = " " if ha == "left" else ""
        ax.text(t, 2, f"{pad}{name} {t:%H:%M} {'' if pad else ' '}", color=S.INK_2, fontsize=9, va="bottom", ha=ha)
    ax.annotate("", xy=(t_up, 76), xytext=(t_down, 76), arrowprops=dict(arrowstyle="->", color=S.INK_2, lw=1.2))
    ax.text(t_up, 76, f"  {lag} min later, {dist_km:.1f} km upstream", ha="left", va="center", color=S.INK_2, fontsize=9)
    ax.set_ylim(0, 80)
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%H:%M"))
    ax.set_ylabel("Speed (mph)")
    ax.set_title(f"5d · One illustrative event, {t_down:%a %d %b %Y}: the jam hits the downstream sensor first, "
                 f"then travels back upstream")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 0.12))
    return fig
