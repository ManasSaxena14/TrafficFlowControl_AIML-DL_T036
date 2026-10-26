"""Phase 4 figures: the four pieces of evidence that per-sensor models miss spatial propagation."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.viz import style as S
from src.viz.eda_plots import save


def plot_error_vs_horizon(xgb: pd.DataFrame, persistence: pd.DataFrame, split: str = "test") -> plt.Figure:
    """Gap 1: XGBoost error grows steadily with horizon; the jam that will arrive is currently at a neighbour."""
    S.apply()
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for df, label, color, ls in ((xgb, "XGBoost MAE", S.BLUE, "-"), (xgb, "XGBoost RMSE", S.BLUE, "--"),
                                 (persistence, "Persistence MAE", S.MUTED, "-")):
        d = df[(df.split == split) & (df.horizon != "avg")]
        col = "rmse" if "RMSE" in label else "mae"
        ax.plot(d.horizon.astype(int) * 5, d[col], ls, marker="o", ms=3.5, color=color, label=label)
    ax.set_xlabel("forecast horizon (min)")
    ax.set_ylabel("error (mph)")
    ax.set_title("Gap 1 · The best per-sensor model degrades steadily with horizon")
    ax.legend()
    return fig


def plot_onset_error(onset: pd.DataFrame) -> plt.Figure:
    """Gap 2: error in the 30 min after a congestion onset vs all other targets, per horizon."""
    S.apply()
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for model, label, color in (("xgb", "XGBoost", S.BLUE), ("persistence", "Persistence", S.MUTED)):
        d = onset[onset.model == model].pivot(index="horizon", columns="group", values="mae")
        if d.empty:
            continue
        ax.plot(d.index * 5, d["onset"], marker="o", ms=3.5, color=color, label=f"{label}: after an onset")
        ax.plot(d.index * 5, d["normal"], "--", color=color, label=f"{label}: other times")
    nb = onset[onset.model == "xgb_both"].pivot(index="horizon", columns="group", values="mae")
    if not nb.empty:
        ax.scatter(nb.index * 5, nb["onset"], color=S.ORANGE, zorder=3, s=36, label="XGBoost + neighbours: after an onset")
    ax.set_xlabel("forecast horizon (min)")
    ax.set_ylabel("MAE (mph)")
    ax.set_title("Gap 2 · Errors are several times larger right after congestion starts")
    ax.legend(fontsize=8)
    return fig


def plot_neighbour_test(table: pd.DataFrame, split: str = "test") -> plt.Figure:
    """Gap 3: MAE at 15 / 30 / 60 min for XGBoost with no / upstream / downstream / both-direction neighbour features."""
    S.apply()
    t = table[table.split == split].set_index("model")
    order = ["xgb (no neighbours)", "xgb + upstream", "xgb + downstream", "xgb + both"]
    colors = [S.MUTED, S.AQUA, S.YELLOW, S.ORANGE]
    mins = [15, 30, 60]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    width = 0.2
    x = np.arange(len(mins))
    for k, (m, c) in enumerate(zip(order, colors)):
        if m not in t.index:
            continue
        vals = [t.loc[m, f"MAE_{mm}min"] for mm in mins]
        bars = ax.bar(x + (k - 1.5) * width, vals, width, color=c, label=m)
        if k:
            for b, mm in zip(bars, mins):
                g = t.loc[m, f"MAE_gain_{mm}min_%"]
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.03, f"{-g:+.1f}%", ha="center", fontsize=7, color=S.INK_2)
    ax.set_xticks(x, [f"{m} min" for m in mins])
    ax.set_ylabel("MAE (mph)")
    ax.set_title("Gap 3 · Neighbours' speeds lower XGBoost's error at every horizon (labels: MAE change)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="x", visible=False)
    return fig


def plot_residual_correlation(res: pd.DataFrame, model: str = "xgb") -> plt.Figure:
    """Gap 4: mean correlation between two sensors' XGBoost errors, by road-graph distance."""
    S.apply()
    fig, ax = plt.subplots(figsize=(8, 4.6))
    d = res[res.model == model]
    groups = list(dict.fromkeys(d.group))
    x = np.arange(len(groups))
    for k, (h, c) in enumerate(zip(sorted(d.horizon.unique()), (S.BLUE_LIGHT, S.BLUE, "#184f95"))):
        v = d[d.horizon == h].set_index("group").loc[groups, "mean_corr"]
        ax.bar(x + (k - 1) * 0.26, v, 0.26, color=c, label=f"{5 * h}-min forecast errors")
    ax.set_xticks(x, groups)
    ax.set_ylabel("mean residual correlation")
    ax.set_title("Gap 4 · Errors of linked sensors move together: a shared spatial cause the model ignores")
    ax.legend()
    ax.grid(axis="x", visible=False)
    return fig


if __name__ == "__main__":
    from src.train.baselines import RESULTS
    from src.train.phase4 import OUT, ROOT

    def read(name):
        return pd.read_csv(RESULTS / f"{name}.csv", dtype={"horizon": str})

    save(plot_error_vs_horizon(read("xgb"), read("persistence")), "fig08_gap_error_vs_horizon.png")
    save(plot_onset_error(pd.read_csv(OUT / "onset_vs_normal.csv")), "fig09_gap_onset_error.png")
    save(plot_neighbour_test(pd.read_csv(ROOT / "results" / "gap_results.csv")), "fig10_gap_neighbour_test.png")
    save(plot_residual_correlation(pd.read_csv(OUT / "residual_correlation.csv")), "fig11_gap_residual_correlation.png")
