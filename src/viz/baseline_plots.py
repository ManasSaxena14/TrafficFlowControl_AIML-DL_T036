"""Phase 3 figures: XGBoost feature importance and the error-vs-horizon curve of every baseline."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from src.viz import style as S
from src.viz.eda_plots import save


def plot_xgb_importance(imp: pd.DataFrame, top: int = 12) -> plt.Figure:
    """Share of total gain per feature at 15 min and at 60 min (horizon 3 and 12)."""
    S.apply()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharex=True)
    for ax, h, color in zip(axes, (3, 12), (S.BLUE, S.ORANGE)):
        top_f = imp[h].sort_values(ascending=False).head(top)[::-1] * 100
        ax.barh(top_f.index, top_f.values, color=color)
        ax.set_title(f"{5 * h}-minute forecast")
        ax.set_xlabel("share of total gain (%)")
        ax.grid(axis="y", visible=False)
    fig.suptitle("XGBoost relies on a sensor's own current speed and the clock, not on anything about the network",
                 x=0.01, ha="left", fontsize=12, fontweight="semibold")
    fig.tight_layout()
    return fig


def plot_error_by_horizon(results: dict[str, pd.DataFrame], metric: str = "mae", split: str = "test") -> plt.Figure:
    """One line per model: masked error against forecast horizon (minutes)."""
    S.apply()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    colors = [S.MUTED, S.AXIS, S.YELLOW, S.BLUE, S.AQUA, S.ORANGE]
    for (label, df), c in zip(results.items(), colors):
        d = df[(df.split == split) & (df.horizon != "avg")]
        ax.plot(d.horizon.astype(int) * 5, d[metric], marker="o", ms=3.5, label=label, color=c)
    ax.set_xlabel("forecast horizon (min)")
    ax.set_ylabel(f"{metric.upper()} (mph)")
    ax.set_title(f"Baselines: {metric.upper()} by horizon ({split} block)")
    ax.legend(ncol=2)
    return fig


if __name__ == "__main__":
    from src.train.baselines import LADDER, RESULTS

    imp = pd.read_csv(RESULTS / "xgb_importance.csv", index_col="feature")
    imp.columns = imp.columns.astype(int)
    save(plot_xgb_importance(imp), "fig06_xgb_importance.png")
    res = {label: pd.read_csv(RESULTS / f"{k}.csv", dtype={"horizon": str}) for k, (_, label) in LADDER.items()
           if (RESULTS / f"{k}.csv").exists()}
    save(plot_error_by_horizon(res), "fig07_baseline_error_by_horizon.png")
