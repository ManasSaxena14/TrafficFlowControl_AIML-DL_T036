# Phase 4 report: identify the gap with evidence

**Status:** 🟡 evidence 1, 2 and 4 complete; evidence 3 (XGB + neighbours) training · **Numbers:** `results/gap/`, [`results/phase4_findings.json`](../../results/phase4_findings.json) · **Code:** `src/analysis/gap.py`, `src/train/phase4.py`, `src/viz/gap_plots.py` · **Tests:** `tests/test_gap.py`

Re-run with `make gap`.

The question: an independent per-sensor model (Phase 3's best, XGBoost) only sees its own sensor's history. Does it fail **exactly where congestion propagates between sensors**? Everything below uses the **test** block and the XGBoost predictions saved in Phase 3.

## Checklist (from the playbook)
- [x] 1 · Error vs horizon
- [x] 2 · Error at congestion onset vs normal periods
- [ ] 3 · Neighbour-information test: XGB + upstream / downstream / both (training)
- [x] 4 · Residual correlation along road links
- [ ] "XGB + neighbours" rows in `results/gap_results.csv`

## 1 · Error grows steadily with horizon
![error vs horizon](../../reports/figures/fig08_gap_error_vs_horizon.png)

XGBoost MAE 3.02 → 3.51 → 4.01 mph and RMSE 5.47 → 6.41 → 7.28 at 15 / 30 / 60 min. Beyond ~15 min the jam that will arrive is not yet visible in the sensor's own history; it is sitting at a neighbour.

## 2 · Errors explode right after congestion starts
![onset](../../reports/figures/fig09_gap_onset_error.png)

Onset = the Phase 1 definition (speed drops > 20 mph within 15 min from ≥ 50 mph; 3-step rolling median). A target counts as "after an onset" if it lies within 30 min after one at that sensor. The test block holds **3,911 onsets**; these targets are 2.2% of all real targets.

| Horizon | XGB MAE after onset | XGB MAE other times | Ratio |
|---|---|---|---|
| 15 min | 16.20 | 2.72 | 6.0× |
| 30 min | 20.12 | 3.14 | 6.4× |
| 60 min | 19.09 | 3.67 | 5.2× |

The model reacts late: it keeps forecasting free flow until the slowdown shows up in its own lags. Persistence fails the same way (and worse), so the issue is the information available, not the learner.

## 4 · Errors of linked sensors move together
![residual correlation](../../reports/figures/fig11_gap_residual_correlation.png)

For each pair of sensors, the correlation of their XGBoost residual series (only timestamps where both are real), grouped by road-graph distance:

| Pair distance | 15 min | 30 min | 60 min |
|---|---|---|---|
| 1 link | **0.115** | **0.158** | **0.200** |
| 2 links | 0.019 | 0.031 | 0.050 |
| 3 links | 0.010 | 0.014 | 0.025 |
| 4+ links | 0.007 | 0.010 | 0.017 |
| not connected | 0.003 | 0.005 | −0.002 |

Errors are correlated **only along the road graph** and the effect decays with every hop. A network-wide cause (e.g. time of day) would correlate unconnected pairs too; it does not. The correlation also grows with horizon, as expected if the missing information is congestion travelling between sensors.

## 3 · Neighbour-information test (training)
XGBoost is given hand-made neighbour features (weighted mean speed now and 5 / 15 / 30 min ago, slowest neighbour, largest 15-min drop) for upstream only, downstream only, and both directions. Same depth, same budget, same early stopping, no extra tuning; trained for 15 / 30 / 60 min. First validation numbers, upstream only: MAE 2.772 vs 2.816 at 15 min and 3.241 vs 3.291 at 30 min. Full results and the figure follow when training finishes.

## Gap statement (draft, completed with evidence 3)
> Independent segment models forecast each sensor from its own history. Congestion forms at bottlenecks and travels backwards along road links at ~16 km/h (Phase 1), so at longer horizons the most predictive information sits at neighbouring sensors. We observe this as error rising steadily with horizon, errors 5–6× larger in the 30 minutes after a congestion onset, and residuals that are correlated only between road-linked sensors. A graph model over the road network should close this gap.
