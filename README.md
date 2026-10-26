# Traffic Flow Prediction & Signal Control via Graph Attention Networks

**Does a model that sees the road network forecast traffic better than strong per-sensor models? And where does the gain come from?**
Speed forecasting 5–60 min ahead on METR-LA (207 Los Angeles freeway sensors), comparing per-segment XGBoost / Random Forest baselines with graph neural networks (STGAT, DCRNN) under a leak-free protocol, plus an optional **simulated** signal-control experiment in SUMO.

> Signal control in this project is **simulation only**. Nothing here controls real traffic.

![A jam reaches the downstream sensor first, then travels back upstream](reports/figures/fig05d_propagation_example.png)

*Phase 1 finding: congestion propagates **backwards** along LA freeways at ≈16 km/h. On strongly coupled links the downstream sensor slows first 4× more often than the upstream one. A per-sensor model can't see this coming; a graph model can.*

## Status

| Phase | | Status |
|---|---|---|
| 0 | Setup, repo, pre-reads | Done |
| 1 | Data + EDA ([report](docs/phase_reports/PHASE_1_REPORT.md)) | Done |
| 2 | Leak-free preprocessing ([report](docs/phase_reports/PHASE_2_REPORT.md)) | Done |
| 3 | ML baselines ([report](docs/phase_reports/PHASE_3_REPORT.md)) | Done |
| 4 | Identify the gap ([report](docs/phase_reports/PHASE_4_REPORT.md)) | Done |
| 5 | Graph models (STGAT, DCRNN) + ablations | Next |
| 6 | Fair comparison, horizon & peak metrics, significance | Not started |
| 7 | Congestion maps | Not started |
| 8 | Simulated signal control (SUMO, optional) | Not started |
| 9 | Extensions & packaging | Not started |

The full plan, with every requirement of the brief mapped to a phase, is in **[docs/PROJECT_GUIDE.md](docs/PROJECT_GUIDE.md)**.

## Results
**AML part (Phases 3–4), test block, masked MAE in mph** ([baselines.csv](results/baselines.csv), [gap_results.csv](results/gap_results.csv)):

| Model | 15 min | 30 min | 60 min |
|---|---|---|---|
| Persistence | 3.49 | 4.20 | 5.37 |
| Historical average | 4.42 | 4.42 | 4.42 |
| Ridge per sensor | 3.31 | 4.08 | 5.17 |
| Random Forest (multi-output) | 3.08 | 3.68 | 4.45 |
| LSTM, no graph | 2.97 | 3.55 | 4.35 |
| **XGBoost per horizon** (best independent model) | 3.02 | 3.51 | 4.01 |
| XGBoost + neighbours (both directions) | **2.91** | **3.42** | **3.95** |

The gap (Phase 4): XGBoost's error is 5–6× higher in the 30 min after a congestion onset, and its errors correlate only between road-linked sensors. Neighbour features help at every horizon, which is the case for a graph model.

*Graph models and the full comparison (peak vs off-peak, significance) follow in Phases 5–6.*

## Dashboard and slides
- `make dashboard` opens an interactive Streamlit dashboard over the saved results (`app.py`, contributed by Sumit).
- [Project slides](docs/presentation/AML_Traffic_Flow_Forecasting.pptx): 11-slide summary of the AML part (data, leak-free pipeline, baselines, all scores, the gap).

## Reproduce
```bash
make setup      # Python 3.11 venv (uv) + pinned requirements; macOS also: brew install libomp
make data       # DCRNN repo graph files + metr-la.h5 from its Google Drive link
make eda        # re-executes notebooks/01_eda.ipynb → reports/figures/, results/phase1_findings.json
make preprocess # Phase 2: data/processed/strict/{train,val,test}.npz + scaler.json + adj.npz
make test lint
```

## Data
METR-LA from [liyaguang/DCRNN](https://github.com/liyaguang/DCRNN#data-preparation): 34,272 five-minute speed readings × 207 sensors, Mar–Jun 2012, with a directed road-distance adjacency. 8.1% of readings are sensor outages (0 mph), which are masked and never treated as traffic.

## Limitations
Speed-only data (flow/occupancy via PeMS is a Phase 9 extension); 2012 freeway data from one city; signal control is evaluated only in a synthetic SUMO network.
