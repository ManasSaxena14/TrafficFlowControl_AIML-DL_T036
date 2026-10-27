# Traffic Flow Prediction and Signal Control via Graph Attention Networks and Flow Models

[Live dashboard](https://trafficflowamldl.streamlit.app) · [Project report (PDF, 22 pages)](docs/report/report.pdf) · [Slides (13)](docs/presentation/AML_Traffic_Flow_Forecasting.pptx)

**Does a model that sees the road network forecast traffic better than strong per-sensor models, and where does the gain come from?**
This project forecasts freeway speed 5 to 60 minutes ahead on METR-LA (207 Los Angeles sensors). It builds a leak-free preprocessing pipeline, six forecasting baselines, and four experiments that measure what independent per-sensor models miss. The graph deep-learning models that should close that gap (graph attention and diffusion networks) are the next stage and are not part of the results below. Signal control is simulation only; nothing here controls real traffic.

![A jam reaches the downstream sensor first, then travels back upstream](docs/figures/fig05d_propagation_example.png)

*Congestion propagates backwards along the freeway at about 16 km/h: the downstream sensor slows first. A model that sees only its own sensor's history cannot see this coming.*

## Results

Test block (the last 24 days, never used for tuning); masked MAE in mph, lower is better. Full tables: [results/baselines.csv](results/baselines.csv), [results/gap_results.csv](results/gap_results.csv).

| Model | 15 min | 30 min | 60 min |
|---|---|---|---|
| Persistence | 3.49 | 4.20 | 5.37 |
| Historical average | 4.42 | 4.42 | 4.42 |
| Ridge per sensor | 3.31 | 4.08 | 5.17 |
| Random Forest (multi-output) | 3.08 | 3.68 | 4.45 |
| LSTM, no graph | 2.97 | 3.55 | 4.35 |
| **XGBoost per horizon** (best independent model) | 3.02 | 3.51 | 4.01 |
| XGBoost + neighbour speeds (both directions) | **2.91** | **3.42** | **3.95** |

**What the experiments show**
- XGBoost is the strongest independent model; its error grows steadily with the forecast horizon.
- In the 30 minutes after a congestion onset its error is 5.9 to 6.4 times larger than at other times (3,911 onsets in the test period).
- The errors of road-linked sensors are correlated (0.20 at 60 minutes for sensors one link apart) while those of unconnected sensors are not.
- Adding hand-made neighbour speeds lowers error at every horizon (up to 3.5%), so the missing information is spatial, but one-hop features capture only part of it. That is the case for a graph model.
- Weekday rush-hour error is 1.4 to 1.95 times off-peak error for every model.

### Model accuracy

Speed forecasting is a regression task, so there is no single accuracy figure. Two plain-language scores on the test block ([results/accuracy.csv](results/accuracy.csv)): **percent accuracy** (100 minus MAPE) and the share of forecasts **within 5 mph** of the true speed.

| Model | Percent accuracy (15 / 30 / 60 min) | Within 5 mph, % (15 / 30 / 60 min) |
|---|---|---|
| Persistence | 91.4 / 89.2 / 85.5 | 80.8 / 77.9 / 73.5 |
| Historical average | 86.1 / 86.1 / 86.1 | 76.5 / 76.5 / 76.5 |
| Ridge per sensor | 91.1 / 88.3 / 84.5 | 82.3 / 77.9 / 70.6 |
| Random Forest | 91.7 / 89.4 / 86.5 | 84.1 / 80.6 / 75.9 |
| LSTM (no graph) | 92.0 / 89.7 / 86.5 | 85.1 / 82.6 / 79.5 |
| **XGBoost (best independent model)** | 91.8 / 89.9 / 88.0 | 84.3 / 81.4 / 78.6 |
| **XGBoost + neighbours (both directions)** | 92.3 / 90.4 / 88.3 | 84.7 / 81.7 / 78.8 |

These percentages look high mainly because traffic flows freely most of the time, so the error in mph and the error during congestion are the more informative quantities. See the report for RMSE, MAPE, peak versus off-peak and validation scores.

## Research foundation

The project follows the hybrid logic of the brief (segment-level machine learning first, network-aware deep learning second) and builds on published work; the report maps each reference to its role in a literature table and the slides summarise it.

| Work | Idea | Used here for |
|---|---|---|
| Li et al., ICLR 2018 (DCRNN) | traffic as diffusion on a directed graph with bidirectional random walks; 12 to 15% gain over its baselines | data, graph and protocol reference; model to reproduce next; motivates the upstream/downstream experiment |
| Veličković et al., ICLR 2018 | graph attention over neighbours | the attention model named in the brief (next stage) |
| Yu et al., IJCAI 2018; Wu et al., IJCAI 2019 | graph forecasting with convolutional structures; learned adjacency | graph baselines and ablations (next stage) |
| Chen and Guestrin 2016; Breiman 2001; Hochreiter and Schmidhuber 1997; Hoerl and Kennard 1970 | XGBoost, Random Forest, LSTM, Ridge | the four learned baselines |
| Kaufman et al. 2012; Hyndman and Koehler 2006 | data leakage; properties of accuracy measures | purge gap and leakage tests; how MAE, RMSE and MAPE are read |
| Zhang et al., WWW 2019 (CityFlow) | reinforcement-learning signal simulator | optional, simulation-only signal control |

Four hypotheses derived from this literature were tested and all were supported: (H1) error grows with the horizon, (H2) errors concentrate after congestion onsets, (H3) residuals correlate along road links, (H4) neighbour information helps, downstream more than upstream, but one-hop features do not close the gap (report, Section 8.1).

## Method in brief

- **Leak-free protocol.** Timestamps are split 70/10/20 into train, validation and test; windows whose targets straddle a boundary are purged, so no target is in two splits. The scaler is fitted on training data only, and a test proves that the reference window-level split lets train and validation targets overlap.
- **Outages are masked, not treated as zero traffic.** Missing readings are NaN targets with an explicit mask; every loss and metric ignores them.
- **Equal history for every model.** The last 12 steps (60 minutes) in, the next 12 steps (5 to 60 minutes) out.
- **Six baselines, one evaluation path.** Persistence, historical average, Ridge per sensor, XGBoost per horizon, a multi-output Random Forest, and an LSTM with shared weights and no graph. Hyper-parameters are chosen on validation only.
- **Gap experiments.** Error against horizon, error after congestion onsets, residual correlation along road links, and a neighbour-information test.

## Repository structure

```
.
├── app.py                  Streamlit dashboard over the saved results (deployed)
├── Makefile                one command per stage (see below)
├── configs/data/           split, window, masking and peak definitions
├── src/
│   ├── data/               loading, leak-free preprocessing, readers
│   ├── features/           46-column per-sensor feature table
│   ├── models/             simple baselines, XGBoost and Random Forest, LSTM
│   ├── eval/               masked metrics, per-horizon and peak evaluation, accuracy
│   ├── analysis/           exploratory analysis and gap analysis
│   ├── train/              runners for the baselines and the gap experiments
│   └── viz/                figures
├── tests/                  automated tests (data, leakage, baselines, gap)
├── results/                every result table (CSV and JSON)
├── notebooks/              exploratory data analysis
├── scripts/                data download; report table generator
└── docs/
    ├── report/             LaTeX report, bibliography, compiled PDF
    ├── presentation/       slide deck
    └── figures/            figures used by the report and README
```

## Getting started

```bash
make setup        # Python 3.11 environment (uv) and pinned requirements; macOS: brew install libomp
make data         # METR-LA and the sensor graph (graph files from the DCRNN repository)
make eda          # exploratory analysis: figures and results/phase1_findings.json
make preprocess   # leak-free windows -> data/processed/strict/
make features     # per-sensor feature tables
make baselines    # six baselines -> results/baselines*.csv
make gap          # gap experiments -> results/gap*/
make accuracy     # accuracy summaries -> results/accuracy.csv
make report       # regenerate tables from results/ and compile docs/report/report.pdf (brew install tectonic)
make test lint
make dashboard    # open the dashboard locally
```

The deployed dashboard at [trafficflowamldl.streamlit.app](https://trafficflowamldl.streamlit.app) reads only the pre-computed files in `results/`. Intermediate data (about 2.5 GB per protocol) is not committed.

## Documentation

- **[Project report](docs/report/report.pdf)**: 22 pages with abstract, literature review, data, methodology, all results, the gap analysis, software and deployment, limitations and a bibliography. The LaTeX source is in `docs/report/`; every table and quoted number is generated from `results/` by `scripts/make_report_tables.py`, and `make report` rebuilds it.
- **[Slides](docs/presentation/AML_Traffic_Flow_Forecasting.pptx)**: 13 slides covering the problem, research foundation, data, leak-free pipeline, baselines, all scores, the gap, conclusion and references.
- **[Live dashboard](https://trafficflowamldl.streamlit.app)**: interactive view of the saved results (`app.py`, contributed by Sumit).

## Data

METR-LA ([Jagadish et al., 2014](https://doi.org/10.1145/2611567)) as distributed with the [DCRNN](https://github.com/liyaguang/DCRNN#data-preparation) code: 34,272 five-minute speed readings from 207 sensors, 1 March to 27 June 2012, with a directed road-distance adjacency matrix. 8.1% of readings are sensor outages (0 mph), which are masked.

## Limitations

METR-LA records speed only, so flow and occupancy features need PeMS data. The data are 2012 freeway readings from one city. Each model was trained once (seed 42) and no significance tests have been run, so small differences should not be over-interpreted. Graph deep-learning models have not been built yet and the reference DCRNN numbers have not been reproduced under this protocol.

## References

The full bibliography, with DOIs and arXiv identifiers, is in [docs/report/references.bib](docs/report/references.bib).

1. Y. Li, R. Yu, C. Shahabi, Y. Liu. *Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting.* ICLR 2018. [arXiv:1707.01926](https://arxiv.org/abs/1707.01926)
2. P. Veličković, G. Cucurull, A. Casanova, A. Romero, P. Liò, Y. Bengio. *Graph Attention Networks.* ICLR 2018. [arXiv:1710.10903](https://arxiv.org/abs/1710.10903)
3. B. Yu, H. Yin, Z. Zhu. *Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting.* IJCAI 2018. [arXiv:1709.04875](https://arxiv.org/abs/1709.04875)
4. Z. Wu, S. Pan, G. Long, J. Jiang, C. Zhang. *Graph WaveNet for Deep Spatial-Temporal Graph Modeling.* IJCAI 2019. [arXiv:1906.00121](https://arxiv.org/abs/1906.00121)
5. H. Zhang et al. *CityFlow: A Multi-Agent Reinforcement Learning Environment for Large Scale City Traffic Scenario.* WWW 2019. [arXiv:1905.05217](https://arxiv.org/abs/1905.05217)
6. T. Chen, C. Guestrin. *XGBoost: A Scalable Tree Boosting System.* KDD 2016. [doi:10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785)
7. L. Breiman. *Random Forests.* Machine Learning 45(1):5-32, 2001.
8. S. Hochreiter, J. Schmidhuber. *Long Short-Term Memory.* Neural Computation 9(8):1735-1780, 1997.
9. A. E. Hoerl, R. W. Kennard. *Ridge Regression: Biased Estimation for Nonorthogonal Problems.* Technometrics 12(1):55-67, 1970.
10. D. P. Kingma, J. Ba. *Adam: A Method for Stochastic Optimization.* ICLR 2015. [arXiv:1412.6980](https://arxiv.org/abs/1412.6980)
11. H. V. Jagadish et al. *Big Data and Its Technical Challenges.* Communications of the ACM 57(7):86-94, 2014. [doi:10.1145/2611567](https://doi.org/10.1145/2611567)
12. S. Kaufman, S. Rosset, C. Perlich, O. Stitelman. *Leakage in Data Mining: Formulation, Detection, and Avoidance.* ACM TKDD 6(4), 2012.
13. R. J. Hyndman, A. B. Koehler. *Another Look at Measures of Forecast Accuracy.* International Journal of Forecasting 22(4):679-688, 2006.

## Contributors

Manas Saxena (pipeline, baselines, gap analysis, report, slides), Harsh Raj (project setup, data loading and exploratory analysis), Sumit (Streamlit dashboard).
