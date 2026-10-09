# Traffic Flow Prediction & Signal Control via Graph Attention Networks and Flow Models
### The complete phase-by-phase guide

> **One-sentence pitch.** Prove, with leak-free numbers, whether a graph model that *sees the road network* beats strong per-sensor models at forecasting traffic speed 5–60 min ahead on METR-LA, explain *where* the gain comes from, and use those forecasts in a **simulated** signal-control experiment.

This guide is the single source of truth for the project. It covers every line of the brief, splits the work into 10 phases with a gate between them, and says for each phase **what** to build, **how** to build it, **what done looks like**, and **what an interviewer will ask**. It extends the original project playbook with things learned from the real data in Phase 1.

---

## Status board

| Phase | Name | Days | Status | Gate to pass before the next phase |
|---|---|---|---|---|
| 0 | Setup, structure, pre-reads | 1–3 | ✅ Done | Repo builds, `make test` passes, paper notes written |
| 1 | Get the data and understand it | 4–6 | ✅ Done | Column order verified; five EDA figures; findings written ([report](phase_reports/PHASE_1_REPORT.md)) |
| 2 | Preprocessing without leakage | 7–9 | ⏭ Next | **Leakage tests pass** ◆ |
| 3 | ML baselines (strong, honest, per-segment) | 10–14 | ☐ | `results/baselines.csv` complete |
| 4 | Identify the gap with evidence | 15–16 | ☐ | Four gap plots + "XGB + neighbours" row |
| 5 | Graph deep-learning models | 17–28 | ☐ | **DCRNN within ~5% of reference** ◆ |
| 6 | Fair comparison | 29–32 | ☐ | **Results tested for significance** ◆ |
| 7 | Congestion maps and visuals | 33–36 | ☐ | Figures 1–6 + interactive map |
| 8 | *Optional* simulated signal control (SUMO) | 37–42 | ☐ | 3 controllers × ≥10 seeds |
| 9 | Extensions, packaging, placement prep | 43–50 | ☐ | README, CI, report, demo, resume bullets |

◆ = hard gate. A leak found after training wastes every model trained before it.

---

## 1. The brief, line by line → where it is done

Every requirement in the project brief is listed here. Tick these off as you go; this table is what you show a reviewer to prove nothing was skipped.

| # | Brief requirement | Where | How it is satisfied |
|---|---|---|---|
| 1 | Predict congestion in a road network | Ph 1–7 | Speed forecasts for 207 sensors, 5–60 min ahead; converted to congestion levels in Ph 7 |
| 2 | Test whether graph learning improves on independent segment forecasts | Ph 3–6 | Same data, same windows, same mask, same metric function for every model |
| 3 | Signal control simulated only; never claim live control | Ph 8, 9 | SUMO synthetic grid; exact wording rule in Ph 8; checked again in README review |
| 4 | **AML baseline:** lagged speed, flow, occupancy, time-of-day, road-segment features | Ph 3 | Lags/rolling/change/missing/time/segment features. METR-LA is speed-only, so flow/occupancy come from PeMS in Ph 9 and the gap is stated in the report |
| 5 | XGBoost per segment **or** multi-output Random Forest | Ph 3 | Both are built |
| 6 | **Identify the gap:** independent models can't represent upstream/downstream propagation | Ph 1, 4 | Ph 1 measured propagation (fig 5); Ph 4 shows the per-sensor model fails exactly there |
| 7 | **DL model:** GAT or diffusion graph model over the road graph | Ph 5 | Both: STGAT (ours, PyG) and DCRNN (diffusion) |
| 8 | METR-LA sensor series + fixed adjacency matrix | Ph 2, 5 | `adj_mx.pkl`, rebuilt and verified from `distances_la_2012.csv`; kept fixed (learned graph only as an ablation) |
| 9 | SUMO only for optional simulation | Ph 8 | SUMO is never used for the forecasting comparison |
| 10 | **Fair comparison:** chronological splits | Ph 2 | 70/10/20 by time + purge gap; tests enforce it |
| 11 | MAE, RMSE, MAPE by forecast horizon | Ph 6 | One `evaluate()` for all models, horizons 1…12 (headline 15/30/60 min) |
| 12 | Equal history windows | Ph 2, 3, 5 | 12 steps = 60 min for every model, including tree-model lags |
| 13 | Peak vs off-peak performance | Ph 6 | Weekday AM 07:00–09:59, PM 16:00–19:59 on the *predicted* timestamp; weekends separate |
| 14 | Pre-reads: Veličković 2018 (GAT), Zheng 2019 (CityFlow) | Ph 0 | `docs/papers/` (+ DCRNN and Graph WaveNet notes) |
| 15 | Starter dataset METR-LA; PeMS later | Ph 1, 9 | METR-LA now; PEMS-BAY already downloaded for the Ph 9 extension |
| 16 | Stack: Python, pandas, scikit-learn, PyTorch, PyG, NetworkX, optional SUMO | Ph 0 | `requirements.txt` (pinned) |
| 17 | Construct adjacency from the supplied graph | Ph 2 | Rebuild from distances, assert equal to `adj_mx.pkl` |
| 18 | Fit scaling only on training timestamps | Ph 2 | Scaler fitted on masked train slice; test asserts it |
| 19 | Cyclic time features | Ph 2 | sin/cos time-of-day, sin/cos day-of-week, weekend flag |
| 20 | Never randomly split sequences | Ph 2 | Shuffling only *within* train for SGD |
| 21 | Mask sensor outages instead of zero traffic | Ph 1, 2, 3, 5, 6 | 0 → NaN; mask channel in inputs; masked loss + masked metrics |
| 22 | **Deliverable:** forecast comparison | Ph 6 | `results/final.csv` |
| 23 | **Deliverable:** congestion maps | Ph 7 | Side-by-side, animated, error map, time-space diagram |
| 24 | **Deliverable:** horizon-wise metrics | Ph 6 | Horizon line plot + table |
| 25 | **Deliverable:** optional simulated signal policy experiment | Ph 8 | Fixed-time vs max-pressure vs forecast-aware |

---

## 2. What makes this "FAANG-grade"

The model is not what impresses. Interviewers at strong ML teams probe five things, and every phase below is designed around them:

1. **Leak-free evaluation.** Chronological splits, purge gaps, train-only scaling, tests that enforce all three.
2. **Honest, strong baselines.** A tuned XGBoost and a no-graph LSTM. If a GNN only beats a weak baseline, nobody believes it.
3. **Ablations that explain *why*.** No graph / random graph / undirected / hop count / attention inspection.
4. **Reproducibility.** Pinned environment, seeds, configs saved with checkpoints, `make` targets, CI.
5. **A clear write-up.** A README that leads with the result, states limitations, and never over-claims.

**Engineering rules for the whole project**
- Logic lives in `src/`; notebooks only call it and tell the story.
- Fix seeds (`random`, `numpy`, `torch`); report neural models as mean ± std over 3 seeds.
- Every training run logs config + metrics (Weights & Biases or MLflow) and saves the config next to its checkpoint.
- Every bug that could silently corrupt results gets a test (`tests/`).
- `make test lint` passes before every commit.

---

## 3. Repository map

```
traffic-gnn/
├── configs/data/metr_la.yaml   # split, window, masking, peaks: single source of truth
├── data/raw/                   # metr-la.h5, pems-bay.h5, adj_mx.pkl, distances, locations (git-ignored)
├── data/processed/             # train/val/test .npz, scaler.json (Phase 2)
├── docs/
│   ├── PROJECT_GUIDE.md        # ← this file
│   ├── project_brief.jpeg      # the brief
│   ├── papers/                 # one-page notes per pre-read
│   └── phase_reports/          # one report per completed phase
├── notebooks/01_eda.ipynb      # Phase 1 (executed, with outputs)
├── reports/figures/            # every figure that goes into the report/README
├── results/                    # metrics tables, phase findings (json/csv)
├── scripts/download_data.sh    # one-command data download
├── src/
│   ├── data/load.py            # robust loaders + sensor-order check
│   ├── analysis/eda.py         # Phase 1 analysis functions
│   ├── features/               # Phase 3 tree-model features
│   ├── models/                 # baselines, xgb, rf, lstm, stgat, dcrnn
│   ├── train/                  # trainer, masked losses, early stopping
│   ├── eval/                   # evaluate(), peak split, significance tests
│   └── viz/                    # style.py (shared look), eda_plots.py, maps
├── sim/                        # SUMO network, routes, controllers (Phase 8)
├── tests/                      # data, leakage, masking, shape tests
├── Makefile                    # make setup | data | eda | test | lint
└── requirements.txt            # pinned
```

---

## Phase 0 — Setup, structure and pre-reads ✅

**Goal.** A reproducible repo and a clear mental model of the key papers before writing model code.

**What was done**
- Python 3.11 virtual env via `uv` (no conda needed); all packages pinned in `requirements.txt`. PyTorch sees the Apple-silicon GPU (`mps`). On macOS, XGBoost needs `brew install libomp`.
- Repo skeleton above; `pyproject.toml` configures `pytest` + `ruff`; `Makefile` gives one-command targets.
- Pre-read notes in `docs/papers/`: GAT, CityFlow, DCRNN, Graph WaveNet.

**How to reproduce from scratch**
```bash
make setup     # venv + pinned packages
make data      # clones DCRNN repo, downloads metr-la.h5 / pems-bay.h5 from the Drive link in its README
make test lint
```

**Still to do yourself:** create a GitHub repo and push (`git init`, first commit, `git remote add origin …`, `git push`). Keep `data/raw/` out of git (already in `.gitignore`).

**Interview angle.** *"How do you make an ML experiment reproducible?"* → pinned env, seeds, config-per-run, data download script, tests, CI.

---

## Phase 1 — Get the data and understand it ✅

**Goal.** Know the data well enough to justify every later modelling choice. Full results: [`phase_reports/PHASE_1_REPORT.md`](phase_reports/PHASE_1_REPORT.md).

**What the data is**

| File | Contents |
|---|---|
| `metr-la.h5` | 34,272 timestamps × 207 sensors, 5-min speed (mph), 1 Mar – 27 Jun 2012 (119 days), no clock gaps |
| `adj_mx.pkl` | `(sensor_ids, id→index, A)`; `A` is 207×207, **directed**; `A[i,j] = exp(−(d_ij/σ)²)` for driving distance i→j, zeroed below 0.1; 1,515 off-diagonal links |
| `graph_sensor_locations.csv` | lat/lon per sensor (maps) |
| `distances_la_2012.csv` | 295k road distances (from, to, metres); rebuild the adjacency in Ph 2 |

**How it was done.** `src/data/load.py` (the `.h5` was written by pandas 0.15 / Python 2 and modern pandas can't parse it, so it is read via `h5py`; `load_all()` raises if sensor order ≠ graph order) → `src/analysis/eda.py` (pure functions) → `src/viz/eda_plots.py` (figures) → `notebooks/01_eda.ipynb` (story). 9 tests in `tests/test_data.py`.

**Headline findings (these change later phases)**
1. **Outages:** 8.11% of readings are 0. **6.27 points are whole-network outages** (2,148 timestamps where all 207 sensors read 0, the longest 31.8 h); only 1.84 points are single-sensor outages. → Phase 2 must mask, never treat 0 as traffic; windows whose targets are fully masked carry no signal.
2. **Daily rhythm:** weekday AM dip 07:05–10:45 (min 49 mph), PM dip 14:30–19:15 (min 45.7 mph); weekends barely dip. The brief's peak definition (07–10, 16–20) sits inside these windows. → Time-of-day and weekend features are essential.
3. **Skew:** median 63 mph; 66% of readings ≥ 60 mph; only 10.5% < 40 mph. → Average MAE is dominated by easy free flow. Always break results down by peak and by congestion onset.
4. **Graph:** mean degree 7.3; **only 27% of links are two-way**; 1 isolated sensor. → Keep the graph directed.
5. **Propagation (the key result):** on strongly coupled links, 61% co-move within 5 min. Where there is a lag, **the downstream sensor slows first on 156 links vs upstream-first on 41** (median lag 10 min). The lag grows with distance (6 → 16 min from <1.5 km to 3–4 km), an implied backward wave of **≈16 km/h**, the textbook shockwave speed. → Information about a coming jam sits at the *downstream* neighbour.
6. **Noisy fixed graph:** **62% of distance-based links barely co-move** (peak correlation < 0.3). → Motivates attention (GAT) and a learned-adjacency ablation.

---

## Phase 2 — Preprocessing without leakage ⏭

**Goal.** One shared pipeline that feeds every model, so the comparison is fair by construction.
**Output.** `data/processed/{train,val,test}.npz` with `X (samples, 12, 207, F)`, `Y (samples, 12, 207)`, `M_y` (target mask), plus `scaler.json` and `adj.npz`.

### Step 1 — Chronological split (never random)
- Split **timestamps** 70 / 10 / 20 (train ≈ 1 Mar – 23 May, val ≈ next 12 days, test ≈ last 24 days), *then* cut windows inside each block.
- **Purge gap:** drop the 11 boundary windows whose targets would cross into the next split. The reference DCRNN script splits *windows* and lets the last train targets overlap val targets. Build **two modes** in config: `protocol: dcrnn` (to compare with published numbers) and `protocol: strict` (purged). Report both once; use strict for the rest.
- *Why:* overlapping windows share 11 of 12 timestamps. A random split puts near-copies of test windows into train.

### Step 2 — Masking outages
```python
speed = df.values.astype("float32")
mask = speed > 0                 # True = real reading
speed[~mask] = np.nan
```
- **Inputs:** fill NaN with the last valid value within the window (forward fill, max 12 steps), else that sensor's train mean for that time slot; **add the mask as an input channel** so the model knows a value was imputed.
- **Targets:** keep NaN. Loss and every metric ignore them (masked MAE below). Never train a model to predict 0 mph for a dead sensor.
- **Whole-network outages (Phase 1 finding):** drop *training* windows whose targets are > 90% masked (pure noise for SGD). Keep every test window: metrics ignore masked points anyway.

$$\mathrm{MAE}_{mask} = \frac{\sum_{t,i} m_{t,i}\,|\hat y_{t,i}-y_{t,i}|}{\sum_{t,i} m_{t,i}}$$

### Step 3 — Scaling fitted on train timestamps only
```python
mu = np.nanmean(speed[:n_train]); sd = np.nanstd(speed[:n_train])   # NaN-aware: zeros don't bias it
speed_z = (speed - mu) / sd                                          # same mu, sd for val/test
json.dump({"mu": float(mu), "sd": float(sd)}, open("data/processed/scaler.json", "w"))
```
(The DCRNN reference computes the mean *including* zeros. This is another difference to mention.) Inverse-transform predictions before any metric, so every error is in mph.

### Step 4 — Cyclic time features
```python
tod = (idx.hour * 60 + idx.minute) / 1440
sin_tod, cos_tod = np.sin(2*np.pi*tod), np.cos(2*np.pi*tod)
dow = idx.dayofweek / 7
sin_dow, cos_dow = np.sin(2*np.pi*dow), np.cos(2*np.pi*dow)
is_weekend = (idx.dayofweek >= 5).astype("float32")
```
23:55 → (−0.022, 0.9998) and 00:05 → (0.022, 0.9998): nearly identical, as they should be.
Final node features **F = 7**: speed_z, mask, sin/cos tod, sin/cos dow, weekend.

### Step 5 — Adjacency from the supplied graph
Rebuild from `distances_la_2012.csv` and assert it equals `adj_mx.pkl` (a common interview question):
$$W_{ij} = \exp\!\left(-\frac{d_{ij}^2}{\sigma^2}\right) \text{ if } \ge \kappa \text{ else } 0,\quad \sigma = \mathrm{std}(d \text{ among the 207 sensors}),\ \kappa = 0.1$$
```python
from torch_geometric.utils import dense_to_sparse
edge_index, edge_weight = dense_to_sparse(torch.tensor(adj))   # directed: A[i,j] != A[j,i]
```
Keep it directed and fixed (brief). A learned adjacency is only an ablation.

### Step 6 — Windowing
Input = last 12 steps (60 min); output = next 12 steps (5…60 min). The **same 12-step history for every model**, which is the equal-history requirement.

### Leakage tests (`tests/test_leakage.py`): the Phase 2 gate
- [ ] max train target timestamp < min val target timestamp < min test target timestamp
- [ ] no target timestamp appears in two splits (strict mode)
- [ ] scaler mean equals the NaN-mean of the train slice only
- [ ] masked positions contribute exactly zero to the loss (perturb them, loss unchanged)
- [ ] sensor order in X equals adjacency order
- [ ] rebuilt adjacency == `adj_mx.pkl`
- [ ] shapes: X (S,12,207,7), Y (S,12,207)

**Interview angle.** *Why not random split? Why mask zeros? Why fit the scaler on train only? What is a purge gap?*

---

## Phase 3 — ML baselines: strong, honest, per-segment

**Goal.** A ladder of baselines from trivial to strong. Each rung is one row in the final table.

| Rung | Model | Why include it |
|---|---|---|
| 0 | Persistence (last value) | The floor; surprisingly hard to beat at 5 min |
| 1 | Historical average (weekday/weekend × slot, train only) | Captures the daily rhythm |
| 2 | Ridge per sensor | Cheap linear reference |
| 3 | **XGBoost per segment** (brief) | Strongest independent model |
| 4 | **Multi-output Random Forest** (brief) | One model, 12 horizons at once |
| 5 | LSTM/GRU, **no graph**, weights shared across sensors | Isolates "deep learning" from "graph". If a GNN beats this, the gain is from the graph |

### Feature table (one row = one sensor at one time t), built in `src/features/`

| Group | Columns | Notes |
|---|---|---|
| Lagged speed | speed[t], speed[t−1] … speed[t−11] | Exactly 12 lags = same 60-min history as the GNN |
| Rolling stats | mean/std/min over last 3, 6, 12 steps | Past only |
| Change | speed[t]−speed[t−1], speed[t]−speed[t−3] | Detects onset of slowdown |
| Missing flags | mask[t−k], count of imputed lags | From Phase 2 |
| Time | sin/cos tod, sin/cos dow, is_weekend | From Phase 2 |
| Segment | sensor ID (categorical / target-encoded on train), train mean speed, free-flow speed (85th pct), in/out degree | Static per sensor |
| Flow, occupancy | lagged flow & occupancy | **Not in METR-LA** (speed only) → Phase 9 with PeMS; state this limitation in the report |

### XGBoost per segment
Pick and state one design:
- **Global per horizon (recommended):** one XGBoost per horizon (12 models) trained on all sensors stacked, with sensor ID as a feature. Still *independent*: it never sees neighbours' current speeds.
- **Direct per sensor:** 207 × 12 ≈ 2,500 small models, parallelised with `joblib`.
```python
for h in range(1, 13):
    m = xgb.XGBRegressor(n_estimators=600, max_depth=8, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.8, tree_method="hist", early_stopping_rounds=50)
    m.fit(Xtr, ytr[:, h-1], sample_weight=mask_tr[:, h-1], eval_set=[(Xva, yva[:, h-1])])
```
Masked targets get weight 0. Tune on the **val block only** (fixed block or `TimeSeriesSplit`), with the same trial budget you will give the GNN.

### Multi-output Random Forest
```python
rf = RandomForestRegressor(n_estimators=300, max_depth=20, min_samples_leaf=5, n_jobs=-1)
rf.fit(Xtr, Ytr)   # Ytr (rows, 12)
```
RF can't take NaN targets: drop training rows with any masked target, but **evaluate on all test rows with the mask**. Memory tip: ~5M rows × ~45 features; subsample training rows (e.g. every 3rd timestamp) if needed and say so.

**Deliverables**
- [ ] `results/baselines.csv`: MAE/RMSE/MAPE at 15/30/60 min for rungs 0–5
- [ ] XGBoost feature-importance plot (expect speed[t] and time-of-day to dominate)
- [ ] One paragraph: which baseline is strongest and at which horizon it degrades

---

## Phase 4 — Identify the gap with evidence

**Goal.** Don't just *assert* that independent models miss propagation: measure it. Phase 1 already showed propagation exists; Phase 4 shows the per-sensor model **fails exactly where it happens**.

1. **Error vs horizon.** XGBoost MAE for h = 1…12. Expect a steep rise after ~15 min: the jam that will arrive is currently *at a neighbour*.
2. **Error at congestion onset.** Reuse `eda.congestion_onsets()` (speed drops > 20 mph within 15 min). Compare XGBoost error in the 30 min after onsets vs normal periods. Expect a large gap: the model reacts late.
3. **Neighbour-information test (the key experiment).** Add neighbours' lagged speeds as XGBoost features ("XGB + neighbours"). Phase 1 says **use both directions, and especially downstream neighbours** (downstream-first was 4× more common). Run three variants: upstream only, downstream only, both. If error drops, spatial information matters. This is a hand-made graph feature; the GNN learns it automatically and over multiple hops.
4. **Residual correlation.** Correlate XGBoost residuals of each sensor with its neighbours' residuals. High correlation along edges = errors with a shared spatial cause the model ignores.

**Worked example (real, from Phase 1, fig 5d).** Wed 2 May 2012: sensor 717462 and sensor 717472, 1.5 km upstream, both flow at ~68 mph. At 06:40 the downstream sensor crashes to ~20 mph; the upstream one is still at 66 mph. A per-sensor model at 717472, seeing only its own steady ~66 mph history, has no reason to forecast anything but ~65 mph for 06:50. The truth is 17 mph, an error of ~48 mph. A model that watches 717462 would have seen it coming.

**Gap statement for the report**
> Independent segment models forecast each sensor from its own history. Congestion forms at bottlenecks and travels backwards along road links at ~16 km/h (Phase 1), so at longer horizons the most predictive information sits at neighbouring sensors. We observe this as rising error with horizon, large error at congestion onset, and spatially correlated residuals. A graph model over the road network should close this gap.

**Deliverables**
- [ ] Four gap plots
- [ ] "XGB + neighbours" rows added to results (bridges baselines and GNNs)

---

## Phase 5 — Graph deep-learning models

**Goal.** Two graph models over the **fixed** adjacency: a GAT-based spatio-temporal model (ours, PyG) and a diffusion model (DCRNN). Doing both answers *which kind* of graph reasoning helps.

### Model A: STGAT (GAT + GRU)
At each of the 12 input steps a GAT layer mixes each sensor with its neighbours (spatial); a GRU reads the 12 mixed steps (temporal); a linear head outputs 12 horizons.

| Stage | Shape (B = 64) |
|---|---|
| Input X | (64, 12, 207, 7) |
| Input projection | (64, 12, 207, 64) |
| GAT per step (4 heads, concat → 64) | (64, 12, 207, 64) |
| GRU over time (per node) | (64, 207, 64) |
| Output head | (64, 12, 207) |

```python
class STGAT(nn.Module):
    def __init__(self, in_dim=7, hid=64, heads=4, horizon=12, edge_index=None, edge_attr=None):
        super().__init__()
        self.register_buffer("ei", edge_index); self.register_buffer("ea", edge_attr)
        self.inp = nn.Linear(in_dim, hid)
        self.gat1 = GATv2Conv(hid, hid // heads, heads=heads, edge_dim=edge_attr.size(1))
        self.gat2 = GATv2Conv(hid, hid // heads, heads=heads, edge_dim=edge_attr.size(1))
        self.gru = nn.GRU(hid, hid, num_layers=2, batch_first=True, dropout=0.1)
        self.head = nn.Linear(hid, horizon)
    def spatial(self, h, ei, ea):                      # residuals stop over-smoothing
        h = torch.relu(self.gat1(h, ei, ea)) + h
        return torch.relu(self.gat2(h, ei, ea)) + h
    def forward(self, x):                              # x: (B, T, N, F)
        B, T, N, _ = x.shape
        ei, ea = batch_edges(self.ei, self.ea, B, N)   # offset graph copies per sample
        h = self.inp(x)
        h = torch.stack([self.spatial(h[:, t].reshape(B*N, -1), ei, ea).reshape(B, N, -1) for t in range(T)], 1)
        _, last = self.gru(h.permute(0, 2, 1, 3).reshape(B*N, T, -1))
        return self.head(last[-1]).reshape(B, N, -1).permute(0, 2, 1)   # (B, 12, N)
```
**Two design decisions that come straight from Phase 1:**
1. **Message direction.** In PyG, an edge `i → j` sends a message from i to j. With the raw adjacency, each sensor would only hear from its **upstream** neighbours. But Phase 1 showed jams travel **backwards**, so the downstream neighbour carries the early warning. Fix: add the reversed edges too, with an edge feature `[weight, is_reverse]` (`edge_dim=2`). Attention then learns how much to trust each direction. This is the GAT equivalent of DCRNN's dual random walk, and an excellent interview point.
2. **Attention, not fixed weights.** 62% of distance-based links barely co-move. GAT can learn to ignore them, while a GCN would average them in.

Use **GATv2** (Brody et al. 2022): the original GAT's attention is "static" (the neighbour ranking is the same for every query node); GATv2 makes it dynamic. Be ready to explain this.

### Model B: DCRNN (diffusion)
Diffusion convolution = random walks over the **directed** graph, forward (downstream) and backward (upstream), up to K hops, inside a GRU encoder–decoder (seq2seq):
$$X_{:,p} \star_{\mathcal G} f_\theta = \sum_{k=0}^{K-1}\left(\theta_{k,1}(D_O^{-1}W)^k + \theta_{k,2}(D_I^{-1}W^\top)^k\right)X_{:,p}$$
Reference config (from the repo's `dcrnn_la.yaml`): 2 RNN layers, 64 units, K = 2, `dual_random_walk`, seq_len = horizon = 12, lr 0.01, LR steps at epochs 20/30/40/50 (×0.1), max grad norm 5, curriculum learning (scheduled sampling, decay 2000 steps).
Don't copy the TensorFlow repo. Port it to PyTorch (best for learning) or use `torch_geometric_temporal`'s `DCRNN` cell inside your own seq2seq wrapper.

**Sanity target (DCRNN README, METR-LA, after its 2018 bug fix):**

| Horizon | 5 min | 15 min | 30 min | 60 min |
|---|---|---|---|---|
| MAE | 2.18 | 2.67 | 3.08 | 3.56 |
| RMSE | 3.77 | 5.17 | 6.30 | 7.52 |
| MAPE | 5.17% | 6.84% | 8.38% | 10.30% |

Within ~5% of these under `protocol: dcrnn` means your pipeline is correct. Your strict-protocol numbers will be slightly different. Explain why.

### Training recipe (both models)

| Setting | Value |
|---|---|
| Loss | Masked MAE on inverse-scaled speed |
| Optimiser | Adam, lr 1e-3 (STGAT) / 1e-2 (DCRNN); MultiStepLR ×0.1 at 20, 30, 40 |
| Batch | 64 |
| Grad clipping | max norm 5 |
| Early stopping | patience 15–20 on val masked MAE |
| Epochs | up to 100 |
| Seeds | 3 per model; report mean ± std |
| Hardware | Apple M3 Pro `mps` works; Colab/Kaggle T4 if an epoch is too slow |

### Ablations (what impresses reviewers)
- **No graph:** identity adjacency → should collapse to LSTM level.
- **Directed vs undirected**, and **forward-only vs forward+reverse edges** (tests the Phase 1 finding directly).
- **Hops:** K = 1, 2, 3 (DCRNN) or 1 vs 2 GAT layers.
- **Random graph** with the same edge count → proves the *real* road structure matters.
- **Attention inspection:** for a congested sensor, plot its top-attended neighbours on the map. Are they the downstream ones?

**Deliverables**
- [ ] STGAT trained, 3 seeds
- [ ] DCRNN trained, within ~5% of reference
- [ ] Ablation table

---

## Phase 6 — Fair comparison

**Goal.** Every model is scored by **one function**, on the same test windows, with the same mask, in mph. That single `evaluate()` is the backbone of your credibility.

### Fairness contract

| Rule | How it is enforced |
|---|---|
| Same chronological test period | Every model reads `test.npz` from Phase 2 |
| Equal history window | 12 steps (60 min) for every model, including tree lags |
| Same masking | Metrics only where mask = 1 |
| Same tuning budget | Each model tuned on val only; similar number of trials |
| Same units | Inverse-scaled to mph before metrics |

```python
def evaluate(y_pred, y_true, mask, horizons=(3, 6, 12)):   # steps → 15, 30, 60 min
    out = {}
    for h in horizons:
        p, t, m = y_pred[:, h-1], y_true[:, h-1], mask[:, h-1]
        e = (p - t)[m]
        out[f"MAE@{h*5}"]  = np.abs(e).mean()
        out[f"RMSE@{h*5}"] = np.sqrt((e**2).mean())
        out[f"MAPE@{h*5}"] = (np.abs(e) / t[m]).mean() * 100      # mask also removes t = 0
    return out
```
Also report all 12 horizons as a line plot (error vs minutes ahead, one line per model).

**Peak vs off-peak.** Defined on the **timestamp being predicted**, weekdays only: AM peak 07:00–09:59, PM peak 16:00–19:59, off-peak = other weekday hours; weekends reported separately. Phase 1 confirmed these windows sit inside the observed dips. State the definition in the report.

### Final results table

| Model | MAE 15m | MAE 30m | MAE 60m | RMSE 60m | MAPE 60m | Peak MAE 60m | Off-peak MAE 60m |
|---|---|---|---|---|---|---|---|
| Persistence | | | | | | | |
| Historical average | | | | | | | |
| Ridge | | | | | | | |
| XGBoost (per segment) | | | | | | | |
| Multi-output RF | | | | | | | |
| XGB + neighbours | | | | | | | |
| LSTM (no graph) | | | | | | | |
| **STGAT (ours)** | | | | | | | |
| DCRNN | | | | | | | |

### Make the claim statistically
- Mean ± std over 3 seeds for neural models.
- **Diebold–Mariano test** (or paired bootstrap over test *days*, since errors within a day are correlated) between the best baseline and the best GNN at each horizon. "GNN beats XGBoost at 60 min, p < 0.01" is far stronger than a bare number.
- Break the gain down: by horizon, by peak/off-peak, by node degree (expect bigger gains at well-connected sensors), and at congestion onsets (Phase 4).

**Hypothesis (write it down before you look):** at 5–15 min tree models are close to GNNs; at 60 min and in peaks the graph models win clearly. If you see something else, report it honestly and explain it. That earns more respect than a cherry-picked win.

**Deliverables**
- [ ] `results/final.csv` + horizon line plot
- [ ] Peak/off-peak bar chart
- [ ] Significance tests

---

## Phase 7 — Congestion maps and visuals

**Goal.** The brief asks for congestion maps; they also make the project memorable on a portfolio. Coordinates from `graph_sensor_locations.csv`.

**Congestion level** per sensor, relative to its own free-flow speed (85th percentile of *train* speeds):

| Level | speed / free-flow | Colour |
|---|---|---|
| Free | ≥ 0.8 | Green (+ label) |
| Slow | 0.5–0.8 | Amber (+ label) |
| Congested | < 0.5 | Red (+ label) |
| Sensor down | masked | Grey (+ label) |

(Status colours always come with a text label or icon, so colour-blind readers aren't left out. `src/viz/style.py` already defines them.)

**Figures to build**
1. **Side-by-side map** (folium or plotly `scatter_mapbox`): Actual | XGBoost forecast | GNN forecast, one PM-peak test moment, 30 min ahead. The reader should see the GNN catching the red zone XGBoost misses.
2. **Animated map** over a test evening (17:00–20:00), slider for horizon. Export as HTML for the portfolio.
3. **Error map:** per-sensor MAE difference (XGB − GNN) on a diverging blue–red scale with grey at 0. Expect GNN-wins clusters at junctions.
4. **Time-space diagram:** a corridor of ~15 consecutive sensors on one freeway; x = time, y = position, colour = speed. Congestion waves appear as diagonal stripes moving *upstream* (Phase 1 predicts the slope: ~16 km/h). Actual vs predicted.
5. **Attention map:** for one congested sensor, edges to neighbours with width = GAT attention.
6. **Horizon curves + peak/off-peak bars** from Phase 6.

Optional polish: a small **Streamlit** app (pick test timestamp + horizon → maps), deployed free on Streamlit Community Cloud and linked from the README.

**Deliverables**
- [ ] Figures 1–6 in `reports/figures/`
- [ ] Interactive HTML map / Streamlit demo

---

## Phase 8 — *Optional* simulated signal-control experiment

**Hard rule from the brief: simulation only.** METR-LA sensors sit on freeways, which have no traffic lights, so you cannot replay METR-LA through signals. Run the control experiment on a small **synthetic** arterial network in SUMO, and say so plainly.

**Setup**
1. Network: a 3×3 grid of signalised intersections, `netgenerate --grid --grid.number=3`, or a small LA arterial area from OpenStreetMap with `osmWebWizard.py`.
2. Demand: `randomTrips.py` with a time-varying rate shaped like the **Phase 1 weekday daily profile** (that is where the METR-LA EDA connects to the simulation).
3. Detectors: SUMO induction loops (E1) on every approach, aggregated to 5-min speed/flow. This gives a small graph dataset in the same format as METR-LA.
4. Train the Phase 5 STGAT on simulated data from some seeds; test on unseen seeds.

**Controllers (via TraCI)**

| Controller | Rule |
|---|---|
| Fixed-time | SUMO default cycle (e.g. 90 s, equal splits) |
| Actuated / max-pressure | Reacts to current queues only |
| **Forecast-aware (ours)** | Every 5 min, predict 15-min-ahead speed per approach; give more green to approaches forecast to congest |

```python
import traci
traci.start(["sumo", "-c", "sim/grid.sumocfg"])
while traci.simulation.getMinExpectedNumber() > 0:
    traci.simulationStep()
    if traci.simulation.getTime() % 300 == 0:          # every 5 min
        x = collect_detector_window()                  # last 12 steps
        pred = model(x)                                # (12, N)
        for tls in traci.trafficlight.getIDList():
            set_split(tls, pred_for_approaches(tls, pred, h=3))
```
**Metrics:** average travel time, average waiting time, total queue length, throughput. ≥ 10 random seeds per controller; mean ± 95% CI. Stretch goal: a DQN agent in **CityFlow** with forecasts in its state (connects to the Zheng 2019 pre-read).

**Wording to use:** *"In a SUMO simulation of a synthetic 9-intersection grid, a forecast-aware controller reduced mean waiting time by X% vs fixed-time."* Never "controls LA traffic".

Install: `brew install --cask sumo` (or the official macOS installer), then `pip install traci sumolib`; set `SUMO_HOME`.

**Deliverables**
- [ ] SUMO network + demand
- [ ] Three controllers, ≥ 10 seeds each
- [ ] Results table + one simulation GIF

---

## Phase 9 — Extensions, packaging and placement prep

**Extensions (pick 1–2)**
- **PEMS-BAY** (325 sensors; `pems-bay.h5` is already in `data/raw/`): rerun the whole pipeline by changing one config. Shows generalisation.
- **Flow and occupancy:** pull raw PeMS station data (pems.dot.ca.gov, free account) for one district. This finally covers the brief's flow/occupancy features, for both the baselines and the GNN.
- **Graph WaveNet / learned adjacency** as an ablation against the fixed graph (Phase 1 showed 62% of fixed links are weak, so a learned graph may help).
- **Uncertainty:** quantile loss (10/50/90%) so maps show confidence bands.
- **Efficiency:** latency and parameter count per model; ONNX export of STGAT.

**Engineering polish (what recruiters actually open)**
- `make data`, `make train MODEL=stgat`, `make eval` for one-command reproduction.
- YAML/Hydra configs, W&B dashboard link, Dockerfile, GitHub Actions running `pytest` + `ruff`.
- Pinned `requirements.txt`, seeds, model cards for STGAT and DCRNN.
- README in this order: one-line result → hero congestion-map GIF → results table → how to run → method → limitations (speed-only data, simulated control, 2012 data).
- A 4–6 page arXiv-style report or a blog post; a 2-minute demo video.

**Resume bullets (fill with your real numbers)**
- Built a spatio-temporal Graph Attention Network (PyTorch Geometric) for traffic forecasting on the 207-sensor METR-LA network; reduced 60-min MAE by X% vs a tuned per-segment XGBoost baseline, with gains concentrated in peak hours (p < 0.01).
- Designed a leakage-safe evaluation pipeline (chronological splits with purge gaps, train-only scaling, masked metrics for sensor outages) benchmarking 9 models across 12 forecast horizons.
- Showed empirically that congestion propagates upstream at ~16 km/h on LA freeways and used it to design bidirectional graph attention; ablations confirmed the real road graph beats a random graph by Y%.
- Simulated forecast-aware signal control in SUMO, cutting mean waiting time by Z% vs fixed-time plans on a 9-intersection grid.

**Deliverables**
- [ ] One extension done
- [ ] README, CI, Docker, report, demo video
- [ ] Resume bullets with real numbers

---

## Interview question bank

| Question | Key point of your answer |
|---|---|
| Why not a random split? | Overlapping windows share 11/12 timestamps; a random split leaks the future into train |
| What is a purge gap? | Dropping boundary windows so no target timestamp is in two splits; the DCRNN reference doesn't do it |
| Why mask zeros? | 0 mph = sensor down; 6.3 of the 8.1 points of zeros are whole-network outages. Training on them teaches fake city-wide jams, and MAPE explodes |
| Why fit the scaler on train only? | Val/test statistics would leak into training |
| Which direction does congestion travel? | Mostly **backwards** (downstream sensor first) at ~16 km/h. Phase 1 measured it: 156 vs 41 links |
| How did that change your model? | Added reverse edges with a direction feature so GAT hears downstream neighbours; tested it as an ablation |
| GAT vs GCN vs diffusion conv? | Learned neighbour weights vs fixed normalised weights vs directed multi-hop random walks |
| GAT vs GATv2? | GAT attention is static (same neighbour ranking for every query); GATv2 applies the nonlinearity before `a`, which makes it dynamic |
| Why did the GNN win (or not) at 5 min? | Short horizons are dominated by the sensor's own recent history |
| Why include an LSTM with no graph? | To separate "deep learning helps" from "the graph helps" |
| How do you know the graph matters? | Random-graph and identity-graph ablations |
| How do you deploy this for real? | Streaming features, cold-start sensors, drift monitoring, graph changes, fallback to persistence |
| How would it scale to 10k sensors? | Sparse ops, neighbour sampling, graph partitioning (DCRNN was scaled to 11,160 sensors this way), mini-batching over subgraphs |
| What would you do with 2 more weeks? | Learned adjacency, uncertainty, PeMS flow/occupancy |

## Pitfalls that sink traffic projects
- Random train/test split of windows → inflated scores that collapse under questioning.
- Treating 0 mph as real traffic → fake jams; MAPE division by zero.
- Scaler fitted on the full series → silent leakage.
- Sensor column order ≠ adjacency order → the GNN trains on a scrambled graph and *still trains*, so it's hard to spot. (`load_all()` now raises.)
- Comparing a GNN with 12 steps of history against XGBoost with 3 lags → unfair; keep 12 for all.
- Reporting only average MAE → hides where the graph helps; always split by horizon and peak.
- Weak baselines → a GNN "win" nobody believes; tune XGBoost properly.
- One-directional message passing on a directed graph → the model can't hear the downstream jam.
- Claiming real signal control → violates the brief.

## Timeline
About 50 days at 2–3 focused hours a day. Phase 5 is the longest and most important block. Do not start a phase until the gate before it passes.

## Sources
- liyaguang/DCRNN repository: data, sensor graph, data-prep scripts, reference config and METR-LA results. https://github.com/liyaguang/DCRNN
- Veličković et al. (2018), *Graph Attention Networks*, ICLR.
- Zheng et al. (2019), *CityFlow: A Multi-Agent Reinforcement Learning Environment for Large Scale City Traffic Scenario*, WWW.
- Li et al. (2018), *Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting*, ICLR.
- Wu et al. (2019), *Graph WaveNet for Deep Spatial-Temporal Graph Modeling*, IJCAI.
- Brody, Alon & Yahav (2022), *How Attentive are Graph Attention Networks?*, ICLR.
