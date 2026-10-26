# Phase 2 report: preprocessing without leakage

**Status:** Complete · **Code:** `src/data/preprocess.py`, `src/data/dataset.py`, `src/eval/metrics.py` · **Config:** `configs/data/metr_la.yaml` · **Tests:** `tests/test_leakage.py` (12 tests)

Re-run with `make preprocess` (or `python -m src.data.preprocess --protocol strict|dcrnn`).

## What it produces
`data/processed/<protocol>/{train,val,test}.npz`, `scaler.json`, and a shared `data/processed/adj.npz`.

| Array | Shape | Meaning |
|---|---|---|
| `X` | (S, 12, 207, 7) | speed_z, mask, sin/cos time-of-day, sin/cos day-of-week, weekend |
| `Y` | (S, 12, 207) | next 12 steps (5–60 min) in **mph**; **NaN** where the sensor was down |
| `M_y` | (S, 12, 207) | target mask (True = real reading) used by every loss and metric |
| `origin_idx`, `target_time` | (S,), (S, 12) | for peak / off-peak breakdowns in Phase 6 |

`adj.npz` holds the dense directed adjacency, the PyG-style `edge_index` / `edge_weight`, and the sensor order.

## Results (strict protocol)

| Split | Windows | Targets from → to | Masked targets |
|---|---|---|---|
| train | 22,933 | 2012-03-01 01:00 → 05-23 07:05 | 2.96% |
| val | 3,417 | 05-23 07:10 → 06-04 04:45 | 6.98% |
| test | 6,843 | 06-04 04:50 → 06-27 23:55 | 12.12% |

- **Purge:** 11 windows lost per boundary; no target timestamp appears in two splits.
- **Dropped train windows** (targets > 90% masked, mostly whole-network outages): 1,034. Val and test keep every window; metrics ignore masked points.
- **Scaler (train timestamps only, outages excluded):** mean 58.58 mph, std 12.82 mph.
- **Adjacency:** rebuilt from `distances_la_2012.csv` (σ = std of finite road distances, κ = 0.1) and asserted equal to `adj_mx.pkl`; still directed.

## Strict vs the DCRNN reference protocol
Both are built so Phase 6 can report each once.

| | strict (used) | dcrnn (reference) |
|---|---|---|
| Split unit | timestamps, then windows | windows |
| Val targets begin | 05-23 07:10, after the last train target (07:05) | **05-23 06:50, before the last train target (07:40)** → overlap |
| Scaler | mean 58.58, std 12.82 (outages excluded) | mean 54.41, std 19.50 (0-mph outages averaged in) |
| Train windows dropped for outages | 1,034 | 0 |

## The gate: leakage tests
- [x] target timestamps: train < val < test, strictly
- [x] no target timestamp in two splits (and the dcrnn protocol is shown to overlap, so the purge is proven necessary)
- [x] scaler equals the NaN-mean/std of the train slice, and is unchanged when val/test data is destroyed
- [x] inputs never see the future: altering readings after t₀ does not change any window ending at or before t₀
- [x] masked positions contribute exactly zero to MAE/RMSE/MAPE (perturbed with NaN targets and absurd forecasts)
- [x] masked targets are NaN (never 0 mph); inputs finite, with an explicit mask channel
- [x] sensor order in X equals the adjacency order
- [x] rebuilt adjacency equals `adj_mx.pkl`
- [x] shapes: X (S,12,207,7), Y (S,12,207); sizes equal block length minus 11
- [x] cyclic features: 23:55 and 00:05 are neighbours

## Design notes
- **Why purge?** Overlapping windows share 11 of 12 timestamps, so a boundary-straddling window puts near-copies of val/test targets in train.
- **Imputation:** a down sensor is forward-filled from its last real reading inside the same 12-step input window; if none, the train mean of that (sensor, time-of-day slot). The mask channel tells the model the value is imputed.
- **Inputs may look back across a block boundary** (e.g. the first val window's inputs end in train data). That is the past, not a label, and matches deployment.
- **Storage:** X is stored as float32 and is ~2.5 GB per protocol. `data/processed/` is git-ignored.

## Hand-off to Phase 3
Load with `src.data.dataset.load_split("train")`, `load_scaler()`, `inverse_scale()`; score with `src.eval.metrics`. Phase 3 feature tables must use the same 12-step history and the same masks.
