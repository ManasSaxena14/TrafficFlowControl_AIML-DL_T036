# DCRNN: Li, Yu, Shahabi, Liu, ICLR 2018
*Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting*. This is the paper behind our dataset repo.

**Read for:** diffusion convolution on **directed** road graphs, and the METR-LA benchmark protocol that every later paper copies.

## Core idea
Model traffic as a **diffusion process** on the directed road graph: information spreads like a random walk, forward along the traffic direction and backward against it.

**Diffusion convolution** (K hops, both directions):
```
X ⋆_G f_θ = Σ_{k=0}^{K-1} ( θ_{k,1} (D_O⁻¹ W)^k  +  θ_{k,2} (D_I⁻¹ Wᵀ)^k ) X
```
- `D_O⁻¹ W`: forward random-walk transition (downstream). `D_I⁻¹ Wᵀ`: backward (upstream).
- The two directions have **separate** parameters. This "dual random walk" is why DCRNN can capture jams that travel backwards (Phase 1 measured exactly this).

**DCGRU:** replace every matrix multiplication inside a GRU cell with diffusion convolution. This gives spatial and temporal modelling in one cell.
**Seq2seq:** an encoder reads 12 steps; the decoder emits 12 steps one at a time.
**Scheduled sampling (curriculum):** during training the decoder is fed ground truth at first and its own predictions later. This reduces train/test mismatch.

## The METR-LA protocol (what "comparable to published numbers" means)
- 207 loop detectors, LA County freeways, Mar–Jun 2012, 5-min speed. (PEMS-BAY: 325 sensors, Jan–May 2017.)
- Graph: `W_ij = exp(−d_ij²/σ²)` for road distance i→j, zero below 0.1.
- 12-in / 12-out windows, **70/10/20 chronological split over windows**.
- Inputs: speed + time-of-day. Scaler mean/std computed on train **including zeros**.
- **Masked MAE**: targets equal to 0 are ignored in loss and metrics.
- Reference config: 2 layers × 64 units, K = 2, lr 0.01, batch 64, grad-norm 5.

**Results (repo README, after a 2018 bug fix): METR-LA MAE 2.67 / 3.08 / 3.56 at 15 / 30 / 60 min.** Phase 5 target: within ~5%.

## Protocol quirks we fix in strict mode (good interview material)
1. Splitting **windows** instead of timestamps → the last train targets overlap the first val targets (a small leak). → **Purge gap.**
2. Scaler mean includes outage zeros → biased low. → **NaN-aware train-only scaler.**
3. Inputs keep raw zeros with no mask channel. → **Impute + mask channel.**

## Write-down for interviews
- Forward/backward random walks, K hops, seq2seq 12→12, masked MAE.
- "Diffusion conv is directional and multi-hop with fixed transition weights; GAT is learned but single-hop per layer."
- The DCRNN line was later scaled to 11,160 California sensors with graph partitioning (Mallick et al.). Use this for "how would you scale it?"
