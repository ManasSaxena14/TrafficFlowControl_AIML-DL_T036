# Graph WaveNet: Wu, Pan, Long, Jiang, Zhang, IJCAI 2019 (optional pre-read)
*Graph WaveNet for Deep Spatial-Temporal Graph Modeling*

**Read for:** a **learned adaptive adjacency**, i.e. why a fixed distance graph may miss hidden links. This is our Phase 9 ablation.

## Core ideas
1. **Self-adaptive adjacency.** Learn two node-embedding matrices E₁, E₂ and build
   `Ã_adp = SoftMax(ReLU(E₁ E₂ᵀ))`.
   No prior knowledge needed; it discovers dependencies the road distance misses (e.g. two freeways fed by the same on-ramp traffic) and down-weights links that don't matter.
2. **Graph convolution = diffusion over the given forward/backward transition matrices *plus* the adaptive one.**
3. **Temporal modelling with dilated causal convolutions** (gated TCN, as in WaveNet), not RNNs. Receptive field grows exponentially with depth; it trains in parallel and is faster than DCRNN.
4. Outputs all 12 horizons at once (non-autoregressive), so there is no error accumulation from feeding back predictions.

## Why it matters for us
Phase 1 found **62% of the fixed graph's links barely co-move**. That is direct evidence the fixed adjacency is noisy and incomplete. The brief requires the fixed graph for the main model, so the learned graph is only an **ablation**: "fixed graph vs fixed + adaptive". If the adaptive graph helps, plot the learned links that are *not* in the road graph on the map: a great figure.

## Reported METR-LA results (paper)
MAE ≈ 2.69 / 3.07 / 3.53 at 15 / 30 / 60 min, roughly on par with DCRNN's post-fix numbers, with faster training.

## Write-down for interviews
- `A_adp = softmax(relu(E₁E₂ᵀ))`: learned, directed, dense-then-sparsified.
- "Fixed graph = physical prior; adaptive graph = data-driven correction. Use both."
- TCN vs RNN: parallel training, no vanishing gradients through time, explicit receptive field.
