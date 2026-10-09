# Graph Attention Networks: Veličković et al., ICLR 2018

**Read for:** how a node weights its neighbours with *learned* attention. Phase 5's STGAT is built on this.

## Core idea
A GCN averages neighbours with **fixed** weights from the graph structure (degree normalisation). GAT lets each node **learn how much to listen to each neighbour**, from the neighbours' features.

## The layer
For node *i* with neighbourhood 𝒩(i) (including itself):

1. Shared linear map: `z_j = W h_j`
2. Unnormalised score: `e_ij = LeakyReLU(aᵀ [z_i ‖ z_j])` (LeakyReLU slope 0.2)
3. Normalise over neighbours: `α_ij = softmax_j(e_ij) = exp(e_ij) / Σ_{k∈𝒩(i)} exp(e_ik)`
4. Aggregate: `h_i' = σ( Σ_{j∈𝒩(i)} α_ij z_j )`

**Multi-head attention:** run K independent heads. Hidden layers **concatenate** the K outputs; the final layer **averages** them. Heads stabilise training, like multiple filters in a CNN.

## Why masking to the adjacency matters
Attention is computed **only over graph neighbours** ("masked attention"). That is how the graph structure enters the model at all. Without the mask it would be a Transformer over all nodes, with no road structure. In our project, the mask is the METR-LA adjacency. Message direction follows `edge_index` (source → target), which is why STGAT adds reverse edges (Phase 1 finding: the downstream neighbour carries the early warning).

## Properties worth quoting
- Cost per layer is O(|V|·F·F' + |E|·F'), parallel across edges; no eigendecomposition, unlike spectral GCNs.
- Works inductively (on unseen graphs), because the weights are shared and don't depend on a fixed Laplacian.
- Attention weights are inspectable, which is our Phase 5 "attention map" ablation.

## Follow-up you must know: GATv2 (Brody, Alon, Yahav, ICLR 2022)
In GAT, `aᵀ[W h_i ‖ W h_j]` splits into a term for i plus a term for j, and LeakyReLU is monotonic. So **every query node ranks its neighbours in the same order**: the attention is *static*. GATv2 applies the nonlinearity first, `e_ij = aᵀ LeakyReLU(W [h_i ‖ h_j])`, which makes the attention *dynamic*. PyG: `GATv2Conv`. We use GATv2.

## Write-down for interviews
- The attention formula above, and the concat-vs-average rule for heads.
- "GCN = fixed neighbour weights; GAT = learned, feature-dependent weights; diffusion conv = directed multi-hop random walks."
- Why it suits METR-LA: 62% of distance-based links barely co-move (Phase 1), so learned weights can ignore them.
