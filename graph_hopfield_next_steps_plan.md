# Graph Hopfield Networks (GHN): Next Steps Plan

This is a detailed, implementation-oriented plan for turning the **Graph Hopfield Networks** idea (Hopfield/associative retrieval + graph Laplacian coupling) into a **working PyTorch Geometric layer** and a **clean benchmark suite** for a paper-quality result.

---

## 0) Decide what your “memories” are (this determines everything)

You need to choose what the memory patterns $M_i$ mean for graphs. For a first paper, pick something that:

- is clearly “associative retrieval”
- is implementable in ~1 day
- gives a crisp corruption story + ablations

### Recommended: global memory bank of learnable prototypes
Let $M \in \mathbb{R}^{K \times d}$ be **K learnable prototype vectors** (codebook slots). Each node state $x_v$ retrieves from this memory via Hopfield/attention:

$$
r_v = M^\top \;\mathrm{softmax}(\beta M x_v)
$$

Then graph-couple it with Laplacian smoothing and residual mixing.

**Why this is the best “first pass”:**
- No need to store per-graph memories or build subgraph dictionaries.
- Clean ablations: remove memory, remove smoothing, compare to attention/MLP.
- Prototypes are interpretable later (nearest neighbors, class alignment).

**Alternative (strong denoising framing):** memory bank = clean augmentations (BYOL-style). Stronger story but more engineering.

---

## 1) Implement the Graph Hopfield Layer (two equivalent routes)

You have two practical implementation routes.

### Route A (recommended): unrolled fixed-point / gradient-style steps
Treat the layer as $T$ iterations of:
1) Hopfield retrieval per node  
2) Graph smoothing/mixing step  
3) Residual + normalization  

A clean update:

$$
X^{t+1} = \mathrm{Norm}\Big((1-\alpha)X^t + \alpha R(X^t) - \eta \cdot 2\lambda L X^t\Big)
$$

- $R(X)$ applies retrieval row-wise.
- $L$ is a Laplacian-like operator (often normalized).

**Implementation details**
- Store prototypes as rows of `M` (shape `K x d`).
- Retrieval:
  - `R = softmax(beta * (X @ M.T)) @ M`
- Laplacian term efficiently:
  - Use `L = D - A` as `DX - AX`, or normalized $L_\text{sym} = I - D^{-1/2} A D^{-1/2}$.
  - `AX` via sparse SpMM using `edge_index` (PyG / torch_sparse).
  - `DX` via degree-vector multiply.

**Hyperparams to tune**
- `K` (memory size): 16, 32, 64, 128
- `beta`: 1, 2, 5, 10
- `lambda`: 0, 0.1, 1, 5 (scale depends on Laplacian normalization)
- `T`: 1–3 (start with 1 or 2)
- `alpha`, `eta`: mixing/step-size scalars (or learn them)

This route is stable and easy to debug.

### Route B (optional later): implicit fixed-point solver
Solve $X = T(X)$ using Anderson acceleration / implicit differentiation.

Cool if you want a stronger “energy minimization” angle, but adds failure modes. Only do this after Route A works.

---

## 2) Build the model wrapper (how GHN competes with GNNs)

You want fair baselines and a minimal architecture that still tests the idea.

### Minimal model: shallow encoder + GHN block + classifier
- `Input MLP` (features → $d$)
- `GHNBlock` (as above; `T=1–3`)
- `Output linear` ($d$ → #classes)

This tests whether “memory + graph coupling” is already strong.

### Stronger variant: light GNN backbone + GHN
- Replace input MLP with 1–2 layers of GCN/GAT (light message passing)
- Insert GHN block
- Output head

This tests GHN as an add-on robustness module.

---

## 3) Define corruption benchmarks (make “corrupted retrieval” concrete)

Pick corruption protocols that are standard, reproducible, and clearly hurt GNNs.

Start with **2 tasks**.

### Task A: node classification under feature corruption
Datasets: Cora, Citeseer, Pubmed (Planetoid), optionally `ogbn-arxiv` later.

Corruptions:
1) **Feature masking**: randomly set a fraction `p` of node features to zero.
2) **Gaussian noise**: add `sigma * N(0,1)` to features.
3) Optional: **feature replacement**: replace masked features with features from random nodes.

Report accuracy vs corruption strength.

### Task B: link prediction under edge corruption
Use same datasets or an OGB link dataset.

Corruptions:
1) **Edge dropout**: drop a fraction `q` of edges.
2) **Edge additions**: add random edges to break locality.
3) **Rewiring**: degree-preserving random rewiring (optional if time).

Report AUC/AP vs corruption strength.

**Key:** sweep corruption levels and plot curves. Robustness is usually best shown as curves, not single points.

---

## 4) Baselines (small but defensible)

Use ~3 strong baselines + 2 essential ablations.

Baselines:
1) **GCN**
2) **GAT**
3) **GraphSAGE**

Ablations (must-have):
- **Memory-only**: set `lambda=0` (no graph coupling).
- **Graph-only**: remove retrieval (use identity or MLP), keep Laplacian smoothing.

Optional:
- “Attention MLP” baseline: per-node attention over prototypes but no iterative dynamics (helps show it’s not just extra parameters).

Fairness:
- Match hidden dim $d$ and report parameter counts.
- Match training budget (epochs, LR schedule, early stopping).

---

## 5) Ablations reviewers will expect

Minimum:
- `lambda` sweep (coupling matters)
- `beta` sweep (retrieval sharpness matters)
- `K` sweep (memory size)
- `T` iterations: 1 vs 2 vs 3
- Swap Laplacian smoothing with message passing and compare (optional but strong)

Interpretability nice-to-have:
- Prototype usage distribution per class
- Entropy of prototype attention vs corruption (does retrieval “sharpen” as inputs degrade?)

---

## 6) Implementation checklist (PyTorch Geometric)

### Data and splits
- Start with Planetoid default splits for speed.
- Add random splits later if needed.
- Use 3–10 seeds, report mean ± std.

### Efficient Laplacian application
Given `edge_index`:
- Compute degrees `deg`.
- `AX` with sparse SpMM: `spmm(edge_index, edge_weight, n, n, X)`.
- `DX = deg.unsqueeze(-1) * X`.
- Unnormalized: `LX = DX - AX`.
- Normalized: $L_\text{sym} X = X - D^{-1/2} A D^{-1/2} X$.

**Recommendation:** start normalized for stability.

### GHNBlock skeleton
- Params: prototypes `M (K x d)`, scalars (`beta`, `lambda`, `alpha`, `eta`) fixed or learned.
- Forward:
  - retrieval `R`
  - laplacian `LX`
  - update `X`
  - norm + dropout

Stability tricks:
- LayerNorm
- beta warmup (start small, ramp)
- weight decay on prototypes
- dropout on `X` or on attention weights

---

## 7) What “success” looks like (fast diagnostic)

You want a signal within 48 hours:

- Under feature masking `p=0.3–0.6`, GHN degrades **less** than GCN/GAT.
- Full GHN should beat **both** memory-only and graph-only ablations.
- If it doesn’t:
  - likely `lambda` scaling / Laplacian normalization is off,
  - prototypes collapse (use weight decay / diversity penalty),
  - beta is too sharp too early.

---

## 8) Timeline (aggressive but realistic)

### Days 1–2: layer + sanity tests
- Implement GHNBlock (`T=1`) + MLP head
- Overfit a tiny subset
- Check gradients, no NaNs, stable loss

### Days 3–4: node classification corruption curves
- Cora/Citeseer with masking + noise sweeps
- Generate first plots

### Days 5–7: baselines + ablations
- Add GCN/GAT/SAGE + memory-only + graph-only
- Run 3 seeds

### Week 2: link prediction + stronger variant
- Link prediction setup + edge corruption curves
- Optional: backbone + GHN insertion

### Week 3: paper packaging
- Final tables + plots + clear method section
- Add interpretability plots if time

---

## 9) One low-cost “paper pop” add-on
Add a **corruption-aware retrieval diagnostic**:
- Track average entropy of prototype attention per node as corruption increases.
- Show GHN falls back to stable prototypes when features are noisy.

This supports the associative memory claim visually.

---

## Suggested next action (today)
1) Implement `GHNBlock(T=1)` with prototypes + normalized Laplacian.  
2) Run Cora node classification with feature masking sweep and compare to GCN.  
3) If you see robustness improvement, expand to Citeseer + add ablations.
