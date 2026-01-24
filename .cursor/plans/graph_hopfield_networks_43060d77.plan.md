---
name: Graph Hopfield Networks
overview: A structured plan to implement Graph Hopfield Networks with rigorous mathematical derivation, PyTorch Geometric implementation, and corruption benchmarks to validate that associative retrieval improves robustness under structured corruption.
todos:
  - id: setup-project
    content: Initialize project structure with requirements.txt (PyTorch, PyTorch Geometric, etc.)
    status: pending
  - id: math-notebook
    content: Create math derivation notebook with energy function, update dynamics, and convergence proof
    status: pending
  - id: hopfield-layer
    content: Implement GraphHopfieldLayer in PyTorch Geometric
    status: pending
  - id: memory-bank
    content: Implement memory bank module (learnable patterns)
    status: pending
  - id: baselines
    content: Implement baseline models (GCN, GAT, GraphSAGE) with matched capacity
    status: pending
  - id: corruption-utils
    content: Implement corruption utilities (feature noise, edge noise, label corruption)
    status: pending
  - id: training-pipeline
    content: Create training and evaluation scripts
    status: pending
  - id: run-clean-baselines
    content: Run baselines on clean datasets to establish baseline accuracy
    status: pending
  - id: run-corruption-benchmarks
    content: Run full corruption benchmark experiments
    status: pending
  - id: run-ablations
    content: Run ablation studies (beta, lambda, iterations, memory size)
    status: pending
  - id: analysis-viz
    content: Create analysis notebooks with robustness curves and ablation plots
    status: pending
isProject: false
---

# Graph Hopfield Networks Research Implementation Plan

## Defensible Claim

"Adding associative retrieval (Hopfield/attention-style) as an explicit denoising memory operator improves robustness of graph prediction under structured corruption, beyond standard message-passing baselines at matched capacity."

---

## Phase 1: Mathematical Foundation

### 1.1 Core Energy Function Derivation

**Modern Hopfield Network (Reference)**:

- Energy: `E_MH(ξ) = -β⁻¹ log Σᵢ exp(β · ξᵀmᵢ) + ½||ξ||²`
- Update: `ξ_new = Softmax(β · Mᵀξ) · M` (attention-weighted pattern retrieval)

**Graph Hopfield Energy Function**:

```
E_GH(X) = Σᵥ [-β⁻¹ lse(β, MᵀXᵥ) + ½||Xᵥ||²] + λ Σ_(u,v)∈E ||Xᵤ - Xᵥ||²
```

Where:

- First term: Hopfield retrieval energy per node (attracts to stored patterns)
- Second term: Graph Laplacian regularization (enforces neighbor smoothness)
- M: Memory bank of stored patterns (can be learnable or fixed)
- β: Inverse temperature (controls pattern selectivity)
- λ: Graph smoothness weight

**Equivalent Matrix Form**:

```
E_GH(X) = -β⁻¹ Σᵥ lse(β, MᵀXᵥ) + ½||X||²_F + λ · tr(XᵀLX)
```

where L is the graph Laplacian.

### 1.2 Update Dynamics

**Gradient of Energy**:

```
∇_Xᵥ E_GH = -M · Softmax(β · MᵀXᵥ) + Xᵥ + 2λ Σ_u∈N(v) (Xᵥ - Xᵤ)
```

**Fixed-Point Iteration (Graph Hopfield Update)**:

```
Xᵥ ← M · Softmax(β · MᵀXᵥ) - 2λ Σ_u∈N(v) (Xᵥ - Xᵤ)
```

Or with momentum/damping for stability:

```
Xᵥ ← (1-α)Xᵥ + α[M · Softmax(β · MᵀXᵥ) - 2λ(Dᵥᵥ Xᵥ - Σ_u∈N(v) Xᵤ)]
```

### 1.3 Convergence Analysis

**Theorem (Convergence)**: The Graph Hopfield energy E_GH(X) is a Lyapunov function for the update dynamics, guaranteeing convergence to a local minimum.

**Proof Sketch**:

1. The log-sum-exp term is convex in X (composition of convex functions)
2. The quadratic regularization ½||X||² is strictly convex
3. The Laplacian term tr(XᵀLX) is convex (L is positive semi-definite)
4. Sum of convex functions is convex, gradient descent converges

**Stability Condition**: For guaranteed convergence with step size η:

```
η < 2 / (β||M||² + 1 + 2λ·d_max)
```

where d_max is the maximum node degree.

### 1.4 Connection to Attention

The update rule can be written as:

```
X ← Attention(X, M, M) + GraphSmooth(X, A)
```

Where:

- `Attention(Q, K, V) = Softmax(βQKᵀ)V` is standard self-attention
- `GraphSmooth(X, A) = -λLX = -λ(D-A)X` is graph diffusion

This makes the Graph Hopfield layer interpretable as: **"attention-based pattern retrieval + graph-aware smoothing"**.

---

## Phase 2: Implementation (PyTorch Geometric)

### 2.1 Project Structure

```
graph-hopfield-networks/
├── src/
│   ├── __init__.py
│   ├── layers/
│   │   ├── __init__.py
│   │   ├── graph_hopfield.py      # Core GH layer
│   │   └── memory_bank.py         # Pattern storage
│   ├── models/
│   │   ├── __init__.py
│   │   ├── ghn.py                 # Graph Hopfield Network
│   │   └── baselines.py           # GCN, GAT, GraphSAGE
│   ├── data/
│   │   ├── __init__.py
│   │   ├── corruption.py          # Noise injection utilities
│   │   └── datasets.py            # Dataset loaders
│   └── utils/
│       ├── __init__.py
│       └── metrics.py             # Evaluation metrics
├── experiments/
│   ├── train.py                   # Training script
│   ├── evaluate.py                # Evaluation script
│   └── configs/                   # Experiment configs
├── tests/
│   └── test_layers.py             # Unit tests
├── notebooks/
│   └── math_derivation.ipynb      # Interactive math exploration
├── requirements.txt
└── README.md
```

### 2.2 Graph Hopfield Layer Implementation

```python
class GraphHopfieldLayer(nn.Module):
    def init(self, in_dim, out_dim, num_patterns, beta=1.0, lambda_graph=0.1):
        super().__init__()
        self.memory = nn.Parameter(torch.randn(num_patterns, out_dim))
        self.proj_in = nn.Linear(in_dim, out_dim)
        self.beta = beta
        self.lambda_graph = lambda_graph
    
    def forward(self, x, edge_index, num_iterations=1):
        x = self.proj_in(x)
        for _ in range(num_iterations):
            # Hopfield retrieval
            scores = self.beta * x @ self.memory.T
            attn = F.softmax(scores, dim=-1)
            hopfield_out = attn @ self.memory
            
            # Graph smoothing via message passing
            row, col = edge_index
            deg = degree(col, x.size(0), dtype=x.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
            smooth_out = propagate(edge_index, x=x, norm=norm)
            
            # Combined update
            x = hopfield_out + self.lambda_graph * (smooth_out - x)
        return x
```

### 2.3 Baselines (Matched Capacity)

For fair comparison, all models should have approximately the same number of parameters:

- **GCN**: 2-layer GCN with hidden_dim chosen to match GHN params
- **GAT**: 2-layer GAT with attention heads
- **GraphSAGE**: 2-layer SAGE with mean aggregation

---

## Phase 3: Corruption Benchmarks and Ablations

### 3.1 Datasets

- **Cora** (2,708 nodes, 5,429 edges, 7 classes)
- **Citeseer** (3,327 nodes, 4,732 edges, 6 classes)
- **Pubmed** (19,717 nodes, 44,338 edges, 3 classes)

### 3.2 Corruption Types

| Type | Description | Implementation |

|------|-------------|----------------|

| **Feature Noise** | Gaussian perturbation to node features | `X_noisy = X + σ * N(0,1)` |

| **Edge Noise** | Random edge addition/removal | `flip_edges(A, p)` |

| **Label Corruption** | Flip training labels | `y_noisy = flip(y, p)` |

Corruption levels: 0%, 10%, 20%, 30%, 40%, 50%

### 3.3 Ablation Studies

1. **β (inverse temperature)**: Test β ∈ {0.1, 0.5, 1.0, 2.0, 5.0, 10.0}

   - Hypothesis: Higher β → sharper pattern selection → better denoising

2. **λ (graph smoothness)**: Test λ ∈ {0.0, 0.01, 0.1, 0.5, 1.0}

   - λ=0 → pure Hopfield (baseline within ablation)
   - Hypothesis: Optimal λ balances retrieval with graph structure

3. **Number of iterations**: Test T ∈ {1, 2, 3, 5}

   - Hypothesis: More iterations → better energy minimization → cleaner retrieval

4. **Memory size**: Test |M| ∈ {num_classes, 2×num_classes, 5×num_classes}

### 3.4 Evaluation Metrics

- **Accuracy**: Classification accuracy on test set
- **Robustness Curve**: Accuracy vs. corruption level
- **Area Under Robustness Curve (AURC)**: Single metric for robustness
- **Convergence Analysis**: Energy values across iterations

### 3.5 Expected Results Table

| Model | Clean Acc | 20% Feat | 40% Feat | 20% Edge | AURC |

|-------|-----------|----------|----------|----------|------|

| GCN | ~81% | ~75% | ~68% | ~72% | X |

| GAT | ~83% | ~77% | ~70% | ~74% | X |

| **GHN** | ~82% | **~80%** | **~76%** | **~78%** | **Best** |

---

## Timeline Suggestion

- **Days 1-4**: Complete math derivation, create derivation notebook
- **Days 5-10**: Implement core layers, baselines, and training pipeline
- **Days 11-17**: Run experiments, corruption benchmarks, ablations
- **Days 18-21**: Analysis, visualization, write-up

---

## Key Files to Create First

1. `src/layers/graph_hopfield.py` - Core layer implementation
2. `src/data/corruption.py` - Corruption utilities
3. `experiments/train.py` - Training loop
4. `notebooks/math_derivation.ipynb` - Math documentation