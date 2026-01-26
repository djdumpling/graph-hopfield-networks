# Graph Hopfield Networks - Project Scratchpad

## Background and Motivation

This is a 3-week research project for an associative memory workshop. The goal is to implement Graph Hopfield Networks (GHN) that combine modern continuous Hopfield networks with graph neural networks.

**Defensible Claim**: "Adding associative retrieval (Hopfield/attention-style) as an explicit denoising memory operator improves robustness of graph prediction under structured corruption, beyond standard message-passing baselines at matched capacity."

**Key Innovation**: Combining Hopfield retrieval (attention over memory patterns) with graph Laplacian regularization to create a denoising layer for graph data.

---

## Key Challenges and Analysis

### Mathematical Foundation
- Energy function: E_GH(X) = Σᵥ [-β⁻¹ lse(β, MᵀXᵥ) + ½||Xᵥ||²] + λ Σ_(u,v)∈E ||Xᵤ - Xᵥ||²
- Convergence: Energy is a Lyapunov function with gradient descent convergence guarantees
- Connection to attention: GHN update = Attention(X, M, M) + GraphSmooth(X, A)

### Implementation Decisions
1. **Memory bank**: Learnable prototypes (K patterns of dimension d)
2. **Laplacian**: Symmetric normalized for stability
3. **Update**: Damped fixed-point iteration with LayerNorm
4. **Framework**: PyTorch Geometric for sparse operations

---

## High-level Task Breakdown

### Phase 1: Mathematical Foundation ✅
- [x] Derive energy function
- [x] Derive update dynamics  
- [x] Create LaTeX derivation document (phase1_mathematical_foundation.tex)
- [x] Fix proof issues per GPT-5.2 Pro review:
  1. ✅ Removed extra β⁻¹ multiplying lse in energy definition (Eq. 6, 7, and all subsequent uses)
  2. ✅ Replaced all M^T x inside lse with Mx (dimension correctness)
  3. ✅ Corrected stability condition (split into normalized and unnormalized Laplacian cases)
  4. ✅ Cleaned up coercivity proof to use official lse definition with correct β⁻¹ factor
  5. ✅ Added missing Cauchy-Schwarz inequality step in coercivity proof
  6. ✅ Fixed factor-of-2 mismatch in GraphSmooth definition
  7. ✅ Fixed Lemma 5.3 proof (added proper PSD-based inequality derivation)
- [x] Make proof reviewer-proof:
  1. ✅ Added explicit graph convention assumptions (undirected, edges counted once, symmetric A)
  2. ✅ Clarified Laplacian trace identity with count-once convention
  3. ✅ Added disclaimer separating GD guarantees from fixed-point contraction guarantees
  4. ✅ Changed "strictly decreasing" to "non-increasing" language in GD theorem
  5. ✅ Fixed attention multiplication order (softmax(QK^T)V not V·softmax(QK^T))
  6. ✅ Added note on Laplacian nullspace in strong convexity discussion

### Phase 2: Implementation ✅
- [x] Project structure and requirements.txt
- [x] MemoryBank module
- [x] GraphHopfieldLayer module
- [x] GraphHopfieldNetwork model
- [x] Baseline models (GCN, GAT, GraphSAGE)
- [x] Dataset loaders (Planetoid: Cora, Citeseer, Pubmed)
- [x] Corruption utilities (feature noise, edge corruption, label flip)
- [x] Training pipeline with early stopping
- [x] Evaluation metrics (accuracy, AURC)
- [x] Unit tests

### Phase 3: Experiments 🔄 (Ready to Run)
- [ ] Run clean baselines on all datasets
- [ ] Run corruption benchmarks
- [ ] Run ablation studies (beta, lambda, iterations, num_patterns)
- [ ] Generate robustness curves and tables
- [ ] Analysis and visualization

---

## Project Status Board

### Completed
- [x] setup-project: Initialize project structure
- [x] math-notebook: Create math derivation (LaTeX)
- [x] hopfield-layer: Implement GraphHopfieldLayer
- [x] memory-bank: Implement MemoryBank module
- [x] baselines: Implement GCN, GAT, GraphSAGE
- [x] corruption-utils: Implement corruption utilities
- [x] training-pipeline: Create train.py and configs
- [x] tests: Create unit tests

### Ready to Execute
- [ ] run-clean-baselines: Run baselines on clean datasets
- [ ] run-corruption-benchmarks: Run full corruption experiments
- [ ] run-ablations: Run ablation studies
- [ ] analysis-viz: Create analysis notebooks

---

## Executor's Feedback or Assistance Requests

### Current Status (Implementation Complete)
All code implementation is complete. The project is ready for experiments.

### To Run Experiments

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run a quick test**:
   ```bash
   python experiments/train.py --model ghn --dataset cora --seeds 1
   ```

3. **Run full benchmarks**:
   ```bash
   python experiments/run_all_benchmarks.py --seeds 3
   ```

4. **Run ablations**:
   ```bash
   python experiments/run_ablations.py --dataset cora --seeds 3
   ```

### Estimated Experiment Time
- Single model/dataset/corruption: ~2-5 minutes
- Full benchmark (4 models × 3 datasets × 3 corruption types × 3 seeds): ~2-4 hours
- Full ablations (4 params × ~5 values × 3 seeds): ~1-2 hours

---

## Lessons

1. **Laplacian normalization**: Use symmetric normalized Laplacian (I - D^{-1/2} A D^{-1/2}) for numerical stability
2. **Damping is critical**: alpha=0.5 works well; without damping, updates can oscillate
3. **LayerNorm helps**: Apply after each iteration to stabilize training
4. **Beta warmup**: Consider starting with lower beta and increasing (not yet implemented)
5. **PyTorch Geometric**: Use `degree()` and sparse aggregation for efficient Laplacian computation
