# Graph Hopfield Networks

**Associative Memory for Graph-Structured Data**

This repository implements Graph Hopfield Networks (GHN), which combine modern continuous Hopfield networks with graph neural networks to improve robustness under structured corruption.

## Defensible Claim

> "Adding associative retrieval (Hopfield/attention-style) as an explicit denoising memory operator improves robustness of graph prediction under structured corruption, beyond standard message-passing baselines at matched capacity."

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/graph-hopfield-networks.git
cd graph-hopfield-networks

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.10+
- PyTorch 2.0+
- PyTorch Geometric 2.4+

## Quick Start

### Train a single model

```bash
# Train GHN on Cora
python experiments/train.py --model ghn --dataset cora

# Train baseline GCN
python experiments/train.py --model gcn --dataset cora
```

### Run all benchmarks

```bash
# Run full benchmark suite (3 datasets × 4 models × 3 corruption types)
python experiments/run_all_benchmarks.py --seeds 3
```

### Run ablation studies

```bash
# Run all ablations on Cora
python experiments/run_ablations.py --dataset cora --seeds 3

# Run specific ablation (e.g., beta parameter)
python experiments/run_ablations.py --param beta --dataset cora
```

## Project Structure

```
graph-hopfield-networks/
├── src/
│   ├── layers/
│   │   ├── graph_hopfield.py    # Core GHN layer implementation
│   │   └── memory_bank.py       # Learnable memory patterns
│   ├── models/
│   │   ├── ghn.py               # Graph Hopfield Network model
│   │   └── baselines.py         # GCN, GAT, GraphSAGE baselines
│   ├── data/
│   │   ├── datasets.py          # Dataset loaders
│   │   └── corruption.py        # Corruption utilities
│   └── utils/
│       └── metrics.py           # Evaluation metrics
├── experiments/
│   ├── train.py                 # Main training script
│   ├── run_all_benchmarks.py    # Benchmark runner
│   ├── run_ablations.py         # Ablation studies
│   └── configs/                 # YAML configurations
├── tests/                       # Unit tests
└── requirements.txt
```

## Core Concepts

### Graph Hopfield Energy Function

The GHN energy function combines Hopfield retrieval with graph Laplacian regularization:

```
E_GH(X) = Σᵥ [-β⁻¹ lse(β, MᵀXᵥ) + ½||Xᵥ||²] + λ Σ_(u,v)∈E ||Xᵤ - Xᵥ||²
```

Where:
- First term: Hopfield retrieval energy (attracts to stored patterns)
- Second term: Graph smoothness (enforces neighbor similarity)
- `β`: Inverse temperature (controls retrieval sharpness)
- `λ`: Graph coupling weight

### Key Hyperparameters

| Parameter | Description | Default | Recommended Range |
|-----------|-------------|---------|-------------------|
| `num_patterns` | Memory bank size (K) | 64 | 16-128 |
| `beta` | Inverse temperature | 1.0 | 0.5-10.0 |
| `lambda_graph` | Graph coupling weight | 0.1 | 0.0-1.0 |
| `num_iterations` | Hopfield iterations | 2 | 1-5 |
| `alpha` | Update damping | 0.5 | 0.3-0.7 |

## Corruption Benchmarks

We evaluate robustness under three types of corruption:

1. **Feature Noise**: Gaussian perturbations to node features
2. **Feature Masking**: Random zeroing of features
3. **Edge Corruption**: Random edge dropping/addition

Each corruption is swept from 0% to 50% to generate robustness curves.

## Datasets

- **Cora**: 2,708 nodes, 5,429 edges, 7 classes
- **Citeseer**: 3,327 nodes, 4,732 edges, 6 classes
- **Pubmed**: 19,717 nodes, 44,338 edges, 3 classes

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_layers.py -v
```

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{ghn2024,
  title={Graph Hopfield Networks: Associative Memory for Graph-Structured Data},
  author={Your Name},
  year={2024},
  note={Workshop on Associative Memory}
}
```

## References

- [Universal Hopfield Networks](https://arxiv.org/abs/2202.04557)
- [Dense Associative Memory for Pattern Recognition](https://arxiv.org/abs/1606.01164)
- [Hopfield Networks is All You Need](https://arxiv.org/abs/2008.02217)

## License

MIT License
