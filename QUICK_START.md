# Quick Start Guide - Running Benchmarks

## Step 1: Install Dependencies

```bash
cd /Users/alexwa/Documents/GitHub/graph-hopfield-networks
pip3 install -r requirements.txt
```

**Note**: On macOS, use `pip3` instead of `pip`. If you encounter issues with PyTorch Geometric, you may need to install PyTorch first:
```bash
pip3 install torch torchvision torchaudio
pip3 install torch_geometric
```

## Step 2: Verify Installation

Run a quick test to make sure everything works:
```bash
python3 -c "from src.layers.graph_hopfield import GraphHopfieldLayer; print('✓ Import successful')"
```

**Note**: On macOS, if you encounter SSL certificate errors when downloading datasets, use:
```bash
SSL_CERT_FILE=$(python3 -m certifi) python3 experiments/train.py --model ghn --dataset cora --seeds 1
```

## Step 3: Run Benchmarks

### Option A: Quick Test (Single Model, Single Dataset, 1 seed)
```bash
# On macOS, use SSL_CERT_FILE to avoid SSL errors:
SSL_CERT_FILE=$(python3 -m certifi) python3 experiments/train.py --model ghn --dataset cora --seeds 1
```

### Option B: Full Benchmark Suite (All Models, All Datasets, All Corruption Types)
```bash
# This will run: 4 models × 3 datasets × 3 corruption types × 3 seeds = 108 experiments
# Estimated time: 2-4 hours
SSL_CERT_FILE=$(python3 -m certifi) python3 experiments/run_all_benchmarks.py --seeds 3
```

### Option C: Custom Benchmark (Specific Models/Datasets)
```bash
# Only GHN and GCN on Cora and Citeseer
SSL_CERT_FILE=$(python3 -m certifi) python3 experiments/run_all_benchmarks.py \
    --models ghn gcn \
    --datasets cora citeseer \
    --seeds 3
```

### Option D: Single Model with Specific Corruption
```bash
# Test GHN on Cora with feature noise
SSL_CERT_FILE=$(python3 -m certifi) python3 experiments/train.py \
    --model ghn \
    --dataset cora \
    --seeds 3
```

## Step 4: View Results

Results are saved to `./results/benchmark/` as JSON files. You can also see summary tables printed at the end of the run.

## Common Options

```bash
# Reduce output verbosity
SSL_CERT_FILE=$(python3 -m certifi) python3 experiments/run_all_benchmarks.py --seeds 3 --quiet

# Use fewer seeds for faster testing
SSL_CERT_FILE=$(python3 -m certifi) python3 experiments/run_all_benchmarks.py --seeds 1

# Specify custom output directory
SSL_CERT_FILE=$(python3 -m certifi) python3 experiments/run_all_benchmarks.py --output ./my_results --seeds 3
```

**Tip**: To avoid typing `SSL_CERT_FILE=$(python3 -m certifi)` every time, you can export it:
```bash
export SSL_CERT_FILE=$(python3 -m certifi)
python3 experiments/train.py --model ghn --dataset cora --seeds 1
```

## Test Optimal Hyperparameters

After running ablations, test the optimal hyperparameters (beta=0.1, lambda=1.0):

```bash
# Test optimal GHN
SSL_CERT_FILE=$(python3 -m certifi) python3 experiments/test_optimal_ghn.py --dataset cora --seeds 3

# Compare optimal GHN vs GCN
SSL_CERT_FILE=$(python3 -m certifi) python3 experiments/test_optimal_ghn.py \
    --dataset cora --seeds 3 --compare-baseline
```

## Troubleshooting

**Issue**: `ModuleNotFoundError: No module named 'torch_geometric'`
- **Solution**: Run `pip install torch_geometric`

**Issue**: Dataset download fails
- **Solution**: The datasets will auto-download on first use. Make sure you have internet connection.

**Issue**: Out of memory
- **Solution**: Reduce `hidden_dim` in `experiments/configs/default.yaml` or use fewer seeds

**Issue**: Experiments take too long
- **Solution**: Start with `--seeds 1` and fewer datasets/models for quick testing
