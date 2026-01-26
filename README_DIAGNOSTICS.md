# GHN Diagnostic Tools

This document describes the diagnostic tools for analyzing Graph Hopfield Networks.

## Energy Logging

Energy logging tracks whether the energy function decreases across iterations, as guaranteed by the mathematical theory.

### Usage

Enable energy logging in config:
```yaml
experiment:
  log_energy: true
```

Or via command line:
```bash
python3 experiments/diagnose_ghn.py --dataset cora --seed 42
```

### What it tracks:
- Energy before and after each iteration
- Energy change per iteration
- Whether energy decreased (should be true for valid energy descent)

## Attention Analysis

Attention analysis examines how memory patterns are being used.

### Metrics:
- **Attention Entropy**: Measures how focused/uniform the attention distribution is
  - Low entropy = focused retrieval (one pattern dominates)
  - High entropy = uniform retrieval (all patterns used equally)
- **Pattern Usage**: Which patterns are most/least used
- **Pattern Collapse**: Whether memory patterns have become too similar

### Visualizations:
- Attention heatmap: Shows attention weights across nodes and patterns
- Attention entropy distribution: Histogram of entropy values
- Pattern similarity matrix: Cosine similarity between memory patterns

## Pattern Collapse Detection

Pattern collapse occurs when memory patterns become too similar, reducing the diversity of retrievable information.

### Detection metrics:
- **Max Similarity**: Maximum cosine similarity between any two patterns
- **Mean Similarity**: Average pairwise similarity
- **Pattern Diversity**: 1 - mean_similarity (higher is better)
- **Active Patterns**: Number of patterns with usage above uniform baseline

### Thresholds:
- Similarity > 0.95: Patterns are collapsed
- Similarity > 0.90: Patterns are highly similar (warning)

## Running Diagnostics

### Quick diagnostic:
```bash
python3 experiments/diagnose_ghn.py --dataset cora --seed 42
```

### Full diagnostic with custom config:
```bash
python3 experiments/diagnose_ghn.py \
    --config experiments/configs/default.yaml \
    --dataset citeseer \
    --seed 42 \
    --output-dir results/my_diagnostics
```

### Output:
The diagnostic script creates:
- `energy_evolution.png`: Energy over training
- `energy_changes.png`: Energy change per iteration
- `pattern_similarity_layer_X.png`: Similarity matrix for each layer
- `attention_heatmap_layer_X.png`: Attention weights visualization
- `attention_entropy_layer_X.png`: Entropy distribution
- `diagnostic_results.json`: Complete analysis results

## Interpreting Results

### Energy Descent:
- **Descent rate > 0.8**: Energy decreases reliably (good)
- **Descent rate < 0.5**: Energy increases often (problematic)
- **Energy trend decreasing**: Overall energy decreases (good)

### Pattern Collapse:
- **Max similarity < 0.7**: Patterns are diverse (good)
- **Max similarity > 0.95**: Patterns collapsed (bad)
- **Active patterns < K/2**: Only few patterns used (may indicate collapse)

### Attention Entropy:
- **Mean entropy close to max**: Uniform attention (low beta, all patterns used equally)
- **Mean entropy low**: Focused attention (high beta, selective retrieval)
- **For beta=0.1**: Expect entropy close to max (uniform)
- **For beta=1.0+**: Expect lower entropy (focused)

## Common Issues

### Energy not decreasing:
- LayerNorm may be breaking energy descent
- Learning rate too high
- Beta too low (softmax too flat)

### Pattern collapse:
- Memory patterns initialized too close together
- Need weight decay on memory keys
- Need diversity regularization

### Uniform attention:
- Beta too low (e.g., 0.1)
- Memory patterns not learned properly
- Need higher beta for selective retrieval
