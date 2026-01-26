# Hyperparameter Sweep Guide

## Overview

The hyperparameter sweep runs 360 configurations on Citeseer with 10 seeds each, designed to complete in ~7 hours (assuming ~70 seconds per run).

## Running the Sweep

```bash
python3 experiments/run_hyperparameter_sweep.py
```

The script will:
1. Generate 360 hyperparameter configurations
2. Save a sweep plan to `results/sweeps/citeseer_sweep_plan_<timestamp>.json`
3. Run each configuration sequentially
4. Save progress every 10 runs
5. Generate final summary

## Hyperparameter Space

### Core Grid Search (320 configs)
- **beta**: [0.1, 0.15, 0.2, 0.25] (4 values)
- **lambda**: [0.05, 0.1, 0.15, 0.2] (4 values)
- **num_iterations**: [1, 2, 3, 4, 5] (5 values)
- **num_patterns**: [32, 64, 96, 128] (4 values)

### Random Sampling (40 additional configs)
- **alpha**: [0.3, 0.5, 0.7]
- **num_layers**: [1, 2, 3, 4]
- **dropout**: [0.3, 0.5, 0.7]
- **lr**: [0.005, 0.01, 0.02]
- **weight_decay**: [0.0005, 0.001, 0.01]

## Result Files

Results are saved with descriptive filenames:
```
ghn_citeseer_b0.20_l0.10_iter2_pat64_a0.5_lay2_dr0.5_lr0.010_wd0.0005_s10_<timestamp>.json
```

Format: `model_dataset_beta_lambda_iter_patterns_alpha_layers_dropout_lr_weightdecay_seeds_timestamp.json`

## Analyzing Results

After the sweep completes, analyze results:

```bash
python3 experiments/analyze_sweep_results.py \
    --sweep-results results/sweeps/citeseer_sweep_results_<timestamp>.json
```

This will:
- Match configurations to result files
- Find top 10 configurations
- Identify best hyperparameters
- Generate summary statistics

## Monitoring Progress

Progress is saved every 10 runs to:
```
results/sweeps/citeseer_sweep_progress_<timestamp>.json
```

Check this file to see:
- Number of completed runs
- Elapsed time
- Estimated remaining time

## Manual CLI Usage

You can also run individual configurations manually:

```bash
python3 experiments/train.py \
    --model ghn \
    --dataset citeseer \
    --seeds 10 \
    --beta 0.2 \
    --lambda 0.1 \
    --num-iterations 3 \
    --num-patterns 64 \
    --alpha 0.5 \
    --num-layers 2 \
    --dropout 0.5 \
    --lr 0.01 \
    --weight-decay 0.0005
```

## Notes

- Each run uses 10 seeds for statistical significance
- Results are automatically saved with descriptive filenames
- The sweep can be interrupted and resumed (though you'd need to modify the script to skip completed configs)
- Estimated total time: ~7 hours (360 runs × 70 seconds)
