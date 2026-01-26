"""Hyperparameter sweep for Citeseer dataset.

Runs ~360 configurations with 10 seeds each, designed to complete in ~7 hours.
Uses focused grid search on key hyperparameters with random sampling for others.
"""

import sys
import subprocess
import itertools
import random
from pathlib import Path
from datetime import datetime
import json
import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def generate_hyperparameter_configs():
    """
    Generate exactly 360 hyperparameter configurations.
    
    Strategy:
    - Full grid on beta (4) * lambda (4) * num_iterations (5) = 80 core combinations
    - For each core, vary num_patterns (4) = 320 configs
    - Add 40 more random samples for exploration
    - Randomly sample secondary params (alpha, num_layers, dropout, lr, weight_decay)
    """
    
    # Core hyperparameters (grid search)
    beta_values = [0.1, 0.15, 0.2, 0.25]
    lambda_values = [0.05, 0.1, 0.15, 0.2]
    num_iterations_values = [1, 2, 3, 4, 5]
    num_patterns_values = [32, 64, 96, 128]
    
    # Secondary hyperparameters (random sampling)
    alpha_values = [0.3, 0.5, 0.7]
    num_layers_values = [1, 2, 3, 4]
    dropout_values = [0.3, 0.5, 0.7]
    lr_values = [0.005, 0.01, 0.02]
    weight_decay_values = [0.0005, 0.001, 0.01]
    
    configs = []
    random.seed(42)  # For reproducibility
    
    # Core grid: beta * lambda * num_iterations = 4 * 4 * 5 = 80
    core_triplets = list(itertools.product(
        beta_values,
        lambda_values,
        num_iterations_values
    ))
    
    # For each core triplet, vary num_patterns (4 values) = 320 configs
    for beta, lambda_val, num_iter in core_triplets:
        for num_pat in num_patterns_values:
            # Sample secondary parameters
            alpha = random.choice(alpha_values)
            num_layers = random.choice(num_layers_values)
            dropout = random.choice(dropout_values)
            lr = random.choice(lr_values)
            weight_decay = random.choice(weight_decay_values)
            
            configs.append({
                "beta": beta,
                "lambda": lambda_val,
                "num_iterations": num_iter,
                "num_patterns": num_pat,
                "alpha": alpha,
                "num_layers": num_layers,
                "dropout": dropout,
                "lr": lr,
                "weight_decay": weight_decay,
            })
    
    # Add 40 more random samples for exploration (total = 360)
    for _ in range(40):
        beta = random.choice(beta_values)
        lambda_val = random.choice(lambda_values)
        num_iter = random.choice(num_iterations_values)
        num_pat = random.choice(num_patterns_values)
        alpha = random.choice(alpha_values)
        num_layers = random.choice(num_layers_values)
        dropout = random.choice(dropout_values)
        lr = random.choice(lr_values)
        weight_decay = random.choice(weight_decay_values)
        
        configs.append({
            "beta": beta,
            "lambda": lambda_val,
            "num_iterations": num_iter,
            "num_patterns": num_pat,
            "alpha": alpha,
            "num_layers": num_layers,
            "dropout": dropout,
            "lr": lr,
            "weight_decay": weight_decay,
        })
    
    # Shuffle for better distribution over time
    random.shuffle(configs)
    
    assert len(configs) == 360, f"Expected 360 configs, got {len(configs)}"
    return configs


def run_single_config(config: dict, config_idx: int, total_configs: int, base_config_path: str):
    """Run a single hyperparameter configuration."""
    
    print(f"\n{'='*80}")
    print(f"Configuration {config_idx + 1}/{total_configs}")
    print(f"{'='*80}")
    print(f"beta={config['beta']:.2f}, lambda={config['lambda']:.2f}, "
          f"iter={config['num_iterations']}, patterns={config['num_patterns']}, "
          f"alpha={config['alpha']:.1f}, layers={config['num_layers']}, "
          f"dropout={config['dropout']:.1f}, lr={config['lr']:.3f}, "
          f"wd={config['weight_decay']:.4f}")
    print(f"{'='*80}\n")
    
    # Build command
    cmd = [
        "python3", "experiments/train.py",
        "--config", base_config_path,
        "--model", "ghn",
        "--dataset", "citeseer",
        "--seeds", "10",
        "--beta", str(config['beta']),
        "--lambda", str(config['lambda']),
        "--num-iterations", str(config['num_iterations']),
        "--num-patterns", str(config['num_patterns']),
        "--alpha", str(config['alpha']),
        "--num-layers", str(config['num_layers']),
        "--dropout", str(config['dropout']),
        "--lr", str(config['lr']),
        "--weight-decay", str(config['weight_decay']),
        "--quiet",  # Reduce verbosity for sweep
    ]
    
    # Run command
    try:
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent.parent,
            capture_output=False,  # Show output
            text=True,
            check=True,
        )
        return {"status": "success", "config": config}
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Configuration {config_idx + 1} failed!")
        return {"status": "failed", "config": config, "error": str(e)}


def main():
    """Run hyperparameter sweep."""
    
    base_config_path = "experiments/configs/default.yaml"
    
    # Generate configurations
    print("Generating hyperparameter configurations...")
    configs = generate_hyperparameter_configs()
    print(f"Generated {len(configs)} configurations")
    
    # Save sweep plan
    sweep_dir = Path("results/sweeps")
    sweep_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_plan_file = sweep_dir / f"citeseer_sweep_plan_{timestamp}.json"
    
    with open(sweep_plan_file, "w") as f:
        json.dump({
            "dataset": "citeseer",
            "num_configs": len(configs),
            "num_seeds": 10,
            "estimated_time_per_run_seconds": 70,
            "estimated_total_hours": len(configs) * 70 / 3600,
            "configurations": configs,
        }, f, indent=2)
    
    print(f"Sweep plan saved to: {sweep_plan_file}")
    
    # Run configurations
    results = []
    start_time = datetime.now()
    
    for idx, config in enumerate(configs):
        result = run_single_config(config, idx, len(configs), base_config_path)
        results.append(result)
        
        # Save progress periodically
        if (idx + 1) % 10 == 0:
            progress_file = sweep_dir / f"citeseer_sweep_progress_{timestamp}.json"
            elapsed = (datetime.now() - start_time).total_seconds()
            avg_time = elapsed / (idx + 1)
            remaining = avg_time * (len(configs) - idx - 1)
            
            with open(progress_file, "w") as f:
                json.dump({
                    "completed": idx + 1,
                    "total": len(configs),
                    "elapsed_seconds": elapsed,
                    "avg_time_per_run_seconds": avg_time,
                    "estimated_remaining_seconds": remaining,
                    "results": results,
                }, f, indent=2)
            
            print(f"\nProgress: {idx + 1}/{len(configs)} completed")
            print(f"Elapsed: {elapsed/3600:.2f} hours")
            print(f"Estimated remaining: {remaining/3600:.2f} hours")
    
    # Save final results
    final_results_file = sweep_dir / f"citeseer_sweep_results_{timestamp}.json"
    elapsed_total = (datetime.now() - start_time).total_seconds()
    
    with open(final_results_file, "w") as f:
        json.dump({
            "dataset": "citeseer",
            "total_configs": len(configs),
            "num_seeds": 10,
            "total_time_seconds": elapsed_total,
            "total_time_hours": elapsed_total / 3600,
            "results": results,
        }, f, indent=2)
    
    print(f"\n{'='*80}")
    print("SWEEP COMPLETE")
    print(f"{'='*80}")
    print(f"Total configurations: {len(configs)}")
    print(f"Total time: {elapsed_total/3600:.2f} hours")
    print(f"Results saved to: {final_results_file}")
    
    # Summary
    successful = sum(1 for r in results if r["status"] == "success")
    failed = len(results) - successful
    print(f"\nSuccessful: {successful}")
    print(f"Failed: {failed}")


if __name__ == "__main__":
    main()
