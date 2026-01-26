"""Test script to compare GHN with and without LayerNorm.

This tests whether removing LayerNorm (which breaks energy descent guarantees)
improves performance by aligning with the mathematical formulation.
"""

import sys
from pathlib import Path
import yaml
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.train import run_full_experiment, save_results


def main():
    """Run experiments with and without LayerNorm."""
    
    # Load default config
    config_path = Path(__file__).parent / "configs" / "default.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Ensure we're testing GHN
    config["model"]["name"] = "ghn"
    config["experiment"]["num_seeds"] = 3  # Fewer seeds for quick test
    
    print("=" * 70)
    print("Testing Graph Hopfield Network: LayerNorm vs No LayerNorm")
    print("=" * 70)
    
    results = {}
    
    # Test WITH LayerNorm (default)
    print("\n" + "=" * 70)
    print("Experiment 1: WITH LayerNorm (default)")
    print("=" * 70)
    config["model"]["use_layer_norm"] = True
    config["model"]["normalize_memory_keys"] = True
    config["model"]["normalize_memory_queries"] = True
    
    results_with_ln = run_full_experiment(config, verbose=True)
    results["with_layernorm"] = {
        "test_acc_mean": results_with_ln["test_acc_mean"],
        "test_acc_std": results_with_ln["test_acc_std"],
        "aurc_mean": results_with_ln["aurc_mean"],
        "aurc_std": results_with_ln["aurc_std"],
    }
    
    # Test WITHOUT LayerNorm
    print("\n" + "=" * 70)
    print("Experiment 2: WITHOUT LayerNorm (energy descent preserving)")
    print("=" * 70)
    config["model"]["use_layer_norm"] = False
    
    results_without_ln = run_full_experiment(config, verbose=True)
    results["without_layernorm"] = {
        "test_acc_mean": results_without_ln["test_acc_mean"],
        "test_acc_std": results_without_ln["test_acc_std"],
        "aurc_mean": results_without_ln["aurc_mean"],
        "aurc_std": results_without_ln["aurc_std"],
    }
    
    # Print comparison
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"\nWITH LayerNorm:")
    print(f"  Test Accuracy: {results['with_layernorm']['test_acc_mean']*100:.2f} ± {results['with_layernorm']['test_acc_std']*100:.2f}%")
    print(f"  AURC:          {results['with_layernorm']['aurc_mean']*100:.2f} ± {results['with_layernorm']['aurc_std']*100:.2f}%")
    
    print(f"\nWITHOUT LayerNorm:")
    print(f"  Test Accuracy: {results['without_layernorm']['test_acc_mean']*100:.2f} ± {results['without_layernorm']['test_acc_std']*100:.2f}%")
    print(f"  AURC:          {results['without_layernorm']['aurc_mean']*100:.2f} ± {results['without_layernorm']['aurc_std']*100:.2f}%")
    
    diff_acc = results['without_layernorm']['test_acc_mean'] - results['with_layernorm']['test_acc_mean']
    diff_aurc = results['without_layernorm']['aurc_mean'] - results['with_layernorm']['aurc_mean']
    
    print(f"\nDifference (No LN - With LN):")
    print(f"  Test Accuracy: {diff_acc*100:+.2f}%")
    print(f"  AURC:          {diff_aurc*100:+.2f}%")
    
    # Save results
    if config["experiment"]["save_results"]:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = Path(config["experiment"]["results_dir"]) / f"layernorm_comparison_{timestamp}.json"
        results_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_file, "w") as f:
            json.dump({
                "config": config,
                "comparison": results,
                "full_results_with_ln": results_with_ln,
                "full_results_without_ln": results_without_ln,
            }, f, indent=2, default=str)
        
        print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
