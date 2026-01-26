"""Test Graph Hopfield Network with optimal hyperparameters (beta=0.1, lambda=1.0)."""

import argparse
import sys
import yaml
import json
from pathlib import Path
from datetime import datetime

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from train import run_experiment, get_device, save_results


def main():
    parser = argparse.ArgumentParser(description="Test GHN with optimal hyperparameters")
    parser.add_argument(
        "--dataset",
        type=str,
        default="cora",
        help="Dataset name (default: cora)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=3,
        help="Number of seeds (default: 3)",
    )
    parser.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Also run GCN baseline for comparison",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/default.yaml",
        help="Base config file",
    )
    
    args = parser.parse_args()
    
    # Load base config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Set optimal hyperparameters
    config["model"]["name"] = "ghn"
    config["model"]["beta"] = 0.1
    config["model"]["lambda_graph"] = 1.0
    config["dataset"]["name"] = args.dataset
    config["experiment"]["num_seeds"] = args.seeds
    
    device = get_device(config["experiment"]["device"])
    
    print("=" * 70)
    print("TESTING OPTIMAL GHN HYPERPARAMETERS")
    print("=" * 70)
    print(f"Dataset: {args.dataset}")
    print(f"Beta: 0.1 (optimal)")
    print(f"Lambda: 1.0 (optimal)")
    print(f"Seeds: {args.seeds}")
    print(f"Device: {device}")
    print("=" * 70)
    
    # Run GHN with optimal hyperparameters
    print("\n[1/2] Running GHN with optimal hyperparameters...")
    
    from train import run_full_experiment
    ghn_results = run_full_experiment(config, verbose=True)
    
    ghn_mean_acc = ghn_results["test_acc_mean"] * 100
    ghn_std_acc = ghn_results["test_acc_std"] * 100
    ghn_mean_aurc = ghn_results["aurc_mean"] * 100
    ghn_std_aurc = ghn_results["aurc_std"] * 100
    
    print(f"\n✓ GHN (beta=0.1, lambda=1.0) Results:")
    print(f"  Test Accuracy: {ghn_mean_acc:.2f} ± {ghn_std_acc:.2f}%")
    print(f"  AURC: {ghn_mean_aurc:.2f} ± {ghn_std_aurc:.2f}%")
    
    # Compare with GCN baseline if requested
    if args.compare_baseline:
        print("\n[2/2] Running GCN baseline for comparison...")
        
        config_baseline = config.copy()
        config_baseline["model"] = config["model"].copy()
        config_baseline["model"]["name"] = "gcn"
        
        gcn_results = run_full_experiment(config_baseline, verbose=True)
        
        gcn_mean_acc = gcn_results["test_acc_mean"] * 100
        gcn_std_acc = gcn_results["test_acc_std"] * 100
        gcn_mean_aurc = gcn_results["aurc_mean"] * 100
        gcn_std_aurc = gcn_results["aurc_std"] * 100
        
        print(f"\n✓ GCN Baseline Results:")
        print(f"  Test Accuracy: {gcn_mean_acc:.2f} ± {gcn_std_acc:.2f}%")
        print(f"  AURC: {gcn_mean_aurc:.2f} ± {gcn_std_aurc:.2f}%")
        
        # Comparison
        print("\n" + "=" * 70)
        print("COMPARISON: Optimal GHN vs GCN")
        print("=" * 70)
        print(f"{'Metric':<20} {'GHN (Optimal)':<20} {'GCN':<20} {'Difference':<15}")
        print("-" * 70)
        
        acc_diff = ghn_mean_acc - gcn_mean_acc
        aurc_diff = ghn_mean_aurc - gcn_mean_aurc
        
        print(f"{'Test Accuracy':<20} {ghn_mean_acc:>6.2f}% ± {ghn_std_acc:>5.2f}%  {gcn_mean_acc:>6.2f}% ± {gcn_std_acc:>5.2f}%  {acc_diff:>+6.2f}%")
        print(f"{'AURC':<20} {ghn_mean_aurc:>6.2f}% ± {ghn_std_aurc:>5.2f}%  {gcn_mean_aurc:>6.2f}% ± {gcn_std_aurc:>5.2f}%  {aurc_diff:>+6.2f}%")
        
        if acc_diff > 0:
            print(f"\n✓ GHN outperforms GCN by {acc_diff:.2f}% in accuracy!")
        elif abs(acc_diff) < 1.0:
            print(f"\n≈ GHN and GCN perform similarly (difference: {acc_diff:.2f}%)")
        else:
            print(f"\n✗ GCN outperforms GHN by {abs(acc_diff):.2f}% in accuracy")
        
        if aurc_diff > 0:
            print(f"✓ GHN is more robust (AURC difference: {aurc_diff:.2f}%)")
        elif abs(aurc_diff) < 1.0:
            print(f"≈ Similar robustness (AURC difference: {aurc_diff:.2f}%)")
        else:
            print(f"✗ GCN is more robust (AURC difference: {abs(aurc_diff):.2f}%)")
        
        # Save comparison results
        comparison = {
            "dataset": args.dataset,
            "ghn": {
                "beta": 0.1,
                "lambda": 1.0,
                "test_acc_mean": ghn_mean_acc / 100,
                "test_acc_std": ghn_std_acc / 100,
                "aurc_mean": ghn_mean_aurc / 100,
                "aurc_std": ghn_std_aurc / 100,
            },
            "gcn": {
                "test_acc_mean": gcn_mean_acc / 100,
                "test_acc_std": gcn_std_acc / 100,
                "aurc_mean": gcn_mean_aurc / 100,
                "aurc_std": gcn_std_aurc / 100,
            },
            "difference": {
                "acc_diff": acc_diff / 100,
                "aurc_diff": aurc_diff / 100,
            }
        }
        
        results_dir = Path(config["experiment"]["results_dir"]) / "optimal_comparison"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_file = results_dir / f"ghn_optimal_vs_gcn_{args.dataset}_{timestamp}.json"
        
        with open(comparison_file, "w") as f:
            json.dump(comparison, f, indent=2)
        
        print(f"\nResults saved to: {comparison_file}")
    
    else:
        print("\n" + "=" * 70)
        print("RESULTS SUMMARY")
        print("=" * 70)
        print(f"GHN with optimal hyperparameters (beta=0.1, lambda=1.0):")
        print(f"  Test Accuracy: {ghn_mean_acc:.2f} ± {ghn_std_acc:.2f}%")
        print(f"  AURC: {ghn_mean_aurc:.2f} ± {ghn_std_aurc:.2f}%")
        print(f"\nRun with --compare-baseline to compare with GCN")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
