"""Run all benchmark experiments comparing GHN to baselines."""

import argparse
import os
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from train import run_full_experiment, get_device


# Experiment configurations
DATASETS = ["cora", "citeseer", "pubmed"]
MODELS = ["ghn", "gcn", "gat", "graphsage"]
CORRUPTION_TYPES = ["feature_noise", "feature_mask", "edge_drop"]


def create_config(
    model: str,
    dataset: str,
    corruption_type: str,
    base_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Create experiment config from base config."""
    config = base_config.copy()
    config["model"] = base_config["model"].copy()
    config["dataset"] = base_config["dataset"].copy()
    config["corruption"] = base_config["corruption"].copy()
    config["experiment"] = base_config["experiment"].copy()
    
    config["model"]["name"] = model
    config["dataset"]["name"] = dataset
    config["corruption"]["type"] = corruption_type
    
    return config


def run_all_benchmarks(
    base_config_path: str = "experiments/configs/default.yaml",
    num_seeds: int = 3,
    datasets: List[str] = None,
    models: List[str] = None,
    corruption_types: List[str] = None,
    output_dir: str = "./results/benchmark",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run all benchmark experiments.
    
    Returns:
        Dict with all results
    """
    # Load base config
    with open(base_config_path, "r") as f:
        base_config = yaml.safe_load(f)
    
    base_config["experiment"]["num_seeds"] = num_seeds
    
    # Set defaults
    datasets = datasets or DATASETS
    models = models or MODELS
    corruption_types = corruption_types or CORRUPTION_TYPES
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    total_experiments = len(datasets) * len(models) * len(corruption_types)
    current = 0
    
    for dataset in datasets:
        all_results[dataset] = {}
        
        for corruption_type in corruption_types:
            all_results[dataset][corruption_type] = {}
            
            for model in models:
                current += 1
                
                if verbose:
                    print(f"\n{'='*60}")
                    print(f"Experiment {current}/{total_experiments}")
                    print(f"Dataset: {dataset}, Model: {model}, Corruption: {corruption_type}")
                    print(f"{'='*60}")
                
                config = create_config(model, dataset, corruption_type, base_config)
                
                try:
                    results = run_full_experiment(config, verbose=verbose)
                    all_results[dataset][corruption_type][model] = {
                        "test_acc_mean": results["test_acc_mean"],
                        "test_acc_std": results["test_acc_std"],
                        "aurc_mean": results["aurc_mean"],
                        "aurc_std": results["aurc_std"],
                        "corruption_levels": results["all_results"][0]["corruption_results"]["levels"],
                        "accuracies_by_seed": [
                            r["corruption_results"]["accuracies"]
                            for r in results["all_results"]
                        ],
                    }
                except Exception as e:
                    print(f"Error running experiment: {e}")
                    all_results[dataset][corruption_type][model] = {"error": str(e)}
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_path / f"benchmark_results_{timestamp}.json"
    
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    # Print summary table
    print_summary_table(all_results)
    
    return all_results


def print_summary_table(results: Dict[str, Any]) -> None:
    """Print a summary table of results."""
    print("\n" + "="*80)
    print("SUMMARY TABLE: Test Accuracy (mean ± std)")
    print("="*80)
    
    for dataset in results:
        print(f"\n--- {dataset.upper()} ---")
        
        # Get corruption types
        corruption_types = list(results[dataset].keys())
        
        for corruption_type in corruption_types:
            print(f"\nCorruption: {corruption_type}")
            print("-" * 40)
            
            models_results = results[dataset][corruption_type]
            
            for model, model_results in models_results.items():
                if "error" in model_results:
                    print(f"  {model}: ERROR - {model_results['error']}")
                else:
                    acc_mean = model_results["test_acc_mean"] * 100
                    acc_std = model_results.get("test_acc_std", 0.0) * 100
                    aurc_mean = model_results["aurc_mean"] * 100
                    aurc_std = model_results.get("aurc_std", 0.0) * 100
                    
                    # Handle NaN or single seed case
                    if acc_std > 0:
                        print(f"  {model:12s}: Acc={acc_mean:5.2f}±{acc_std:4.2f}%, AURC={aurc_mean:5.2f}±{aurc_std:4.2f}%")
                    else:
                        print(f"  {model:12s}: Acc={acc_mean:5.2f}%, AURC={aurc_mean:5.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Run all GHN benchmark experiments")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/default.yaml",
        help="Base config file",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=3,
        help="Number of seeds per experiment",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=None,
        help="Datasets to run (default: cora citeseer pubmed)",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="Models to run (default: ghn gcn gat graphsage)",
    )
    parser.add_argument(
        "--corruption-types",
        type=str,
        nargs="+",
        default=None,
        help="Corruption types (default: feature_noise feature_mask edge_drop)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./results/benchmark",
        help="Output directory",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output",
    )
    
    args = parser.parse_args()
    
    run_all_benchmarks(
        base_config_path=args.config,
        num_seeds=args.seeds,
        datasets=args.datasets,
        models=args.models,
        corruption_types=args.corruption_types,
        output_dir=args.output,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
