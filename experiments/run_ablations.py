"""Run ablation studies for Graph Hopfield Networks."""

import argparse
import os
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
from itertools import product

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from train import run_full_experiment, get_device


# Ablation configurations
ABLATIONS = {
    "beta": [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
    "lambda_graph": [0.0, 0.01, 0.1, 0.5, 1.0],
    "num_iterations": [1, 2, 3, 5],
    "num_patterns": [16, 32, 64, 128],
}


def run_ablation_study(
    ablation_param: str,
    values: List[Any],
    base_config_path: str = "experiments/configs/default.yaml",
    dataset: str = "cora",
    num_seeds: int = 3,
    output_dir: str = "./results/ablations",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run ablation study for a single parameter.
    
    Returns:
        Dict with ablation results
    """
    # Load base config
    with open(base_config_path, "r") as f:
        base_config = yaml.safe_load(f)
    
    base_config["model"]["name"] = "ghn"
    base_config["dataset"]["name"] = dataset
    base_config["experiment"]["num_seeds"] = num_seeds
    
    results = {
        "parameter": ablation_param,
        "values": values,
        "dataset": dataset,
        "results_by_value": {},
    }
    
    for value in values:
        if verbose:
            print(f"\n{'='*50}")
            print(f"Ablation: {ablation_param} = {value}")
            print(f"{'='*50}")
        
        # Create config with this value
        config = base_config.copy()
        config["model"] = base_config["model"].copy()
        config["model"][ablation_param] = value
        
        try:
            exp_results = run_full_experiment(config, verbose=verbose)
            results["results_by_value"][str(value)] = {
                "test_acc_mean": exp_results["test_acc_mean"],
                "test_acc_std": exp_results["test_acc_std"],
                "aurc_mean": exp_results["aurc_mean"],
                "aurc_std": exp_results["aurc_std"],
            }
        except Exception as e:
            print(f"Error: {e}")
            results["results_by_value"][str(value)] = {"error": str(e)}
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_path / f"ablation_{ablation_param}_{dataset}_{timestamp}.json"
    
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    if verbose:
        print(f"\nResults saved to: {results_file}")
        print_ablation_results(results)
    
    return results


def run_all_ablations(
    base_config_path: str = "experiments/configs/default.yaml",
    dataset: str = "cora",
    num_seeds: int = 3,
    output_dir: str = "./results/ablations",
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run all ablation studies."""
    all_results = {}
    
    for param, values in ABLATIONS.items():
        if verbose:
            print(f"\n{'#'*60}")
            print(f"ABLATION STUDY: {param}")
            print(f"{'#'*60}")
        
        results = run_ablation_study(
            ablation_param=param,
            values=values,
            base_config_path=base_config_path,
            dataset=dataset,
            num_seeds=num_seeds,
            output_dir=output_dir,
            verbose=verbose,
        )
        all_results[param] = results
    
    # Save combined results
    output_path = Path(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    combined_file = output_path / f"all_ablations_{dataset}_{timestamp}.json"
    
    with open(combined_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nAll ablation results saved to: {combined_file}")
    
    return all_results


def print_ablation_results(results: Dict[str, Any]) -> None:
    """Print ablation results."""
    print(f"\n--- Ablation: {results['parameter']} ---")
    print(f"Dataset: {results['dataset']}")
    print("-" * 40)
    
    for value, res in results["results_by_value"].items():
        if "error" in res:
            print(f"  {value}: ERROR - {res['error']}")
        else:
            acc = res["test_acc_mean"] * 100
            acc_std = res["test_acc_std"] * 100
            aurc = res["aurc_mean"] * 100
            print(f"  {value:8s}: Acc={acc:5.2f}±{acc_std:4.2f}%, AURC={aurc:5.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Run GHN ablation studies")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/default.yaml",
        help="Base config file",
    )
    parser.add_argument(
        "--param",
        type=str,
        default=None,
        choices=list(ABLATIONS.keys()) + ["all"],
        help="Parameter to ablate (default: all)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cora",
        help="Dataset for ablation",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=3,
        help="Number of seeds",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./results/ablations",
        help="Output directory",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output",
    )
    
    args = parser.parse_args()
    
    if args.param is None or args.param == "all":
        run_all_ablations(
            base_config_path=args.config,
            dataset=args.dataset,
            num_seeds=args.seeds,
            output_dir=args.output,
            verbose=not args.quiet,
        )
    else:
        run_ablation_study(
            ablation_param=args.param,
            values=ABLATIONS[args.param],
            base_config_path=args.config,
            dataset=args.dataset,
            num_seeds=args.seeds,
            output_dir=args.output,
            verbose=not args.quiet,
        )


if __name__ == "__main__":
    main()
