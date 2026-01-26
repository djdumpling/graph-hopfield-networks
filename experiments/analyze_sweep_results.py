"""Analyze hyperparameter sweep results to find optimal configuration."""

import sys
import json
import glob
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_sweep_results(sweep_results_file: str) -> Dict[str, Any]:
    """Load sweep results from JSON file."""
    with open(sweep_results_file, "r") as f:
        return json.load(f)


def find_result_files(config: Dict[str, Any], results_dir: str = "results") -> List[str]:
    """
    Find result files matching a configuration.
    
    Uses the descriptive filename format from save_results().
    More flexible matching to handle floating point precision issues.
    """
    beta = config["beta"]
    lambda_val = config["lambda"]
    num_iter = config["num_iterations"]
    num_pat = config["num_patterns"]
    alpha = config["alpha"]
    num_layers = config["num_layers"]
    dropout = config["dropout"]
    lr = config["lr"]
    weight_decay = config["weight_decay"]
    
    # Search all result files and match by parsing filenames
    results_path = Path(results_dir)
    all_files = list(results_path.glob("ghn_citeseer_*.json"))
    
    matches = []
    for filepath in all_files:
        filename = filepath.stem  # Without .json extension
        
        # Parse filename: ghn_citeseer_b0.20_l0.10_iter2_pat64_a0.5_lay2_dr0.5_lr0.010_wd0.0005_s10_timestamp
        try:
            parts = filename.split("_")
            if len(parts) < 10:
                continue
            
            # Extract values from filename
            file_beta = float(parts[2][1:])  # b0.20 -> 0.20
            file_lambda = float(parts[3][1:])  # l0.10 -> 0.10
            file_iter = int(parts[4][4:])  # iter2 -> 2
            file_pat = int(parts[5][3:])  # pat64 -> 64
            file_alpha = float(parts[6][1:])  # a0.5 -> 0.5
            file_layers = int(parts[7][3:])  # lay2 -> 2
            file_dropout = float(parts[8][2:])  # dr0.5 -> 0.5
            file_lr = float(parts[9][2:])  # lr0.010 -> 0.010
            file_wd = float(parts[10][2:])  # wd0.0005 -> 0.0005
            
            # Match with tolerance for floating point
            if (abs(file_beta - beta) < 0.01 and
                abs(file_lambda - lambda_val) < 0.01 and
                file_iter == num_iter and
                file_pat == num_pat and
                abs(file_alpha - alpha) < 0.05 and
                file_layers == num_layers and
                abs(file_dropout - dropout) < 0.05 and
                abs(file_lr - lr) < 0.001 and
                abs(file_wd - weight_decay) < 0.0001):
                matches.append(str(filepath))
        except (ValueError, IndexError):
            continue
    
    return matches


def extract_results_from_files(result_files: List[str]) -> List[Dict[str, Any]]:
    """Extract test accuracy and AURC from result files."""
    results = []
    
    for filepath in result_files:
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
            
            results.append({
                "test_acc_mean": data.get("test_acc_mean", 0),
                "test_acc_std": data.get("test_acc_std", 0),
                "aurc_mean": data.get("aurc_mean", 0),
                "aurc_std": data.get("aurc_std", 0),
                "file": filepath,
            })
        except Exception as e:
            print(f"Warning: Could not load {filepath}: {e}")
    
    return results


def analyze_sweep(sweep_results_file: str, results_dir: str = "results"):
    """Analyze sweep results and find best configurations."""
    
    print("Loading sweep results...")
    sweep_data = load_sweep_results(sweep_results_file)
    
    # Handle both sweep_results format and sweep_plan format
    if "configurations" in sweep_data:
        # This is a sweep plan file
        configs = sweep_data["configurations"]
        results_list = None
    elif "results" in sweep_data:
        # This is a sweep results file - extract configs from results
        results_list = sweep_data["results"]
        configs = [r["config"] for r in results_list if r.get("config")]
    else:
        raise ValueError("Unknown file format - expected 'configurations' or 'results' key")
    
    print(f"Analyzing {len(configs)} configurations...")
    print("Loading result files (this may take a while)...")
    
    # Match configurations to result files
    all_results = []
    
    for idx, config in enumerate(configs):
        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(configs)} configurations...")
        
        result_files = find_result_files(config, results_dir)
        
        if result_files:
            file_results = extract_results_from_files(result_files)
            for fr in file_results:
                all_results.append({
                    **config,
                    **fr,
                })
        else:
            # Configuration not run or failed
            # Check if we have status info from sweep results
            status = "unknown"
            if results_list and idx < len(results_list):
                status = results_list[idx].get("status", "unknown")
            
            all_results.append({
                **config,
                "test_acc_mean": None,
                "test_acc_std": None,
                "aurc_mean": None,
                "aurc_std": None,
                "file": None,
                "status": status,
            })
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(all_results)
    
    # Filter out failed runs
    df_valid = df[df["test_acc_mean"].notna()].copy()
    
    print(f"\nValid results: {len(df_valid)}/{len(df)}")
    
    if len(df_valid) == 0:
        print("No valid results found!")
        return
    
    # Find best configurations
    print("\n" + "="*80)
    print("TOP 10 CONFIGURATIONS BY TEST ACCURACY")
    print("="*80)
    
    top10 = df_valid.nlargest(10, "test_acc_mean")[
        ["beta", "lambda", "num_iterations", "num_patterns", "alpha", 
         "num_layers", "dropout", "lr", "weight_decay",
         "test_acc_mean", "test_acc_std", "aurc_mean"]
    ]
    
    print(top10.to_string(index=False))
    
    # Best overall
    best = df_valid.loc[df_valid["test_acc_mean"].idxmax()]
    
    print("\n" + "="*80)
    print("BEST CONFIGURATION")
    print("="*80)
    print(f"beta:            {best['beta']:.2f}")
    print(f"lambda:         {best['lambda']:.2f}")
    print(f"num_iterations: {best['num_iterations']}")
    print(f"num_patterns:   {best['num_patterns']}")
    print(f"alpha:          {best['alpha']:.1f}")
    print(f"num_layers:     {best['num_layers']}")
    print(f"dropout:        {best['dropout']:.1f}")
    print(f"lr:             {best['lr']:.3f}")
    print(f"weight_decay:   {best['weight_decay']:.4f}")
    print(f"\nTest Accuracy:  {best['test_acc_mean']*100:.2f}% ± {best['test_acc_std']*100:.2f}%")
    print(f"AURC:           {best['aurc_mean']*100:.2f}% ± {best['aurc_std']*100:.2f}%")
    print(f"\nResult file:    {best['file']}")
    
    # Save analysis
    analysis_file = Path("results/sweeps") / f"analysis_{Path(sweep_results_file).stem}.json"
    analysis_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(analysis_file, "w") as f:
        json.dump({
            "best_config": best.to_dict(),
            "top10": top10.to_dict("records"),
            "summary_stats": {
                "total_configs": len(df),
                "valid_configs": len(df_valid),
                "mean_accuracy": float(df_valid["test_acc_mean"].mean()),
                "std_accuracy": float(df_valid["test_acc_mean"].std()),
                "max_accuracy": float(df_valid["test_acc_mean"].max()),
                "min_accuracy": float(df_valid["test_acc_mean"].min()),
            }
        }, f, indent=2, default=str)
    
    print(f"\nAnalysis saved to: {analysis_file}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze hyperparameter sweep results")
    parser.add_argument(
        "--sweep-results",
        type=str,
        required=True,
        help="Path to sweep results JSON file",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory containing result JSON files",
    )
    
    args = parser.parse_args()
    
    analyze_sweep(args.sweep_results, args.results_dir)


if __name__ == "__main__":
    main()
