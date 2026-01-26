"""Analyze all Citeseer result files to find best hyperparameters.

This script scans all GHN Citeseer result files and finds the best configurations.
No sweep plan file needed - just analyzes whatever results exist.
"""

import sys
import json
import glob
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_result_filename(filename: str) -> Dict[str, Any]:
    """Parse hyperparameters from descriptive filename."""
    # Format: ghn_citeseer_b0.20_l0.10_iter2_pat64_a0.5_lay2_dr0.5_lr0.010_wd0.0005_s10_timestamp.json
    parts = filename.replace(".json", "").split("_")
    
    if len(parts) < 10 or parts[0] != "ghn" or parts[1] != "citeseer":
        return None
    
    try:
        return {
            "beta": float(parts[2][1:]),  # b0.20 -> 0.20
            "lambda": float(parts[3][1:]),  # l0.10 -> 0.10
            "num_iterations": int(parts[4][4:]),  # iter2 -> 2
            "num_patterns": int(parts[5][3:]),  # pat64 -> 64
            "alpha": float(parts[6][1:]),  # a0.5 -> 0.5
            "num_layers": int(parts[7][3:]),  # lay2 -> 2
            "dropout": float(parts[8][2:]),  # dr0.5 -> 0.5
            "lr": float(parts[9][2:]),  # lr0.010 -> 0.010
            "weight_decay": float(parts[10][2:]),  # wd0.0005 -> 0.0005
            "num_seeds": int(parts[11][1:]),  # s10 -> 10
        }
    except (ValueError, IndexError) as e:
        print(f"Warning: Could not parse {filename}: {e}")
        return None


def load_all_citeseer_results(results_dir: str = "results") -> pd.DataFrame:
    """Load all Citeseer result files and extract hyperparameters + metrics."""
    
    results_path = Path(results_dir)
    all_files = list(results_path.glob("ghn_citeseer_*.json"))
    
    print(f"Found {len(all_files)} Citeseer result files")
    
    all_results = []
    
    for filepath in all_files:
        # Parse hyperparameters from filename
        config = parse_result_filename(filepath.name)
        if config is None:
            continue
        
        # Load result data
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
            
            all_results.append({
                **config,
                "test_acc_mean": data.get("test_acc_mean"),
                "test_acc_std": data.get("test_acc_std"),
                "aurc_mean": data.get("aurc_mean"),
                "aurc_std": data.get("aurc_std"),
                "file": str(filepath),
            })
        except Exception as e:
            print(f"Warning: Could not load {filepath}: {e}")
    
    return pd.DataFrame(all_results)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze all Citeseer result files")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory containing result JSON files",
    )
    
    args = parser.parse_args()
    
    # Load all results
    print("Loading all Citeseer results...")
    df = load_all_citeseer_results(args.results_dir)
    
    if len(df) == 0:
        print("No valid Citeseer results found!")
        return
    
    # Filter valid results
    df_valid = df[df["test_acc_mean"].notna()].copy()
    
    print(f"\nValid results: {len(df_valid)}/{len(df)}")
    
    if len(df_valid) == 0:
        print("No valid results with test accuracy!")
        return
    
    # Find best configurations
    print("\n" + "="*80)
    print("TOP 20 CONFIGURATIONS BY TEST ACCURACY")
    print("="*80)
    
    top20 = df_valid.nlargest(20, "test_acc_mean")[
        ["beta", "lambda", "num_iterations", "num_patterns", "alpha", 
         "num_layers", "dropout", "lr", "weight_decay",
         "test_acc_mean", "test_acc_std", "aurc_mean"]
    ]
    
    print(top20.to_string(index=False))
    
    # Best overall
    best = df_valid.loc[df_valid["test_acc_mean"].idxmax()]
    
    print("\n" + "="*80)
    print("BEST CONFIGURATION")
    print("="*80)
    print(f"beta:            {best['beta']:.2f}")
    print(f"lambda:         {best['lambda']:.2f}")
    print(f"num_iterations: {int(best['num_iterations'])}")
    print(f"num_patterns:   {int(best['num_patterns'])}")
    print(f"alpha:          {best['alpha']:.1f}")
    print(f"num_layers:     {int(best['num_layers'])}")
    print(f"dropout:        {best['dropout']:.1f}")
    print(f"lr:             {best['lr']:.3f}")
    print(f"weight_decay:   {best['weight_decay']:.4f}")
    print(f"\nTest Accuracy:  {best['test_acc_mean']*100:.2f}% ± {best['test_acc_std']*100:.2f}%")
    print(f"AURC:           {best['aurc_mean']*100:.2f}% ± {best['aurc_std']*100:.2f}%")
    print(f"\nResult file:    {best['file']}")
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"Total configurations: {len(df)}")
    print(f"Valid configurations: {len(df_valid)}")
    print(f"Mean accuracy:        {df_valid['test_acc_mean'].mean()*100:.2f}%")
    print(f"Std accuracy:         {df_valid['test_acc_mean'].std()*100:.2f}%")
    print(f"Max accuracy:         {df_valid['test_acc_mean'].max()*100:.2f}%")
    print(f"Min accuracy:         {df_valid['test_acc_mean'].min()*100:.2f}%")
    
    # Save analysis
    analysis_file = Path("results/sweeps") / "citeseer_analysis_all.json"
    analysis_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(analysis_file, "w") as f:
        json.dump({
            "best_config": best.to_dict(),
            "top20": top20.to_dict("records"),
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


if __name__ == "__main__":
    main()
