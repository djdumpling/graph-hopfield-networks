"""Compare all models (GHN, GCN, GAT, GraphSAGE) on Citeseer."""

import sys
import json
from pathlib import Path
from typing import Dict, List
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_model_results(results_dir: str = "results") -> Dict[str, List[Dict]]:
    """Load results for all models on Citeseer."""
    
    results_path = Path(results_dir)
    # GHN Citeseer results live in results/ghn_citeseer/
    ghn_files = list((results_path / "ghn_citeseer").glob("ghn_citeseer_*.json"))
    other_files = list(results_path.glob("*_citeseer_*.json"))  # gcn, gat, graphsage
    all_files = ghn_files + other_files
    
    model_results = {
        "ghn": [],
        "gcn": [],
        "gat": [],
        "graphsage": [],
    }
    
    for filepath in all_files:
        filename = filepath.name
        
        # Determine model type
        if filename.startswith("ghn_citeseer"):
            model = "ghn"
        elif filename.startswith("gcn_citeseer"):
            model = "gcn"
        elif filename.startswith("gat_citeseer"):
            model = "gat"
        elif filename.startswith("graphsage_citeseer"):
            model = "graphsage"
        else:
            continue
        
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
            
            # Extract key metrics
            result = {
                "file": str(filepath),
                "test_acc_mean": data.get("test_acc_mean"),
                "test_acc_std": data.get("test_acc_std"),
                "aurc_mean": data.get("aurc_mean"),
                "aurc_std": data.get("aurc_std"),
                "num_seeds": data.get("num_seeds", 1),
            }
            
            # For GHN, also extract hyperparameters from filename
            if model == "ghn":
                parts = filename.replace(".json", "").split("_")
                if len(parts) >= 10:
                    try:
                        result["beta"] = float(parts[2][1:])
                        result["lambda"] = float(parts[3][1:])
                        result["num_iterations"] = int(parts[4][4:])
                        result["num_patterns"] = int(parts[5][3:])
                    except:
                        pass
            
            model_results[model].append(result)
        except Exception as e:
            print(f"Warning: Could not load {filepath}: {e}")
    
    return model_results


def main():
    print("="*80)
    print("CITESEER MODEL COMPARISON")
    print("="*80)
    
    # Load all results
    print("\nLoading results...")
    all_results = load_model_results()
    
    # Aggregate by model
    print("\n" + "="*80)
    print("MODEL COMPARISON SUMMARY")
    print("="*80)
    
    comparison = []
    
    for model_name, results in all_results.items():
        if not results:
            print(f"\n{model_name.upper()}: No results found")
            continue
        
        # Filter valid results
        valid = [r for r in results if r["test_acc_mean"] is not None]
        
        if not valid:
            print(f"\n{model_name.upper()}: No valid results")
            continue
        
        # Calculate statistics
        accs = [r["test_acc_mean"] for r in valid]
        aurcs = [r["aurc_mean"] for r in valid if r["aurc_mean"] is not None]
        
        # Find best result
        best = max(valid, key=lambda x: x["test_acc_mean"] if x["test_acc_mean"] else 0)
        
        comparison.append({
            "model": model_name.upper(),
            "num_runs": len(valid),
            "best_acc": best["test_acc_mean"] * 100,
            "best_acc_std": best["test_acc_std"] * 100 if best["test_acc_std"] else 0,
            "mean_acc": sum(accs) / len(accs) * 100,
            "best_aurc": best["aurc_mean"] * 100 if best["aurc_mean"] else None,
            "best_file": best["file"],
        })
        
        print(f"\n{model_name.upper()}:")
        print(f"  Number of runs: {len(valid)}")
        print(f"  Best accuracy:  {best['test_acc_mean']*100:.2f}% ± {best['test_acc_std']*100:.2f}%")
        if best["aurc_mean"]:
            print(f"  Best AURC:      {best['aurc_mean']*100:.2f}% ± {best['aurc_std']*100:.2f}%")
        print(f"  Mean accuracy:  {sum(accs)/len(accs)*100:.2f}%")
        
        # For GHN, show best hyperparameters
        if model_name == "ghn" and "beta" in best:
            print(f"  Best config:")
            print(f"    beta={best.get('beta', 'N/A'):.2f}, "
                  f"lambda={best.get('lambda', 'N/A'):.2f}, "
                  f"iter={best.get('num_iterations', 'N/A')}, "
                  f"patterns={best.get('num_patterns', 'N/A')}")
    
    # Create comparison table
    print("\n" + "="*80)
    print("COMPARISON TABLE")
    print("="*80)
    
    df = pd.DataFrame(comparison)
    if len(df) > 0:
        # Sort by best accuracy
        df = df.sort_values("best_acc", ascending=False)
        
        print(f"\n{'Model':<12} {'Runs':<8} {'Best Acc':<15} {'Mean Acc':<12} {'Best AURC':<12}")
        print("-" * 80)
        for _, row in df.iterrows():
            aurc_str = f"{row['best_aurc']:.2f}%" if row['best_aurc'] else "N/A"
            print(f"{row['model']:<12} {row['num_runs']:<8} "
                  f"{row['best_acc']:.2f}% ± {row['best_acc_std']:.2f}%  "
                  f"{row['mean_acc']:.2f}%  {aurc_str}")
        
        # Find best overall
        best_overall = df.iloc[0]
        print(f"\n{'='*80}")
        print(f"BEST MODEL: {best_overall['model']}")
        print(f"{'='*80}")
        print(f"Test Accuracy: {best_overall['best_acc']:.2f}% ± {best_overall['best_acc_std']:.2f}%")
        if best_overall['best_aurc']:
            print(f"AURC:          {best_overall['best_aurc']:.2f}%")
        print(f"\nResult file: {best_overall['best_file']}")
        
        # Compare GHN to best baseline
        baselines = df[df['model'] != 'GHN']
        if len(baselines) > 0:
            best_baseline = baselines.iloc[0]
            ghn_row = df[df['model'] == 'GHN']
            
            if len(ghn_row) > 0:
                ghn_acc = ghn_row.iloc[0]['best_acc']
                baseline_acc = best_baseline['best_acc']
                gap = ghn_acc - baseline_acc
                
                print(f"\n{'='*80}")
                print("GHN vs BEST BASELINE")
                print(f"{'='*80}")
                print(f"GHN (best):     {ghn_acc:.2f}%")
                print(f"{best_baseline['model']} (best): {baseline_acc:.2f}%")
                print(f"Gap:            {gap:+.2f}%")
                
                if gap > 0:
                    print(f"\n✓ GHN outperforms {best_baseline['model']} by {gap:.2f}%")
                elif gap < -1:
                    print(f"\n✗ GHN underperforms {best_baseline['model']} by {abs(gap):.2f}%")
                else:
                    print(f"\n≈ GHN is competitive with {best_baseline['model']} (within 1%)")
    
    # Save comparison
    output_file = Path("results/sweeps/citeseer_model_comparison.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w") as f:
        json.dump({
            "comparison": df.to_dict("records") if len(df) > 0 else [],
            "all_results": all_results,
        }, f, indent=2, default=str)
    
    print(f"\nComparison saved to: {output_file}")


if __name__ == "__main__":
    main()
