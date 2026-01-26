"""Diagnostic script for GHN: energy descent, attention analysis, pattern collapse detection."""

import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from experiments.train import (
    run_experiment, get_device, create_model
)
from src.data.datasets import load_dataset, print_dataset_info
from src.utils.attention_analysis import (
    compute_attention_entropy,
    compute_pattern_usage,
    detect_pattern_collapse,
    visualize_attention_heatmap,
    visualize_pattern_similarity,
    visualize_attention_entropy,
    analyze_attention_evolution,
)


def analyze_energy_descent(energies_history: list, save_dir: Path):
    """Analyze energy descent across training."""
    if not energies_history:
        print("No energy data available")
        return {}
    
    # Extract energy values from history
    # Structure: energies_history[epoch] = [iter_energy_dict, ...] (flat list from all layers)
    all_energies = []
    energy_changes = []
    descent_rates = []
    
    for epoch_idx, epoch_energies in enumerate(energies_history):
        if not epoch_energies:
            continue
        
        # epoch_energies is a list of iteration energy dicts (from all layers combined)
        if isinstance(epoch_energies, list):
            for iter_energy in epoch_energies:
                if isinstance(iter_energy, dict):
                    energy_before = iter_energy.get("energy_before")
                    if energy_before is not None:
                        all_energies.append(energy_before)
                    
                    energy_change = iter_energy.get("energy_change")
                    if energy_change is not None:
                        energy_changes.append(energy_change)
                    
                    if iter_energy.get("energy_decreased", False):
                        descent_rates.append(1.0)
                    elif "energy_change" in iter_energy:
                        descent_rates.append(0.0)
        elif isinstance(epoch_energies, dict):
            # Single energy dict
            energy_before = epoch_energies.get("energy_before")
            if energy_before is not None:
                all_energies.append(energy_before)
            
            energy_change = epoch_energies.get("energy_change")
            if energy_change is not None:
                energy_changes.append(energy_change)
            
            if epoch_energies.get("energy_decreased", False):
                descent_rates.append(1.0)
            elif "energy_change" in epoch_energies:
                descent_rates.append(0.0)
    
    if not all_energies:
        print("No valid energy data found")
        print(f"  Energies history type: {type(energies_history)}")
        print(f"  Energies history length: {len(energies_history)}")
        if len(energies_history) > 0:
            print(f"  First epoch type: {type(energies_history[0])}")
            if isinstance(energies_history[0], list) and len(energies_history[0]) > 0:
                print(f"  First layer type: {type(energies_history[0][0])}")
        return {}
    
    all_energies = np.array(all_energies)
    energy_changes = np.array(energy_changes) if energy_changes else np.array([])
    descent_rates = np.array(descent_rates) if descent_rates else np.array([])
    
    # Compute statistics
    stats = {
        "mean_energy": float(all_energies.mean()),
        "std_energy": float(all_energies.std()),
        "min_energy": float(all_energies.min()),
        "max_energy": float(all_energies.max()),
        "energy_trend": "decreasing" if len(all_energies) > 1 and all_energies[-1] < all_energies[0] else "increasing",
    }
    
    if len(energy_changes) > 0:
        stats["mean_energy_change"] = float(energy_changes.mean())
        stats["descent_rate"] = float(descent_rates.mean()) if len(descent_rates) > 0 else 0.0
        stats["energy_decreases"] = int(descent_rates.sum())
        stats["energy_increases"] = int(len(descent_rates) - descent_rates.sum())
    
    # Visualize energy evolution
    plt.figure(figsize=(12, 6))
    plt.plot(all_energies, alpha=0.7, label="Energy")
    if len(all_energies) > 1:
        # Moving average
        window = min(50, len(all_energies) // 10)
        if window > 1:
            moving_avg = np.convolve(all_energies, np.ones(window)/window, mode="valid")
            plt.plot(range(window-1, len(all_energies)), moving_avg, 
                    label=f"Moving Average (window={window})", linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel("Energy")
    plt.title("Energy Evolution During Training")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "energy_evolution.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    # Visualize energy changes
    if len(energy_changes) > 0:
        plt.figure(figsize=(12, 6))
        plt.plot(energy_changes, alpha=0.7)
        plt.axhline(0, color="red", linestyle="--", label="Zero change")
        plt.xlabel("Iteration")
        plt.ylabel("Energy Change (After - Before)")
        plt.title("Energy Change Per Iteration")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / "energy_changes.png", dpi=150, bbox_inches="tight")
        plt.close()
    
    return stats


def analyze_memory_patterns(model, data, device, save_dir: Path):
    """Analyze memory patterns for collapse."""
    model.eval()
    
    # Get memory banks from all layers
    memory_banks = []
    if hasattr(model, "ghn_layers"):
        for layer in model.ghn_layers:
            if hasattr(layer, "memory"):
                memory_banks.append(layer.memory)
    elif hasattr(model, "ghn") and hasattr(model.ghn, "memory"):
        # For GraphHopfieldNetworkMinimal
        memory_banks.append(model.ghn.memory)
    
    if not memory_banks:
        print("No memory banks found")
        return {}
    
    all_results = {}
    
    # Get attention from a forward pass
    with torch.no_grad():
        data_device = data.to(device)
        # Forward pass to get attention
        if hasattr(model, "encoder"):
            encoded = model.encoder(data_device.x)
        else:
            encoded = data_device.x
        
        # Collect attention from all layers
        all_attentions = []
        if hasattr(model, "ghn_layers"):
            x = encoded
            for layer in model.ghn_layers:
                x, attn_info = layer(x, data_device.edge_index, return_attention=True)
                if attn_info and "attentions" in attn_info and attn_info["attentions"]:
                    all_attentions.append(attn_info["attentions"][-1])  # Last iteration
        elif hasattr(model, "ghn"):
            x, attn_info = model.ghn(encoded, data_device.edge_index, return_attention=True)
            if attn_info and "attentions" in attn_info and attn_info["attentions"]:
                all_attentions.append(attn_info["attentions"][-1])
    
    for layer_idx, memory in enumerate(memory_banks):
        # Get memory keys
        memory_keys = memory.keys.data
        
        # Get attention for this layer if available
        attention = all_attentions[layer_idx] if layer_idx < len(all_attentions) else None
        
        # Detect pattern collapse
        collapse_stats = detect_pattern_collapse(memory_keys, attention)
        
        all_results[f"layer_{layer_idx}"] = {
            "num_patterns": memory.num_patterns,
            "pattern_dim": memory.pattern_dim,
            **collapse_stats,
        }
        
        # Visualize pattern similarity
        visualize_pattern_similarity(
            memory_keys,
            save_path=save_dir / f"pattern_similarity_layer_{layer_idx}.png",
            title=f"Memory Pattern Similarity (Layer {layer_idx})",
        )
        
        # Visualize attention if available
        if attention is not None:
            # Ensure attention is a torch tensor
            if isinstance(attention, np.ndarray):
                attention = torch.from_numpy(attention)
            
            visualize_attention_heatmap(
                attention,
                save_path=save_dir / f"attention_heatmap_layer_{layer_idx}.png",
                title=f"Attention Weights (Layer {layer_idx})",
            )
            
            visualize_attention_entropy(
                attention,
                save_path=save_dir / f"attention_entropy_layer_{layer_idx}.png",
                title=f"Attention Entropy Distribution (Layer {layer_idx})",
            )
            
            # Compute attention statistics
            entropy = compute_attention_entropy(attention)
            usage_stats = compute_pattern_usage(attention)
            
            all_results[f"layer_{layer_idx}"].update({
                "mean_attention_entropy": float(entropy.mean().item()),
                "std_attention_entropy": float(entropy.std().item()),
                "max_entropy": float(torch.log(torch.tensor(attention.shape[-1], dtype=torch.float))),
                **usage_stats,
            })
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Diagnose GHN: energy, attention, pattern collapse")
    parser.add_argument("--config", type=str, default="experiments/configs/default.yaml",
                       help="Config file")
    parser.add_argument("--dataset", type=str, default="cora",
                       help="Dataset name")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--output-dir", type=str, default="results/diagnostics",
                       help="Output directory for diagnostics")
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Override config for diagnostics
    config["model"]["name"] = "ghn"
    config["dataset"]["name"] = args.dataset
    config["experiment"]["num_seeds"] = 1
    config["experiment"]["log_energy"] = True
    config["experiment"]["log_attention"] = True
    config["training"]["epochs"] = 50  # Shorter run for diagnostics
    
    device = get_device(config["experiment"]["device"])
    
    # Create output directory
    output_dir = Path(args.output_dir) / f"{args.dataset}_seed{args.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("GHN DIAGNOSTIC ANALYSIS")
    print("=" * 70)
    print(f"Dataset: {args.dataset}")
    print(f"Seed: {args.seed}")
    print(f"Output directory: {output_dir}")
    print("=" * 70)
    
    # Run experiment with logging
    print("\n[1/3] Running training with energy and attention logging...")
    result = run_experiment(config, args.seed, device, verbose=True)
    
    # Analyze energy descent
    print("\n[2/3] Analyzing energy descent...")
    energy_stats = {}
    if "training_history" in result and "energies" in result["training_history"]:
        energy_stats = analyze_energy_descent(
            result["training_history"]["energies"],
            output_dir
        )
        mean_energy = energy_stats.get('mean_energy', None)
        if mean_energy is not None:
            print(f"  Mean energy: {mean_energy:.4f}")
        else:
            print(f"  Mean energy: N/A")
        descent_rate = energy_stats.get('descent_rate', None)
        if descent_rate is not None:
            print(f"  Descent rate: {descent_rate:.2%}")
        else:
            print(f"  Descent rate: N/A")
        print(f"  Energy trend: {energy_stats.get('energy_trend', 'N/A')}")
    else:
        print("  No energy data found in results")
    
    # Analyze memory patterns
    print("\n[3/3] Analyzing memory patterns...")
    # Load model and data
    data, dataset_info = load_dataset(
        name=config["dataset"]["name"],
        root=config["dataset"]["root"],
        normalize_features=config["dataset"]["normalize_features"],
    )
    
    model = create_model(config, dataset_info["num_features"], dataset_info["num_classes"])
    # Load best model state if available
    if "best_model_state" in result.get("training_history", {}):
        model.load_state_dict(result["training_history"]["best_model_state"])
    model = model.to(device)
    
    pattern_stats = analyze_memory_patterns(model, data, device, output_dir)
    
    # Print pattern collapse summary
    print("\nPattern Collapse Analysis:")
    for layer_name, stats in pattern_stats.items():
        print(f"\n  {layer_name}:")
        max_sim = stats.get('max_similarity', None)
        if max_sim is not None:
            print(f"    Max similarity: {max_sim:.4f}")
        else:
            print(f"    Max similarity: N/A")
        mean_sim = stats.get('mean_similarity', None)
        if mean_sim is not None:
            print(f"    Mean similarity: {mean_sim:.4f}")
        else:
            print(f"    Mean similarity: N/A")
        pattern_div = stats.get('pattern_diversity', None)
        if pattern_div is not None:
            print(f"    Pattern diversity: {pattern_div:.4f}")
        else:
            print(f"    Pattern diversity: N/A")
        print(f"    Is collapsed: {stats.get('is_collapsed', False)}")
        if "active_patterns" in stats:
            print(f"    Active patterns: {stats['active_patterns']}/{stats.get('total_patterns', 0)}")
        if "mean_attention_entropy" in stats:
            print(f"    Mean attention entropy: {stats['mean_attention_entropy']:.4f}")
            max_ent = stats.get('max_entropy', None)
            if max_ent is not None:
                print(f"    Max entropy (uniform): {max_ent:.4f}")
            else:
                print(f"    Max entropy (uniform): N/A")
    
    # Save comprehensive results
    diagnostic_results = {
        "config": config,
        "dataset": args.dataset,
        "seed": args.seed,
        "energy_analysis": energy_stats,
        "pattern_analysis": pattern_stats,
        "training_results": {
            "final_test_acc": result.get("training_history", {}).get("final_test_acc", 0),
            "final_val_acc": result.get("training_history", {}).get("val_acc", [0])[-1] if result.get("training_history", {}).get("val_acc") else 0,
        },
    }
    
    results_file = output_dir / "diagnostic_results.json"
    with open(results_file, "w") as f:
        json.dump(diagnostic_results, f, indent=2, default=str)
    
    print(f"\n{'='*70}")
    print("DIAGNOSTIC COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to: {results_file}")
    print(f"Visualizations saved to: {output_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
