"""Test script to compare GHN with random vs data-initialized memory.

This tests whether initializing memory patterns from k-means clusters of
node features improves performance compared to random initialization.
"""

import sys
from pathlib import Path
import yaml
import json

import torch
import torch.nn as nn
from sklearn.cluster import KMeans
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.datasets import load_dataset
from src.models.ghn import GraphHopfieldNetwork
from experiments.train import (
    get_device, train_model, evaluate, run_corruption_benchmark,
    run_experiment, run_full_experiment, save_results
)


def initialize_memory_from_data(model: GraphHopfieldNetwork, data, num_patterns: int, strategy: str = "class_conditioned"):
    """
    Initialize memory patterns from k-means clusters of node features.
    
    Args:
        model: GraphHopfieldNetwork model
        data: PyTorch Geometric data object
        num_patterns: Number of memory patterns (clusters)
        strategy: "class_conditioned" (cluster within each class) or "global" (cluster all nodes)
    """
    # Project to hidden dimension first
    model.eval()
    with torch.no_grad():
        # Encode features through the encoder to get the right feature space
        encoded = model.encoder(data.x)
        hidden_dim = encoded.shape[1]
        encoded_np = encoded.cpu().numpy()
        
        # Run k-means clustering in the encoded space (where memory patterns live)
        print(f"  Running k-means with k={num_patterns} on encoded features (dim={hidden_dim})...")
        kmeans = KMeans(n_clusters=num_patterns, random_state=42, n_init=10)
        kmeans.fit(encoded_np)
        
        # Get cluster centers (already in the right dimension)
        cluster_centers = kmeans.cluster_centers_
        
        # Convert to tensor
        cluster_tensor = torch.tensor(cluster_centers, dtype=encoded.dtype, device=encoded.device)
        
        # Initialize memory keys with cluster centers
        # Don't normalize or scale - cluster centers are already in the right space and scale
        for layer in model.ghn_layers:
            if hasattr(layer, 'memory') and hasattr(layer.memory, 'keys'):
                layer.memory.keys.data.copy_(cluster_tensor)
                print(f"  Initialized memory patterns from k-means clusters (shape: {cluster_tensor.shape}, "
                      f"norm range: [{cluster_tensor.norm(dim=1).min():.3f}, {cluster_tensor.norm(dim=1).max():.3f}])")


def create_model_with_memory_init(
    config: dict,
    in_dim: int,
    out_dim: int,
    data,
    use_data_init: bool = False,
) -> nn.Module:
    """Create model with optional data-based memory initialization."""
    from experiments.train import create_model
    
    model = create_model(config, in_dim, out_dim)
    
    if use_data_init and config["model"]["name"].lower() == "ghn":
        strategy = config["model"].get("memory_init_strategy", "class_conditioned")
        print(f"Initializing memory from data (k-means, strategy={strategy})...")
        initialize_memory_from_data(model, data, config["model"]["num_patterns"], strategy=strategy)
    
    return model


def run_experiment_with_memory_init(
    config: dict,
    seed: int,
    device: torch.device,
    use_data_init: bool = False,
    verbose: bool = True,
) -> dict:
    """Run experiment with optional data-initialized memory."""
    import torch
    from experiments.train import get_device, train_model, evaluate, run_corruption_benchmark
    from src.data.datasets import load_dataset, print_dataset_info
    
    # Set seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Load dataset
    data, dataset_info = load_dataset(
        name=config["dataset"]["name"],
        root=config["dataset"]["root"],
        normalize_features=config["dataset"]["normalize_features"],
    )
    
    if verbose:
        print_dataset_info(dataset_info)
    
    # Create model with memory initialization
    model = create_model_with_memory_init(
        config,
        in_dim=dataset_info["num_features"],
        out_dim=dataset_info["num_classes"],
        data=data,
        use_data_init=use_data_init,
    )
    model = model.to(device)
    
    if verbose:
        print(f"\nModel: {config['model']['name']}")
        print(f"Memory init: {'Data (k-means)' if use_data_init else 'Random'}")
        print(f"Parameters: {model.get_num_params():,}")
    
    # Train
    if verbose:
        print("\nTraining...")
    model, history = train_model(model, data, config, device, verbose)
    
    if verbose:
        print(f"\nFinal Test Accuracy: {history['final_test_acc']:.4f}")
    
    # Run corruption benchmark
    if verbose:
        print("\nCorruption Benchmark:")
    corruption_results = run_corruption_benchmark(model, data, config, device, verbose)
    
    return {
        "seed": seed,
        "dataset": dataset_info,
        "model": config["model"]["name"],
        "memory_init": "data" if use_data_init else "random",
        "num_params": model.get_num_params(),
        "training_history": history,
        "corruption_results": corruption_results,
    }


def main():
    """Run experiments comparing random vs data-initialized memory."""
    
    # Load default config
    config_path = Path(__file__).parent / "configs" / "default.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Ensure we're testing GHN
    config["model"]["name"] = "ghn"
    config["experiment"]["num_seeds"] = 3  # Fewer seeds for quick test
    
    print("=" * 70)
    print("Testing Graph Hopfield Network: Random vs Data-Initialized Memory")
    print("=" * 70)
    
    device = get_device(config["experiment"]["device"])
    num_seeds = config["experiment"]["num_seeds"]
    
    results = {}
    
    # Test with RANDOM initialization (default)
    print("\n" + "=" * 70)
    print("Experiment 1: Random Memory Initialization (default)")
    print("=" * 70)
    
    all_results_random = []
    for seed in range(num_seeds):
        if seed > 0:
            print(f"\n{'='*50}")
        print(f"Seed {seed + 1}/{num_seeds}")
        result = run_experiment_with_memory_init(
            config, seed, device, use_data_init=False, verbose=True
        )
        all_results_random.append(result)
    
    # Aggregate random results
    test_accs = [r["training_history"]["final_test_acc"] for r in all_results_random]
    aurc_values = [r["corruption_results"]["robustness"]["normalized_aurc"] for r in all_results_random]
    
    if num_seeds > 1:
        test_acc_std = float(torch.tensor(test_accs).std())
        aurc_std = float(torch.tensor(aurc_values).std())
    else:
        test_acc_std = 0.0
        aurc_std = 0.0
    
    results["random_init"] = {
        "test_acc_mean": float(torch.tensor(test_accs).mean()),
        "test_acc_std": test_acc_std,
        "aurc_mean": float(torch.tensor(aurc_values).mean()),
        "aurc_std": aurc_std,
    }
    
    # Test with DATA initialization
    print("\n" + "=" * 70)
    print("Experiment 2: Data-Initialized Memory (k-means clusters)")
    print("=" * 70)
    
    all_results_data = []
    for seed in range(num_seeds):
        if seed > 0:
            print(f"\n{'='*50}")
        print(f"Seed {seed + 1}/{num_seeds}")
        result = run_experiment_with_memory_init(
            config, seed, device, use_data_init=True, verbose=True
        )
        all_results_data.append(result)
    
    # Aggregate data init results
    test_accs = [r["training_history"]["final_test_acc"] for r in all_results_data]
    aurc_values = [r["corruption_results"]["robustness"]["normalized_aurc"] for r in all_results_data]
    
    if num_seeds > 1:
        test_acc_std = float(torch.tensor(test_accs).std())
        aurc_std = float(torch.tensor(aurc_values).std())
    else:
        test_acc_std = 0.0
        aurc_std = 0.0
    
    results["data_init"] = {
        "test_acc_mean": float(torch.tensor(test_accs).mean()),
        "test_acc_std": test_acc_std,
        "aurc_mean": float(torch.tensor(aurc_values).mean()),
        "aurc_std": aurc_std,
    }
    
    # Print comparison
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"\nRandom Initialization:")
    print(f"  Test Accuracy: {results['random_init']['test_acc_mean']*100:.2f} ± {results['random_init']['test_acc_std']*100:.2f}%")
    print(f"  AURC:          {results['random_init']['aurc_mean']*100:.2f} ± {results['random_init']['aurc_std']*100:.2f}%")
    
    print(f"\nData Initialization (k-means):")
    print(f"  Test Accuracy: {results['data_init']['test_acc_mean']*100:.2f} ± {results['data_init']['test_acc_std']*100:.2f}%")
    print(f"  AURC:          {results['data_init']['aurc_mean']*100:.2f} ± {results['data_init']['aurc_std']*100:.2f}%")
    
    diff_acc = results['data_init']['test_acc_mean'] - results['random_init']['test_acc_mean']
    diff_aurc = results['data_init']['aurc_mean'] - results['random_init']['aurc_mean']
    
    print(f"\nDifference (Data Init - Random Init):")
    print(f"  Test Accuracy: {diff_acc*100:+.2f}%")
    print(f"  AURC:          {diff_aurc*100:+.2f}%")
    
    # Save results
    if config["experiment"]["save_results"]:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = Path(config["experiment"]["results_dir"]) / f"memory_init_comparison_{timestamp}.json"
        results_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_file, "w") as f:
            json.dump({
                "config": config,
                "comparison": results,
                "full_results_random": all_results_random,
                "full_results_data": all_results_data,
            }, f, indent=2, default=str)
        
        print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
