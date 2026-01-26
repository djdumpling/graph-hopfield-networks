"""Training script for Graph Hopfield Networks experiments."""

import argparse
import os
import sys
import yaml
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.datasets import load_dataset, print_dataset_info
from src.data.corruption import apply_corruption, CorruptionConfig, get_corruption_levels
from src.models.ghn import GraphHopfieldNetwork, GraphHopfieldNetworkMinimal
from src.models.baselines import create_baseline
from src.utils.metrics import compute_accuracy, compute_robustness_curve, aggregate_results


def get_device(device_str: str = "auto") -> torch.device:
    """Get the appropriate device."""
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def create_model(
    config: Dict[str, Any],
    in_dim: int,
    out_dim: int,
) -> nn.Module:
    """Create a model based on configuration."""
    model_name = config["model"]["name"].lower()
    hidden_dim = config["model"]["hidden_dim"]
    dropout = config["model"]["dropout"]
    num_layers = config["model"].get("num_layers", 2)
    
    if model_name == "ghn":
        return GraphHopfieldNetwork(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_patterns=config["model"]["num_patterns"],
            beta=config["model"]["beta"],
            lambda_graph=config["model"]["lambda_graph"],
            num_iterations=config["model"]["num_iterations"],
            alpha=config["model"]["alpha"],
            num_layers=num_layers,
            dropout=dropout,
        )
    elif model_name == "ghn_minimal":
        return GraphHopfieldNetworkMinimal(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_patterns=config["model"]["num_patterns"],
            beta=config["model"]["beta"],
            lambda_graph=config["model"]["lambda_graph"],
            num_iterations=config["model"]["num_iterations"],
            alpha=config["model"]["alpha"],
            dropout=dropout,
        )
    else:
        return create_baseline(
            model_name=model_name,
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_layers=num_layers,
            dropout=dropout,
        )


def train_epoch(
    model: nn.Module,
    data,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch."""
    model.train()
    optimizer.zero_grad()
    
    data = data.to(device)
    out, _ = model(data.x, data.edge_index)
    
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    
    return loss.item()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate on all splits."""
    model.eval()
    data = data.to(device)
    
    out, _ = model(data.x, data.edge_index)
    
    results = {
        "train_acc": compute_accuracy(out, data.y, data.train_mask),
        "val_acc": compute_accuracy(out, data.y, data.val_mask),
        "test_acc": compute_accuracy(out, data.y, data.test_mask),
    }
    
    return results


def train_model(
    model: nn.Module,
    data,
    config: Dict[str, Any],
    device: torch.device,
    verbose: bool = True,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Train a model with early stopping.
    
    Returns:
        model: Trained model
        history: Training history
    """
    training_config = config["training"]
    
    optimizer = Adam(
        model.parameters(),
        lr=training_config["lr"],
        weight_decay=training_config["weight_decay"],
    )
    
    best_val_acc = 0
    best_model_state = None
    patience_counter = 0
    
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_acc": [],
        "test_acc": [],
    }
    
    epochs = training_config["epochs"]
    patience = training_config["patience"]
    min_delta = training_config["min_delta"]
    
    pbar = tqdm(range(epochs), disable=not verbose, desc="Training")
    
    for epoch in pbar:
        # Train
        loss = train_epoch(model, data, optimizer, device)
        
        # Evaluate
        metrics = evaluate(model, data, device)
        
        history["train_loss"].append(loss)
        history["train_acc"].append(metrics["train_acc"])
        history["val_acc"].append(metrics["val_acc"])
        history["test_acc"].append(metrics["test_acc"])
        
        # Early stopping
        if metrics["val_acc"] > best_val_acc + min_delta:
            best_val_acc = metrics["val_acc"]
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            if verbose:
                print(f"\nEarly stopping at epoch {epoch + 1}")
            break
        
        pbar.set_postfix({
            "loss": f"{loss:.4f}",
            "val_acc": f"{metrics['val_acc']:.4f}",
            "test_acc": f"{metrics['test_acc']:.4f}",
        })
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Final evaluation
    final_metrics = evaluate(model, data, device)
    history["final_test_acc"] = final_metrics["test_acc"]
    history["best_val_acc"] = best_val_acc
    
    return model, history


def run_corruption_benchmark(
    model: nn.Module,
    data,
    config: Dict[str, Any],
    device: torch.device,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run corruption benchmark on a trained model.
    
    Returns:
        Dict with corruption results
    """
    corruption_config = config["corruption"]
    corruption_type = corruption_config["type"]
    levels = corruption_config["levels"]
    
    configs = get_corruption_levels(corruption_type, levels)
    
    accuracies = []
    
    for level, corr_config in zip(levels, configs):
        corrupted_data = apply_corruption(data, corr_config)
        corrupted_data = corrupted_data.to(device)
        
        model.eval()
        with torch.no_grad():
            out, _ = model(corrupted_data.x, corrupted_data.edge_index)
            acc = compute_accuracy(out, corrupted_data.y, corrupted_data.test_mask)
        
        accuracies.append(acc)
        
        if verbose:
            print(f"  {corruption_type}={level:.2f}: Test Acc = {acc:.4f}")
    
    # Compute robustness metrics
    robustness = compute_robustness_curve(accuracies, levels)
    
    return {
        "corruption_type": corruption_type,
        "levels": levels,
        "accuracies": accuracies,
        "robustness": robustness,
    }


def run_experiment(
    config: Dict[str, Any],
    seed: int,
    device: torch.device,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run a single experiment with a given seed.
    
    Returns:
        Dict with experiment results
    """
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
    
    # Create model
    model = create_model(
        config,
        in_dim=dataset_info["num_features"],
        out_dim=dataset_info["num_classes"],
    )
    model = model.to(device)
    
    if verbose:
        print(f"\nModel: {config['model']['name']}")
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
        "num_params": model.get_num_params(),
        "training_history": history,
        "corruption_results": corruption_results,
    }


def run_full_experiment(config: Dict[str, Any], verbose: bool = True) -> Dict[str, Any]:
    """
    Run full experiment with multiple seeds.
    
    Returns:
        Aggregated results
    """
    device = get_device(config["experiment"]["device"])
    num_seeds = config["experiment"]["num_seeds"]
    
    if verbose:
        print(f"Device: {device}")
        print(f"Running {num_seeds} seeds...\n")
    
    all_results = []
    
    for seed in range(num_seeds):
        if verbose:
            print(f"\n{'='*50}")
            print(f"Seed {seed + 1}/{num_seeds}")
            print(f"{'='*50}")
        
        results = run_experiment(config, seed, device, verbose)
        all_results.append(results)
    
    # Aggregate results
    test_accs = [r["training_history"]["final_test_acc"] for r in all_results]
    aurc_values = [r["corruption_results"]["robustness"]["normalized_aurc"] for r in all_results]
    
    aggregated = {
        "config": config,
        "num_seeds": num_seeds,
        "test_acc_mean": float(torch.tensor(test_accs).mean()),
        "test_acc_std": float(torch.tensor(test_accs).std()),
        "aurc_mean": float(torch.tensor(aurc_values).mean()),
        "aurc_std": float(torch.tensor(aurc_values).std()),
        "all_results": all_results,
    }
    
    if verbose:
        print(f"\n{'='*50}")
        print("AGGREGATED RESULTS")
        print(f"{'='*50}")
        print(f"Test Accuracy: {aggregated['test_acc_mean']*100:.2f} ± {aggregated['test_acc_std']*100:.2f}%")
        print(f"AURC: {aggregated['aurc_mean']*100:.2f} ± {aggregated['aurc_std']*100:.2f}%")
    
    return aggregated


def save_results(results: Dict[str, Any], config: Dict[str, Any]) -> str:
    """Save results to file."""
    results_dir = Path(config["experiment"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = config["model"]["name"]
    dataset_name = config["dataset"]["name"]
    
    filename = f"{model_name}_{dataset_name}_{timestamp}.json"
    filepath = results_dir / filename
    
    # Convert non-serializable items
    def convert(obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2, default=convert)
    
    return str(filepath)


def main():
    parser = argparse.ArgumentParser(description="Train Graph Hopfield Networks")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/default.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override model name in config",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Override dataset name in config",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=None,
        help="Override number of seeds",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity",
    )
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Override config with command line args
    if args.model:
        config["model"]["name"] = args.model
    if args.dataset:
        config["dataset"]["name"] = args.dataset
    if args.seeds:
        config["experiment"]["num_seeds"] = args.seeds
    
    # Run experiment
    results = run_full_experiment(config, verbose=not args.quiet)
    
    # Save results
    if config["experiment"]["save_results"] and not args.no_save:
        filepath = save_results(results, config)
        print(f"\nResults saved to: {filepath}")


if __name__ == "__main__":
    main()
