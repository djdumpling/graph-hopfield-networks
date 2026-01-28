"""Training script for Graph Hopfield Networks experiments."""

import argparse
import os
import sys
import yaml
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union

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
            use_layer_norm=config["model"].get("use_layer_norm", True),
            normalize_memory_keys=config["model"].get("normalize_memory_keys", False),
            normalize_memory_queries=config["model"].get("normalize_memory_queries", False),
            tie_keys_values=config["model"].get("tie_keys_values", False),
            learnable_beta=config["model"].get("learnable_beta", True),
            use_spectral_norm_constraint=config["model"].get("use_spectral_norm_constraint", True),
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
            use_layer_norm=config["model"].get("use_layer_norm", True),
            normalize_memory_keys=config["model"].get("normalize_memory_keys", False),
            normalize_memory_queries=config["model"].get("normalize_memory_queries", False),
            tie_keys_values=config["model"].get("tie_keys_values", False),
            learnable_beta=config["model"].get("learnable_beta", True),
            use_spectral_norm_constraint=config["model"].get("use_spectral_norm_constraint", True),
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
    log_energy: bool = False,
    log_attention: bool = False,
) -> Tuple[float, Optional[dict]]:
    """Train for one epoch."""
    model.train()
    optimizer.zero_grad()
    
    data = data.to(device)
    
    # Forward pass with optional energy/attention logging
    if log_energy or log_attention:
        out, info = model(
            data.x, 
            data.edge_index,
            return_energy=log_energy,
            return_attention=log_attention,
        )
    else:
        out, info = model(data.x, data.edge_index)
    
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    
    return loss.item(), info


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data,
    device: torch.device,
    log_energy: bool = False,
    log_attention: bool = False,
) -> Tuple[Dict[str, float], Optional[dict]]:
    """Evaluate on all splits."""
    model.eval()
    data = data.to(device)
    
    # Forward pass with optional energy/attention logging
    if log_energy or log_attention:
        out, info = model(
            data.x,
            data.edge_index,
            return_energy=log_energy,
            return_attention=log_attention,
        )
    else:
        out, info = model(data.x, data.edge_index)
    
    results = {
        "train_acc": compute_accuracy(out, data.y, data.train_mask),
        "val_acc": compute_accuracy(out, data.y, data.val_mask),
        "test_acc": compute_accuracy(out, data.y, data.test_mask),
    }
    
    return results, info


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
    
    # Energy and attention logging (if enabled)
    log_energy = config.get("experiment", {}).get("log_energy", False)
    log_attention = config.get("experiment", {}).get("log_attention", False)
    
    if log_energy:
        history["energies"] = []
    if log_attention:
        history["attentions"] = []
    
    epochs = training_config["epochs"]
    patience = training_config["patience"]
    min_delta = training_config["min_delta"]
    
    pbar = tqdm(range(epochs), disable=not verbose, desc="Training")
    
    for epoch in pbar:
        # Train
        loss, train_info = train_epoch(model, data, optimizer, device, log_energy, log_attention)
        
        # Evaluate
        metrics, eval_info = evaluate(model, data, device, log_energy, log_attention)
        
        history["train_loss"].append(loss)
        history["train_acc"].append(metrics["train_acc"])
        history["val_acc"].append(metrics["val_acc"])
        history["test_acc"].append(metrics["test_acc"])
        
        # Log energy if enabled
        if log_energy and eval_info is not None and "energies" in eval_info:
            history["energies"].append(eval_info["energies"])
        
        # Log attention if enabled (store periodically to avoid memory issues)
        if log_attention and eval_info is not None and "attentions" in eval_info:
            # Store attention every 10 epochs or at final epoch
            if epoch % 10 == 0 or epoch == epochs - 1:
                history["attentions"].append({
                    "epoch": epoch,
                    "attention": eval_info["attentions"],
                })
        
        # Early stopping check
        if metrics["val_acc"] > best_val_acc + min_delta:
            best_val_acc = metrics["val_acc"]
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Store best model state in history for diagnostics
        if best_model_state is not None:
            history["best_model_state"] = best_model_state
        
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
    log_energy = config.get("experiment", {}).get("log_energy", False)
    log_attention = config.get("experiment", {}).get("log_attention", False)
    final_metrics, _ = evaluate(model, data, device, log_energy, log_attention)
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
    
    # Calculate std, handling case when num_seeds = 1
    if num_seeds > 1:
        test_acc_std = float(torch.tensor(test_accs).std())
        aurc_std = float(torch.tensor(aurc_values).std())
    else:
        test_acc_std = 0.0
        aurc_std = 0.0
    
    aggregated = {
        "config": config,
        "num_seeds": num_seeds,
        "test_acc_mean": float(torch.tensor(test_accs).mean()),
        "test_acc_std": test_acc_std,
        "aurc_mean": float(torch.tensor(aurc_values).mean()),
        "aurc_std": aurc_std,
        "all_results": all_results,
    }
    
    if verbose:
        print(f"\n{'='*50}")
        print("AGGREGATED RESULTS")
        print(f"{'='*50}")
        if num_seeds > 1:
            print(f"Test Accuracy: {aggregated['test_acc_mean']*100:.2f} ± {aggregated['test_acc_std']*100:.2f}%")
            print(f"AURC: {aggregated['aurc_mean']*100:.2f} ± {aggregated['aurc_std']*100:.2f}%")
        else:
            print(f"Test Accuracy: {aggregated['test_acc_mean']*100:.2f}% (single seed)")
            print(f"AURC: {aggregated['aurc_mean']*100:.2f}% (single seed)")
    
    return aggregated


def save_results(results: Dict[str, Any], config: Dict[str, Any]) -> str:
    """Save results to file with descriptive filename."""
    results_dir = Path(config["experiment"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    
    model_name = config["model"]["name"]
    dataset_name = config["dataset"]["name"]
    
    # Build descriptive filename with key hyperparameters
    parts = [model_name, dataset_name]
    
    if model_name.lower() == "ghn":
        # Add GHN-specific hyperparameters
        parts.append(f"b{config['model']['beta']:.2f}")
        parts.append(f"l{config['model']['lambda_graph']:.2f}")
        parts.append(f"iter{config['model']['num_iterations']}")
        parts.append(f"pat{config['model']['num_patterns']}")
        parts.append(f"a{config['model']['alpha']:.1f}")
        parts.append(f"lay{config['model']['num_layers']}")
        parts.append(f"dr{config['model']['dropout']:.1f}")
        parts.append(f"lr{config['training']['lr']:.3f}")
        parts.append(f"wd{config['training']['weight_decay']:.4f}")
    
    parts.append(f"s{config['experiment']['num_seeds']}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts.append(timestamp)
    
    filename = "_".join(parts) + ".json"
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
    parser.add_argument(
        "--beta",
        type=float,
        default=None,
        help="Override beta (inverse temperature) for GHN",
    )
    parser.add_argument(
        "--lambda",
        type=float,
        default=None,
        dest="lambda_graph",
        help="Override lambda_graph (graph coupling weight) for GHN",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=None,
        dest="num_iterations",
        help="Override num_iterations (Hopfield iterations per layer) for GHN",
    )
    parser.add_argument(
        "--num-patterns",
        type=int,
        default=None,
        dest="num_patterns",
        help="Override num_patterns (memory bank size) for GHN",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Override alpha (damping coefficient) for GHN",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        dest="num_layers",
        help="Override num_layers (number of GHN layers)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="Override dropout rate",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        dest="weight_decay",
        help="Override weight decay (L2 regularization)",
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
    if args.beta is not None:
        config["model"]["beta"] = args.beta
    if args.lambda_graph is not None:
        config["model"]["lambda_graph"] = args.lambda_graph
    if args.num_iterations is not None:
        config["model"]["num_iterations"] = args.num_iterations
    if args.num_patterns is not None:
        config["model"]["num_patterns"] = args.num_patterns
    if args.alpha is not None:
        config["model"]["alpha"] = args.alpha
    if args.num_layers is not None:
        config["model"]["num_layers"] = args.num_layers
    if args.dropout is not None:
        config["model"]["dropout"] = args.dropout
    if args.lr is not None:
        config["training"]["lr"] = args.lr
    if args.weight_decay is not None:
        config["training"]["weight_decay"] = args.weight_decay
    
    # Run experiment
    results = run_full_experiment(config, verbose=not args.quiet)
    
    # Save results
    if config["experiment"]["save_results"] and not args.no_save:
        filepath = save_results(results, config)
        print(f"\nResults saved to: {filepath}")


if __name__ == "__main__":
    main()
