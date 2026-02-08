#!/usr/bin/env python3
"""
Fair Comparison Script: Hyperparameter tuning for all methods.

For each model, performs grid search using validation set, then evaluates
the best configuration on the held-out test set with corruption benchmarks.

Usage:
    python experiments/fair_comparison.py --dataset amazon_photo --seeds 5
"""

import argparse
import sys
from pathlib import Path
from itertools import product
from typing import Dict, List, Any, Tuple
import json
from datetime import datetime

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.train import get_device, create_model, train_model, evaluate
from src.data.datasets import load_dataset
from src.data.corruption import apply_corruption, CorruptionConfig


# Hyperparameter search spaces for each model
SEARCH_SPACES = {
    'ghn': {
        'hidden_dim': [64, 128],
        'num_heads': [1, 2, 4],
        'num_patterns': [32, 64, 128],
        'beta': [0.1, 0.2, 0.5],
        'lambda_graph': [0.1, 0.3, 0.5],
        'num_iterations': [3, 4, 5],
        'dropout': [0.3, 0.5],
        'lr': [0.01, 0.005],
    },
    'gcn': {
        'hidden_dim': [64, 128, 256],
        'num_layers': [2, 3],
        'dropout': [0.3, 0.5, 0.7],
        'lr': [0.01, 0.005],
        'weight_decay': [5e-4, 1e-3, 5e-3],
    },
    'gat': {
        'hidden_dim': [64, 128],
        'num_layers': [2, 3],
        'num_heads': [4, 8],
        'dropout': [0.3, 0.5, 0.6],
        'lr': [0.01, 0.005],
        'weight_decay': [5e-4, 1e-3],
    },
    'graphsage': {
        'hidden_dim': [64, 128, 256],
        'num_layers': [2, 3],
        'dropout': [0.3, 0.5],
        'lr': [0.01, 0.005],
        'weight_decay': [5e-4, 1e-3],
    },
}

# Reduced search space for faster iteration
SEARCH_SPACES_FAST = {
    'ghn': {
        'hidden_dim': [64],
        'num_heads': [1, 2, 4],
        'num_patterns': [64],
        'beta': [0.1, 0.2],
        'lambda_graph': [0.2, 0.3],
        'num_iterations': [4],
        'dropout': [0.5],
        'lr': [0.01],
    },
    'gcn': {
        'hidden_dim': [64, 128],
        'num_layers': [2],
        'dropout': [0.5, 0.7],
        'lr': [0.01],
        'weight_decay': [5e-4, 1e-4],  # Added lower weight decay option
    },
    'gat': {
        'hidden_dim': [64],  # Reduced from [64, 128] to save memory on large graphs
        'num_layers': [2],
        'num_heads': [4, 8],
        'dropout': [0.5, 0.6],
        'lr': [0.01, 0.005],  # Added lower LR option
        'weight_decay': [5e-4],
    },
    'graphsage': {
        'hidden_dim': [64, 128],
        'num_layers': [2],
        'dropout': [0.5],
        'lr': [0.01],
        'weight_decay': [5e-4],
    },
}


def generate_configs(search_space: Dict[str, List]) -> List[Dict]:
    """Generate all combinations of hyperparameters."""
    keys = list(search_space.keys())
    values = list(search_space.values())
    
    configs = []
    for combo in product(*values):
        config = dict(zip(keys, combo))
        configs.append(config)
    
    return configs


def run_single_config(
    base_config: Dict,
    model_name: str,
    hp_config: Dict,
    data,
    dataset_info: Dict,
    device: torch.device,
    seed: int,
) -> Tuple[float, float]:
    """
    Train and evaluate a single hyperparameter configuration.
    
    Returns:
        (val_acc, test_acc)
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Build config
    config = yaml.safe_load(yaml.dump(base_config))
    config['model']['name'] = model_name
    
    # Apply hyperparameters
    for key, value in hp_config.items():
        if key in ['lr', 'weight_decay']:
            config['training'][key] = value
        else:
            config['model'][key] = value
    
    # Create and train model
    model = create_model(config, dataset_info['num_features'], dataset_info['num_classes'])
    model = model.to(device)
    model, history = train_model(model, data, config, device, verbose=False)
    
    # Get validation and test accuracy
    metrics, _ = evaluate(model, data, device)
    
    return metrics['val_acc'], metrics['test_acc']


def hyperparameter_search(
    base_config: Dict,
    model_name: str,
    search_space: Dict[str, List],
    data,
    dataset_info: Dict,
    device: torch.device,
    num_seeds: int = 3,
    verbose: bool = True,
) -> Tuple[Dict, float]:
    """
    Grid search over hyperparameters using validation set.
    
    Returns:
        (best_config, best_val_acc)
    """
    configs = generate_configs(search_space)
    
    if verbose:
        print(f"  Searching {len(configs)} configurations...")
    
    best_config = None
    best_val_acc = -1
    
    for i, hp_config in enumerate(configs):
        # Run multiple seeds and average
        val_accs = []
        for seed in range(num_seeds):
            val_acc, _ = run_single_config(
                base_config, model_name, hp_config, data, dataset_info, device, seed
            )
            val_accs.append(val_acc)
        
        avg_val_acc = sum(val_accs) / len(val_accs)
        
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            best_config = hp_config.copy()
        
        if verbose and (i + 1) % 10 == 0:
            print(f"    [{i+1}/{len(configs)}] Best val acc so far: {best_val_acc*100:.2f}%")
    
    return best_config, best_val_acc


def evaluate_with_corruption(
    base_config: Dict,
    model_name: str,
    hp_config: Dict,
    data,
    dataset_info: Dict,
    device: torch.device,
    num_seeds: int = 5,
) -> Dict[str, Any]:
    """
    Evaluate a configuration with multiple seeds and corruption benchmarks.
    
    Returns dict with:
        - test_acc_mean, test_acc_std
        - feature_noise results
        - edge_drop results  
        - label_flip results
    """
    results = {
        'clean': [],
        'feature_noise': {0.0: [], 0.2: [], 0.4: [], 0.6: [], 0.8: [], 1.0: []},
        'edge_drop': {0.0: [], 0.1: [], 0.2: [], 0.3: [], 0.4: [], 0.5: []},
        'label_flip': {0.0: [], 0.1: [], 0.2: [], 0.3: [], 0.4: []},
    }
    
    for seed in range(num_seeds):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        # Reload data for each seed (important for label flip which modifies training labels)
        data_fresh, _ = load_dataset(
            name=base_config['dataset']['name'],
            root=base_config['dataset']['root'],
            normalize_features=base_config['dataset']['normalize_features'],
        )
        
        # Build config
        config = yaml.safe_load(yaml.dump(base_config))
        config['model']['name'] = model_name
        for key, value in hp_config.items():
            if key in ['lr', 'weight_decay']:
                config['training'][key] = value
            else:
                config['model'][key] = value
        
        # Train on clean data
        model = create_model(config, dataset_info['num_features'], dataset_info['num_classes'])
        model = model.to(device)
        model, _ = train_model(model, data_fresh, config, device, verbose=False)
        
        # Clean test accuracy
        metrics, _ = evaluate(model, data_fresh, device)
        results['clean'].append(metrics['test_acc'])
        
        # Feature noise (test-time corruption)
        for noise_level in results['feature_noise'].keys():
            if noise_level == 0:
                test_data = data_fresh
            else:
                corrupt_config = CorruptionConfig(feature_noise_std=noise_level)
                test_data = apply_corruption(data_fresh, corrupt_config)
            
            metrics, _ = evaluate(model, test_data, device)
            results['feature_noise'][noise_level].append(metrics['test_acc'])
        
        # Edge drop (test-time corruption)
        for drop_level in results['edge_drop'].keys():
            if drop_level == 0:
                test_data = data_fresh
            else:
                corrupt_config = CorruptionConfig(edge_drop_ratio=drop_level)
                test_data = apply_corruption(data_fresh, corrupt_config)
            
            metrics, _ = evaluate(model, test_data, device)
            results['edge_drop'][drop_level].append(metrics['test_acc'])
        
        # Label flip (train-time corruption) - need to retrain
        for flip_level in results['label_flip'].keys():
            torch.manual_seed(seed)  # Reset seed for reproducibility
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            
            # Reload fresh data
            data_for_flip, _ = load_dataset(
                name=base_config['dataset']['name'],
                root=base_config['dataset']['root'],
                normalize_features=base_config['dataset']['normalize_features'],
            )
            
            if flip_level > 0:
                corrupt_config = CorruptionConfig(label_flip_ratio=flip_level)
                train_data = apply_corruption(data_for_flip, corrupt_config)
            else:
                train_data = data_for_flip
            
            # Retrain model on corrupted labels
            model_flip = create_model(config, dataset_info['num_features'], dataset_info['num_classes'])
            model_flip = model_flip.to(device)
            model_flip, _ = train_model(model_flip, train_data, config, device, verbose=False)
            
            # Test on clean data
            metrics, _ = evaluate(model_flip, data_for_flip, device)
            results['label_flip'][flip_level].append(metrics['test_acc'])
    
    # Compute statistics
    def compute_stats(values):
        mean = sum(values) / len(values)
        std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
        return mean, std
    
    output = {
        'test_acc_mean': compute_stats(results['clean'])[0],
        'test_acc_std': compute_stats(results['clean'])[1],
    }
    
    for corruption_type in ['feature_noise', 'edge_drop', 'label_flip']:
        output[corruption_type] = {}
        for level, values in results[corruption_type].items():
            mean, std = compute_stats(values)
            output[corruption_type][level] = {'mean': mean, 'std': std}
    
    return output


def print_results_table(all_results: Dict[str, Dict], dataset_name: str):
    """Print a formatted results table."""
    
    print("\n" + "=" * 80)
    print(f"📊 FINAL RESULTS: {dataset_name.upper()}")
    print("=" * 80)
    
    # Clean accuracy
    print("\n### Clean Accuracy (after hyperparameter tuning)")
    print("-" * 60)
    print(f"{'Model':<15} {'Test Acc':>12} {'Best Config'}")
    print("-" * 60)
    
    sorted_models = sorted(all_results.keys(), 
                          key=lambda m: all_results[m]['final_results']['test_acc_mean'], 
                          reverse=True)
    
    for model in sorted_models:
        res = all_results[model]
        acc = res['final_results']['test_acc_mean'] * 100
        std = res['final_results']['test_acc_std'] * 100
        config_str = ', '.join(f"{k}={v}" for k, v in list(res['best_config'].items())[:3])
        marker = "🏆" if model == sorted_models[0] else "  "
        print(f"{marker} {model:<12} {acc:>6.2f}±{std:.2f}%  {config_str}...")
    
    # Feature noise
    print("\n### Robustness: Feature Noise (test-time)")
    print("-" * 70)
    header = f"{'Model':<12}"
    for level in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        header += f" {level:.1f}   "
    print(header)
    print("-" * 70)
    
    for model in sorted_models:
        row = f"{model:<12}"
        for level in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            acc = all_results[model]['final_results']['feature_noise'][level]['mean'] * 100
            row += f" {acc:5.1f} "
        print(row)
    
    # Edge drop
    print("\n### Robustness: Edge Drop (test-time)")
    print("-" * 60)
    header = f"{'Model':<12}"
    for level in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        header += f" {int(level*100):>3}%  "
    print(header)
    print("-" * 60)
    
    for model in sorted_models:
        row = f"{model:<12}"
        for level in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
            acc = all_results[model]['final_results']['edge_drop'][level]['mean'] * 100
            row += f" {acc:5.1f} "
        print(row)
    
    # Label flip
    print("\n### Robustness: Label Flip (train-time)")
    print("-" * 55)
    header = f"{'Model':<12}"
    for level in [0.0, 0.1, 0.2, 0.3, 0.4]:
        header += f" {int(level*100):>3}%  "
    print(header)
    print("-" * 55)
    
    for model in sorted_models:
        row = f"{model:<12}"
        for level in [0.0, 0.1, 0.2, 0.3, 0.4]:
            acc = all_results[model]['final_results']['label_flip'][level]['mean'] * 100
            row += f" {acc:5.1f} "
        print(row)
    
    # Summary
    print("\n" + "=" * 80)
    print("📈 SUMMARY")
    print("=" * 80)
    
    winner = sorted_models[0]
    runner_up = sorted_models[1] if len(sorted_models) > 1 else None
    
    winner_acc = all_results[winner]['final_results']['test_acc_mean'] * 100
    if runner_up:
        runner_up_acc = all_results[runner_up]['final_results']['test_acc_mean'] * 100
        gap = winner_acc - runner_up_acc
        print(f"🏆 Winner: {winner} ({winner_acc:.2f}%)")
        print(f"   Runner-up: {runner_up} ({runner_up_acc:.2f}%)")
        print(f"   Gap: +{gap:.2f}%")
    
    # Degradation comparison
    print("\n📉 Degradation under 40% label flip:")
    for model in sorted_models:
        clean = all_results[model]['final_results']['label_flip'][0.0]['mean'] * 100
        noisy = all_results[model]['final_results']['label_flip'][0.4]['mean'] * 100
        drop = clean - noisy
        print(f"   {model}: {clean:.1f}% → {noisy:.1f}% (lost {drop:.1f} pts)")


def main():
    parser = argparse.ArgumentParser(description="Fair comparison with hyperparameter tuning")
    parser.add_argument("--dataset", type=str, default="amazon_photo",
                       help="Dataset name (amazon_photo, amazon_computers, cora, citeseer, pubmed)")
    parser.add_argument("--seeds", type=int, default=5,
                       help="Number of seeds for final evaluation")
    parser.add_argument("--tuning-seeds", type=int, default=2,
                       help="Number of seeds for hyperparameter tuning")
    parser.add_argument("--fast", action="store_true",
                       help="Use reduced search space for faster iteration")
    parser.add_argument("--output", type=str, default=None,
                       help="Output JSON file for results")
    parser.add_argument("--models", type=str, nargs="+", 
                       default=["ghn", "gcn", "gat", "graphsage"],
                       help="Models to compare")
    
    args = parser.parse_args()
    
    # Load base config
    config_path = Path(__file__).parent / "configs" / "default.yaml"
    with open(config_path, "r") as f:
        base_config = yaml.safe_load(f)
    
    base_config['dataset']['name'] = args.dataset
    
    device = get_device(base_config['experiment']['device'])
    search_spaces = SEARCH_SPACES_FAST if args.fast else SEARCH_SPACES
    
    print("=" * 80)
    print(f"🔬 FAIR COMPARISON: {args.dataset.upper()}")
    print("=" * 80)
    print(f"Models: {', '.join(args.models)}")
    print(f"Search mode: {'FAST' if args.fast else 'FULL'}")
    print(f"Tuning seeds: {args.tuning_seeds}, Final eval seeds: {args.seeds}")
    print(f"Device: {device}")
    print("=" * 80)
    
    # Load dataset once
    data, dataset_info = load_dataset(
        name=args.dataset,
        root=base_config['dataset']['root'],
        normalize_features=base_config['dataset']['normalize_features'],
    )
    
    print(f"\nDataset: {dataset_info['name']}")
    print(f"  Nodes: {dataset_info['num_nodes']}")
    print(f"  Edges: {dataset_info['num_edges']}")
    print(f"  Features: {dataset_info['num_features']}")
    print(f"  Classes: {dataset_info['num_classes']}")
    print(f"  Train/Val/Test: {dataset_info['num_train']}/{dataset_info.get('num_val', 'N/A')}/{dataset_info.get('num_test', 'N/A')}")
    
    all_results = {}
    
    for model_name in args.models:
        print(f"\n{'='*60}")
        print(f"🔍 {model_name.upper()}: Hyperparameter Search")
        print(f"{'='*60}")
        
        if model_name not in search_spaces:
            print(f"  ⚠️ No search space defined for {model_name}, skipping")
            continue
        
        # Hyperparameter search using validation set
        best_config, best_val_acc = hyperparameter_search(
            base_config=base_config,
            model_name=model_name,
            search_space=search_spaces[model_name],
            data=data,
            dataset_info=dataset_info,
            device=device,
            num_seeds=args.tuning_seeds,
            verbose=True,
        )
        
        print(f"\n  ✅ Best config (val acc: {best_val_acc*100:.2f}%):")
        for k, v in best_config.items():
            print(f"     {k}: {v}")
        
        # Final evaluation with corruption benchmarks
        print(f"\n  📊 Final evaluation ({args.seeds} seeds + corruption)...")
        final_results = evaluate_with_corruption(
            base_config=base_config,
            model_name=model_name,
            hp_config=best_config,
            data=data,
            dataset_info=dataset_info,
            device=device,
            num_seeds=args.seeds,
        )
        
        print(f"  ✅ Test accuracy: {final_results['test_acc_mean']*100:.2f}±{final_results['test_acc_std']*100:.2f}%")
        
        all_results[model_name] = {
            'best_config': best_config,
            'best_val_acc': best_val_acc,
            'final_results': final_results,
        }
    
    # Print final results table
    print_results_table(all_results, args.dataset)
    
    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(base_config['experiment']['results_dir']) / f"fair_comparison_{args.dataset}_{timestamp}.json"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to JSON-serializable format
    json_results = {
        'dataset': args.dataset,
        'dataset_info': dataset_info,
        'models': {}
    }
    for model_name, res in all_results.items():
        json_results['models'][model_name] = {
            'best_config': res['best_config'],
            'best_val_acc': float(res['best_val_acc']),
            'test_acc_mean': float(res['final_results']['test_acc_mean']),
            'test_acc_std': float(res['final_results']['test_acc_std']),
            'feature_noise': {str(k): {'mean': float(v['mean']), 'std': float(v['std'])} 
                            for k, v in res['final_results']['feature_noise'].items()},
            'edge_drop': {str(k): {'mean': float(v['mean']), 'std': float(v['std'])} 
                         for k, v in res['final_results']['edge_drop'].items()},
            'label_flip': {str(k): {'mean': float(v['mean']), 'std': float(v['std'])} 
                          for k, v in res['final_results']['label_flip'].items()},
        }
    
    with open(output_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\n📁 Results saved to: {output_path}")


if __name__ == "__main__":
    main()
