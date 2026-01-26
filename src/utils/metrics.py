"""Evaluation metrics for Graph Hopfield Networks experiments."""

import torch
from torch import Tensor
import numpy as np
from typing import Dict, List, Optional, Tuple


def compute_accuracy(
    logits: Tensor,
    labels: Tensor,
    mask: Optional[Tensor] = None,
) -> float:
    """
    Compute classification accuracy.
    
    Args:
        logits: Model predictions [N, C]
        labels: Ground truth labels [N]
        mask: Optional mask for subset evaluation [N]
        
    Returns:
        Accuracy as a float
    """
    preds = logits.argmax(dim=-1)
    
    if mask is not None:
        correct = (preds[mask] == labels[mask]).sum().item()
        total = mask.sum().item()
    else:
        correct = (preds == labels).sum().item()
        total = labels.size(0)
    
    return correct / total if total > 0 else 0.0


def compute_robustness_curve(
    accuracies: List[float],
    corruption_levels: List[float],
) -> Dict[str, float]:
    """
    Compute robustness metrics from accuracy vs corruption curve.
    
    Args:
        accuracies: List of accuracies at each corruption level
        corruption_levels: List of corruption levels
        
    Returns:
        Dict with robustness metrics:
            - clean_acc: Accuracy at corruption=0
            - worst_acc: Minimum accuracy
            - aurc: Area under robustness curve
            - degradation: Relative drop from clean to worst
    """
    accuracies = np.array(accuracies)
    corruption_levels = np.array(corruption_levels)
    
    # Sort by corruption level
    sort_idx = np.argsort(corruption_levels)
    accuracies = accuracies[sort_idx]
    corruption_levels = corruption_levels[sort_idx]
    
    clean_acc = accuracies[0]
    worst_acc = accuracies.min()
    
    # Compute AURC (Area Under Robustness Curve)
    # Using trapezoidal rule
    aurc = np.trapz(accuracies, corruption_levels)
    
    # Normalized AURC (0-1 scale based on corruption range)
    max_level = corruption_levels.max()
    normalized_aurc = aurc / max_level if max_level > 0 else aurc
    
    # Degradation: relative drop from clean
    degradation = (clean_acc - worst_acc) / clean_acc if clean_acc > 0 else 0.0
    
    return {
        "clean_acc": clean_acc,
        "worst_acc": worst_acc,
        "aurc": aurc,
        "normalized_aurc": normalized_aurc,
        "degradation": degradation,
    }


def compute_aurc(
    accuracies: List[float],
    corruption_levels: List[float],
) -> float:
    """
    Compute Area Under Robustness Curve.
    
    Args:
        accuracies: List of accuracies at each corruption level
        corruption_levels: List of corruption levels
        
    Returns:
        AURC value
    """
    return compute_robustness_curve(accuracies, corruption_levels)["normalized_aurc"]


def compute_attention_entropy(
    attention_weights: Tensor,
) -> Tensor:
    """
    Compute entropy of attention weights.
    
    Higher entropy = more uniform attention (less certain retrieval).
    Lower entropy = sharper attention (more confident retrieval).
    
    Args:
        attention_weights: Attention weights [N, K]
        
    Returns:
        Entropy per node [N]
    """
    # Add small epsilon for numerical stability
    eps = 1e-10
    attn = attention_weights + eps
    entropy = -(attn * torch.log(attn)).sum(dim=-1)
    return entropy


def compute_prototype_usage(
    attention_weights: Tensor,
) -> Tensor:
    """
    Compute usage statistics for each prototype.
    
    Args:
        attention_weights: Attention weights [N, K]
        
    Returns:
        Average attention per prototype [K]
    """
    return attention_weights.mean(dim=0)


def aggregate_results(
    results: List[Dict[str, float]],
) -> Dict[str, Tuple[float, float]]:
    """
    Aggregate results across multiple runs.
    
    Args:
        results: List of result dicts from multiple runs
        
    Returns:
        Dict with (mean, std) for each metric
    """
    if not results:
        return {}
    
    keys = results[0].keys()
    aggregated = {}
    
    for key in keys:
        values = [r[key] for r in results]
        aggregated[key] = (np.mean(values), np.std(values))
    
    return aggregated


def format_results(
    results: Dict[str, Tuple[float, float]],
    precision: int = 2,
) -> str:
    """
    Format aggregated results as a string.
    
    Args:
        results: Dict with (mean, std) tuples
        precision: Decimal precision
        
    Returns:
        Formatted string
    """
    lines = []
    for key, (mean, std) in results.items():
        if "acc" in key.lower() or "aurc" in key.lower():
            # Format as percentage
            lines.append(f"{key}: {mean*100:.{precision}f} ± {std*100:.{precision}f}%")
        else:
            lines.append(f"{key}: {mean:.{precision}f} ± {std:.{precision}f}")
    return "\n".join(lines)
