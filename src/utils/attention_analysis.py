"""Utilities for analyzing attention weights and detecting pattern collapse."""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def compute_attention_entropy(attention: torch.Tensor) -> torch.Tensor:
    """
    Compute entropy of attention distribution for each node.
    
    Args:
        attention: Attention weights [N, K] or [B, N, K]
    
    Returns:
        entropy: Entropy per node [N] or [B, N]
    """
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    attention_safe = attention + eps
    
    # Compute entropy: -sum(p * log(p))
    entropy = -(attention_safe * torch.log(attention_safe)).sum(dim=-1)
    
    return entropy


def compute_pattern_usage(attention: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Analyze which memory patterns are being used.
    
    Args:
        attention: Attention weights [N, K] or [B, N, K]
    
    Returns:
        Dictionary with:
            - pattern_usage: Average attention per pattern [K]
            - pattern_std: Std of attention per pattern [K]
            - active_patterns: Number of patterns with usage > threshold
    """
    # Average attention over nodes (and batch if present)
    if attention.dim() == 3:
        pattern_usage = attention.mean(dim=(0, 1))  # [K]
        pattern_std = attention.std(dim=(0, 1))  # [K]
    else:
        pattern_usage = attention.mean(dim=0)  # [K]
        pattern_std = attention.std(dim=0)  # [K]
    
    # Count active patterns (usage > 1/K, i.e., above uniform)
    threshold = 1.0 / attention.shape[-1]
    active_patterns = (pattern_usage > threshold).sum().item()
    
    return {
        "pattern_usage": pattern_usage,
        "pattern_std": pattern_std,
        "active_patterns": active_patterns,
        "total_patterns": attention.shape[-1],
    }


def detect_pattern_collapse(
    memory_keys: torch.Tensor,
    attention: Optional[torch.Tensor] = None,
    similarity_threshold: float = 0.95,
) -> Dict[str, any]:
    """
    Detect if memory patterns have collapsed (become too similar).
    
    Args:
        memory_keys: Memory pattern vectors [K, d]
        attention: Optional attention weights [N, K] for usage analysis
        similarity_threshold: Cosine similarity threshold for collapse detection
    
    Returns:
        Dictionary with collapse metrics
    """
    K, d = memory_keys.shape
    
    # Normalize patterns for cosine similarity
    keys_norm = torch.nn.functional.normalize(memory_keys, p=2, dim=-1)
    
    # Compute pairwise cosine similarities
    similarity_matrix = torch.mm(keys_norm, keys_norm.t())  # [K, K]
    
    # Remove diagonal (self-similarity)
    mask = ~torch.eye(K, dtype=torch.bool, device=memory_keys.device)
    off_diagonal_similarities = similarity_matrix[mask]
    
    # Find highly similar pairs
    high_similarity_pairs = (off_diagonal_similarities > similarity_threshold).sum().item()
    max_similarity = off_diagonal_similarities.max().item()
    mean_similarity = off_diagonal_similarities.mean().item()
    
    # Compute pattern diversity (average pairwise distance)
    pattern_diversity = (1.0 - mean_similarity)  # Higher is more diverse
    
    result = {
        "max_similarity": max_similarity,
        "mean_similarity": mean_similarity,
        "pattern_diversity": pattern_diversity,
        "high_similarity_pairs": high_similarity_pairs,
        "total_pairs": K * (K - 1) // 2,
        "is_collapsed": max_similarity > similarity_threshold,
    }
    
    # Add attention-based analysis if provided
    if attention is not None:
        usage_stats = compute_pattern_usage(attention)
        result["active_patterns"] = usage_stats["active_patterns"]
        result["total_patterns"] = usage_stats["total_patterns"]
        
        # Check if only few patterns are used
        usage_threshold = 0.1  # Pattern must have >10% average usage
        highly_used = (usage_stats["pattern_usage"] > usage_threshold).sum().item()
        result["highly_used_patterns"] = highly_used
        result["usage_imbalance"] = usage_stats["pattern_usage"].std().item()
    
    return result


def visualize_attention_heatmap(
    attention: torch.Tensor,
    save_path: Optional[Path] = None,
    title: str = "Attention Weights",
    max_nodes: int = 100,
):
    """
    Visualize attention weights as a heatmap.
    
    Args:
        attention: Attention weights [N, K]
        save_path: Optional path to save figure
        title: Plot title
        max_nodes: Maximum number of nodes to visualize (for large graphs)
    """
    attn_np = attention.detach().cpu().numpy()
    N, K = attn_np.shape
    
    # Sample nodes if too many
    if N > max_nodes:
        indices = np.linspace(0, N - 1, max_nodes, dtype=int)
        attn_np = attn_np[indices]
        N = max_nodes
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        attn_np,
        cmap="viridis",
        cbar=True,
        xticklabels=[f"P{i}" for i in range(K)],
        yticklabels=False if N > 50 else True,
    )
    plt.xlabel("Memory Pattern")
    plt.ylabel("Node")
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def visualize_pattern_similarity(
    memory_keys: torch.Tensor,
    save_path: Optional[Path] = None,
    title: str = "Memory Pattern Similarity",
):
    """
    Visualize similarity matrix between memory patterns.
    
    Args:
        memory_keys: Memory pattern vectors [K, d]
        save_path: Optional path to save figure
        title: Plot title
    """
    keys_norm = torch.nn.functional.normalize(memory_keys, p=2, dim=-1)
    similarity_matrix = torch.mm(keys_norm, keys_norm.t()).detach().cpu().numpy()
    
    K = similarity_matrix.shape[0]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        similarity_matrix,
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        cbar=True,
        xticklabels=[f"P{i}" for i in range(K)],
        yticklabels=[f"P{i}" for i in range(K)],
    )
    plt.xlabel("Memory Pattern")
    plt.ylabel("Memory Pattern")
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def visualize_attention_entropy(
    attention: torch.Tensor,
    save_path: Optional[Path] = None,
    title: str = "Attention Entropy Distribution",
):
    """
    Visualize distribution of attention entropy across nodes.
    
    Args:
        attention: Attention weights [N, K]
        save_path: Optional path to save figure
        title: Plot title
    """
    entropy = compute_attention_entropy(attention).detach().cpu().numpy()
    max_entropy = np.log(attention.shape[-1])  # Maximum entropy (uniform distribution)
    
    # Determine appropriate number of bins
    entropy_range = entropy.max() - entropy.min()
    if entropy_range < 1e-6:
        # All values are essentially the same
        num_bins = 1
    else:
        # Use fewer bins if range is small
        num_bins = min(50, max(10, int(entropy_range * 20)))
    
    plt.figure(figsize=(10, 6))
    plt.hist(entropy, bins=num_bins, alpha=0.7, edgecolor="black")
    plt.axvline(max_entropy, color="red", linestyle="--", label=f"Max entropy ({max_entropy:.2f})")
    plt.axvline(entropy.mean(), color="green", linestyle="--", label=f"Mean ({entropy.mean():.2f})")
    plt.xlabel("Attention Entropy")
    plt.ylabel("Number of Nodes")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def analyze_attention_evolution(
    attention_history: List[torch.Tensor],
    save_dir: Optional[Path] = None,
) -> Dict[str, any]:
    """
    Analyze how attention changes across iterations/layers.
    
    Args:
        attention_history: List of attention tensors [N, K] from different iterations
        save_dir: Optional directory to save visualizations
    
    Returns:
        Dictionary with evolution metrics
    """
    if len(attention_history) == 0:
        return {}
    
    # Compute metrics for each iteration
    entropies = []
    pattern_usages = []
    
    for attn in attention_history:
        entropies.append(compute_attention_entropy(attn).mean().item())
        usage_stats = compute_pattern_usage(attn)
        pattern_usages.append(usage_stats["pattern_usage"].detach().cpu().numpy())
    
    entropies = np.array(entropies)
    pattern_usages = np.stack(pattern_usages)  # [T, K]
    
    result = {
        "mean_entropy": entropies.tolist(),
        "entropy_std": entropies.std(),
        "entropy_trend": "increasing" if entropies[-1] > entropies[0] else "decreasing",
        "pattern_usage_evolution": pattern_usages.tolist(),
    }
    
    # Visualize entropy evolution
    if save_dir:
        plt.figure(figsize=(10, 6))
        plt.plot(entropies, marker="o")
        plt.xlabel("Iteration")
        plt.ylabel("Mean Attention Entropy")
        plt.title("Attention Entropy Evolution")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / "attention_entropy_evolution.png", dpi=150, bbox_inches="tight")
        plt.close()
        
        # Visualize pattern usage evolution
        plt.figure(figsize=(12, 6))
        K = pattern_usages.shape[1]
        for k in range(min(K, 10)):  # Show top 10 patterns
            plt.plot(pattern_usages[:, k], label=f"Pattern {k}", alpha=0.7)
        plt.xlabel("Iteration")
        plt.ylabel("Average Attention Weight")
        plt.title("Pattern Usage Evolution (Top 10 Patterns)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / "pattern_usage_evolution.png", dpi=150, bbox_inches="tight")
        plt.close()
    
    return result
