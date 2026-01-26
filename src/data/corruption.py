"""Corruption utilities for robustness benchmarks."""

import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops
from dataclasses import dataclass
from typing import Optional, Tuple, Literal
import copy


@dataclass
class CorruptionConfig:
    """Configuration for data corruption experiments."""
    
    # Feature corruption
    feature_noise_std: float = 0.0  # Gaussian noise standard deviation
    feature_mask_ratio: float = 0.0  # Fraction of features to mask (set to 0)
    
    # Edge corruption
    edge_drop_ratio: float = 0.0  # Fraction of edges to drop
    edge_add_ratio: float = 0.0  # Fraction of edges to add (relative to existing)
    
    # Label corruption (for training labels only)
    label_flip_ratio: float = 0.0  # Fraction of training labels to flip
    
    # Random seed
    seed: Optional[int] = None


def add_feature_noise(
    x: Tensor,
    noise_std: float,
    seed: Optional[int] = None,
) -> Tensor:
    """
    Add Gaussian noise to node features.
    
    Args:
        x: Node features [N, d]
        noise_std: Standard deviation of Gaussian noise
        seed: Random seed
        
    Returns:
        Noisy features [N, d]
    """
    if noise_std <= 0:
        return x
    
    if seed is not None:
        torch.manual_seed(seed)
    
    noise = torch.randn_like(x) * noise_std
    return x + noise


def mask_features(
    x: Tensor,
    mask_ratio: float,
    mask_value: float = 0.0,
    seed: Optional[int] = None,
) -> Tensor:
    """
    Randomly mask (zero out) a fraction of features.
    
    Args:
        x: Node features [N, d]
        mask_ratio: Fraction of features to mask (per node)
        mask_value: Value to use for masked features
        seed: Random seed
        
    Returns:
        Masked features [N, d]
    """
    if mask_ratio <= 0:
        return x
    
    if seed is not None:
        torch.manual_seed(seed)
    
    x = x.clone()
    mask = torch.rand_like(x) < mask_ratio
    x[mask] = mask_value
    
    return x


def corrupt_edges(
    edge_index: Tensor,
    num_nodes: int,
    drop_ratio: float = 0.0,
    add_ratio: float = 0.0,
    seed: Optional[int] = None,
) -> Tensor:
    """
    Corrupt edges by dropping and/or adding random edges.
    
    Args:
        edge_index: Edge indices [2, E]
        num_nodes: Total number of nodes
        drop_ratio: Fraction of edges to drop
        add_ratio: Fraction of edges to add (relative to original)
        seed: Random seed
        
    Returns:
        Corrupted edge indices [2, E']
    """
    if drop_ratio <= 0 and add_ratio <= 0:
        return edge_index
    
    if seed is not None:
        torch.manual_seed(seed)
    
    num_edges = edge_index.size(1)
    
    # Drop edges
    if drop_ratio > 0:
        keep_mask = torch.rand(num_edges) > drop_ratio
        edge_index = edge_index[:, keep_mask]
    
    # Add random edges
    if add_ratio > 0:
        num_add = int(num_edges * add_ratio)
        if num_add > 0:
            # Generate random edges
            new_edges = torch.randint(0, num_nodes, (2, num_add), dtype=edge_index.dtype)
            edge_index = torch.cat([edge_index, new_edges], dim=1)
            
            # Remove duplicates and self-loops
            edge_index = to_undirected(edge_index, num_nodes=num_nodes)
            edge_index, _ = remove_self_loops(edge_index)
    
    return edge_index


def corrupt_labels(
    y: Tensor,
    train_mask: Tensor,
    num_classes: int,
    flip_ratio: float,
    seed: Optional[int] = None,
) -> Tensor:
    """
    Corrupt training labels by randomly flipping them.
    
    Only affects labels in the training set.
    
    Args:
        y: Node labels [N]
        train_mask: Training mask [N]
        num_classes: Number of classes
        flip_ratio: Fraction of training labels to flip
        seed: Random seed
        
    Returns:
        Corrupted labels [N]
    """
    if flip_ratio <= 0:
        return y
    
    if seed is not None:
        torch.manual_seed(seed)
    
    y = y.clone()
    train_indices = train_mask.nonzero(as_tuple=True)[0]
    num_train = len(train_indices)
    num_flip = int(num_train * flip_ratio)
    
    if num_flip > 0:
        # Select random training nodes to flip
        flip_indices = train_indices[torch.randperm(num_train)[:num_flip]]
        
        # Generate new random labels (different from original)
        for idx in flip_indices:
            old_label = y[idx].item()
            new_label = torch.randint(0, num_classes - 1, (1,)).item()
            if new_label >= old_label:
                new_label += 1
            y[idx] = new_label
    
    return y


def apply_corruption(
    data: Data,
    config: CorruptionConfig,
) -> Data:
    """
    Apply corruption to a PyG Data object.
    
    Args:
        data: Original PyG Data object
        config: Corruption configuration
        
    Returns:
        Corrupted Data object (copy)
    """
    # Create a copy
    corrupted = Data(
        x=data.x.clone(),
        edge_index=data.edge_index.clone(),
        y=data.y.clone(),
        train_mask=data.train_mask.clone() if hasattr(data, 'train_mask') else None,
        val_mask=data.val_mask.clone() if hasattr(data, 'val_mask') else None,
        test_mask=data.test_mask.clone() if hasattr(data, 'test_mask') else None,
    )
    
    # Apply feature corruption
    if config.feature_noise_std > 0:
        corrupted.x = add_feature_noise(
            corrupted.x, config.feature_noise_std, config.seed
        )
    
    if config.feature_mask_ratio > 0:
        corrupted.x = mask_features(
            corrupted.x, config.feature_mask_ratio, seed=config.seed
        )
    
    # Apply edge corruption
    if config.edge_drop_ratio > 0 or config.edge_add_ratio > 0:
        corrupted.edge_index = corrupt_edges(
            corrupted.edge_index,
            corrupted.num_nodes,
            drop_ratio=config.edge_drop_ratio,
            add_ratio=config.edge_add_ratio,
            seed=config.seed,
        )
    
    # Apply label corruption
    if config.label_flip_ratio > 0 and corrupted.train_mask is not None:
        num_classes = int(corrupted.y.max().item()) + 1
        corrupted.y = corrupt_labels(
            corrupted.y,
            corrupted.train_mask,
            num_classes,
            config.label_flip_ratio,
            seed=config.seed,
        )
    
    return corrupted


def get_corruption_levels(
    corruption_type: Literal["feature_noise", "feature_mask", "edge_drop", "edge_add", "label_flip"],
    levels: Optional[list] = None,
) -> list[CorruptionConfig]:
    """
    Generate a list of corruption configs at different levels.
    
    Args:
        corruption_type: Type of corruption
        levels: List of corruption levels (default: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        
    Returns:
        List of CorruptionConfig objects
    """
    if levels is None:
        levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    
    configs = []
    for level in levels:
        config = CorruptionConfig()
        
        if corruption_type == "feature_noise":
            config.feature_noise_std = level
        elif corruption_type == "feature_mask":
            config.feature_mask_ratio = level
        elif corruption_type == "edge_drop":
            config.edge_drop_ratio = level
        elif corruption_type == "edge_add":
            config.edge_add_ratio = level
        elif corruption_type == "label_flip":
            config.label_flip_ratio = level
        else:
            raise ValueError(f"Unknown corruption type: {corruption_type}")
        
        configs.append(config)
    
    return configs
