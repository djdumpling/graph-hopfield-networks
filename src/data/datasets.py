"""Dataset loading utilities for Graph Hopfield Networks experiments."""

import torch
from torch import Tensor
from torch_geometric.datasets import Planetoid, Amazon, Coauthor
from torch_geometric.data import Data
from torch_geometric.transforms import NormalizeFeatures
from typing import Tuple, Optional, Dict, Any
import os


def get_planetoid_dataset(
    name: str,
    root: str = "./data",
    normalize_features: bool = True,
) -> Tuple[Data, Dict[str, Any]]:
    """
    Load a Planetoid dataset (Cora, Citeseer, Pubmed).
    
    Args:
        name: Dataset name ("cora", "citeseer", or "pubmed")
        root: Root directory for data storage
        normalize_features: Whether to normalize node features
        
    Returns:
        data: PyG Data object with train/val/test masks
        info: Dict with dataset statistics
    """
    name = name.lower()
    if name not in ["cora", "citeseer", "pubmed"]:
        raise ValueError(f"Unknown Planetoid dataset: {name}")
    
    transform = NormalizeFeatures() if normalize_features else None
    
    dataset = Planetoid(
        root=root,
        name=name.capitalize() if name != "pubmed" else "PubMed",
        transform=transform,
    )
    
    data = dataset[0]
    
    info = {
        "name": name,
        "num_nodes": data.num_nodes,
        "num_edges": data.num_edges,
        "num_features": data.num_features,
        "num_classes": dataset.num_classes,
        "num_train": data.train_mask.sum().item(),
        "num_val": data.val_mask.sum().item(),
        "num_test": data.test_mask.sum().item(),
    }
    
    return data, info


def get_amazon_dataset(
    name: str,
    root: str = "./data",
    normalize_features: bool = True,
) -> Tuple[Data, Dict[str, Any]]:
    """
    Load an Amazon dataset (Computers, Photo).
    
    Args:
        name: Dataset name ("computers" or "photo")
        root: Root directory for data storage
        normalize_features: Whether to normalize node features
        
    Returns:
        data: PyG Data object
        info: Dict with dataset statistics
    """
    name = name.lower()
    if name not in ["computers", "photo"]:
        raise ValueError(f"Unknown Amazon dataset: {name}")
    
    transform = NormalizeFeatures() if normalize_features else None
    
    dataset = Amazon(
        root=root,
        name=name.capitalize(),
        transform=transform,
    )
    
    data = dataset[0]
    
    # Create train/val/test splits (no default splits for Amazon)
    data = create_random_splits(data, train_ratio=0.6, val_ratio=0.2)
    
    info = {
        "name": f"amazon_{name}",
        "num_nodes": data.num_nodes,
        "num_edges": data.num_edges,
        "num_features": data.num_features,
        "num_classes": dataset.num_classes,
        "num_train": data.train_mask.sum().item(),
        "num_val": data.val_mask.sum().item(),
        "num_test": data.test_mask.sum().item(),
    }
    
    return data, info


def create_random_splits(
    data: Data,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Data:
    """
    Create random train/val/test splits for a dataset.
    
    Args:
        data: PyG Data object
        train_ratio: Fraction of nodes for training
        val_ratio: Fraction of nodes for validation
        seed: Random seed
        
    Returns:
        Data object with train_mask, val_mask, test_mask
    """
    torch.manual_seed(seed)
    
    num_nodes = data.num_nodes
    indices = torch.randperm(num_nodes)
    
    num_train = int(num_nodes * train_ratio)
    num_val = int(num_nodes * val_ratio)
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[indices[:num_train]] = True
    val_mask[indices[num_train:num_train + num_val]] = True
    test_mask[indices[num_train + num_val:]] = True
    
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    
    return data


def load_dataset(
    name: str,
    root: str = "./data",
    normalize_features: bool = True,
) -> Tuple[Data, Dict[str, Any]]:
    """
    Load a dataset by name.
    
    Supported datasets:
        - Planetoid: "cora", "citeseer", "pubmed"
        - Amazon: "amazon_computers", "amazon_photo"
    
    Args:
        name: Dataset name
        root: Root directory
        normalize_features: Whether to normalize features
        
    Returns:
        data: PyG Data object
        info: Dataset statistics
    """
    name = name.lower()
    
    if name in ["cora", "citeseer", "pubmed"]:
        return get_planetoid_dataset(name, root, normalize_features)
    elif name.startswith("amazon_"):
        amazon_name = name.replace("amazon_", "")
        return get_amazon_dataset(amazon_name, root, normalize_features)
    else:
        raise ValueError(
            f"Unknown dataset: {name}. "
            f"Supported: cora, citeseer, pubmed, amazon_computers, amazon_photo"
        )


def print_dataset_info(info: Dict[str, Any]) -> None:
    """Print dataset statistics."""
    print(f"Dataset: {info['name']}")
    print(f"  Nodes: {info['num_nodes']:,}")
    print(f"  Edges: {info['num_edges']:,}")
    print(f"  Features: {info['num_features']}")
    print(f"  Classes: {info['num_classes']}")
    print(f"  Train/Val/Test: {info['num_train']}/{info['num_val']}/{info['num_test']}")
