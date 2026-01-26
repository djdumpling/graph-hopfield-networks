"""Baseline GNN models for comparison with Graph Hopfield Networks."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from typing import Optional, Tuple


class GCN(nn.Module):
    """
    Graph Convolutional Network (Kipf & Welling, 2017).
    
    Standard 2-layer GCN baseline for node classification.
    """
    
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        dropout: float = 0.5,
    ):
        """
        Initialize GCN.
        
        Args:
            in_dim: Input feature dimension
            hidden_dim: Hidden dimension
            out_dim: Number of output classes
            num_layers: Number of GCN layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(GCNConv(in_dim, hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Output layer
        if num_layers > 1:
            self.convs.append(GCNConv(hidden_dim, out_dim))
        else:
            # Single layer: adjust first layer output
            self.convs[0] = GCNConv(in_dim, out_dim)
    
    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        **kwargs,
    ) -> Tuple[Tensor, None]:
        """Forward pass."""
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        return x, None
    
    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class GAT(nn.Module):
    """
    Graph Attention Network (Veličković et al., 2018).
    
    Multi-head attention-based GNN baseline.
    """
    
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.6,
        attention_dropout: float = 0.6,
    ):
        """
        Initialize GAT.
        
        Args:
            in_dim: Input feature dimension
            hidden_dim: Hidden dimension (per head)
            out_dim: Number of output classes
            num_layers: Number of GAT layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            attention_dropout: Attention coefficient dropout
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(
            GATConv(
                in_dim,
                hidden_dim,
                heads=num_heads,
                dropout=attention_dropout,
                concat=True,
            )
        )
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(
                    hidden_dim * num_heads,
                    hidden_dim,
                    heads=num_heads,
                    dropout=attention_dropout,
                    concat=True,
                )
            )
        
        # Output layer (no concatenation, single head or average)
        if num_layers > 1:
            self.convs.append(
                GATConv(
                    hidden_dim * num_heads,
                    out_dim,
                    heads=1,
                    dropout=attention_dropout,
                    concat=False,
                )
            )
        else:
            self.convs[0] = GATConv(
                in_dim,
                out_dim,
                heads=1,
                dropout=attention_dropout,
                concat=False,
            )
    
    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        **kwargs,
    ) -> Tuple[Tensor, None]:
        """Forward pass."""
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        return x, None
    
    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class GraphSAGE(nn.Module):
    """
    GraphSAGE (Hamilton et al., 2017).
    
    Sampling and aggregating GNN baseline with mean aggregation.
    """
    
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        aggr: str = "mean",
    ):
        """
        Initialize GraphSAGE.
        
        Args:
            in_dim: Input feature dimension
            hidden_dim: Hidden dimension
            out_dim: Number of output classes
            num_layers: Number of SAGE layers
            dropout: Dropout rate
            aggr: Aggregation method ("mean", "max", "sum")
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(SAGEConv(in_dim, hidden_dim, aggr=aggr))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr=aggr))
        
        # Output layer
        if num_layers > 1:
            self.convs.append(SAGEConv(hidden_dim, out_dim, aggr=aggr))
        else:
            self.convs[0] = SAGEConv(in_dim, out_dim, aggr=aggr)
    
    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        **kwargs,
    ) -> Tuple[Tensor, None]:
        """Forward pass."""
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        return x, None
    
    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MLP(nn.Module):
    """
    Simple MLP baseline (ignores graph structure).
    
    Useful ablation to show that graph structure matters.
    """
    
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()
        
        layers = []
        
        # First layer
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, out_dim))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        **kwargs,
    ) -> Tuple[Tensor, None]:
        """Forward pass (ignores edge_index)."""
        return self.mlp(x), None
    
    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_baseline(
    model_name: str,
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create baseline models.
    
    Args:
        model_name: One of "gcn", "gat", "graphsage", "mlp"
        in_dim: Input dimension
        hidden_dim: Hidden dimension
        out_dim: Output dimension
        **kwargs: Additional model-specific arguments
        
    Returns:
        Initialized model
    """
    models = {
        "gcn": GCN,
        "gat": GAT,
        "graphsage": GraphSAGE,
        "mlp": MLP,
    }
    
    if model_name.lower() not in models:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(models.keys())}")
    
    return models[model_name.lower()](in_dim, hidden_dim, out_dim, **kwargs)
