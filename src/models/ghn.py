"""Graph Hopfield Network model for node classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple

from ..layers.graph_hopfield import GraphHopfieldLayer, GraphHopfieldBlock


class GraphHopfieldNetwork(nn.Module):
    """
    Graph Hopfield Network for node classification.
    
    Architecture:
        Input -> [Optional Encoder] -> GraphHopfieldBlock(s) -> Classifier -> Output
    
    This model tests whether associative memory retrieval combined with
    graph structure improves robustness under feature/edge corruption.
    """
    
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_patterns: int = 64,
        beta: float = 1.0,
        lambda_graph: float = 0.1,
        num_iterations: int = 1,
        num_layers: int = 1,
        alpha: float = 0.5,
        dropout: float = 0.5,
        use_encoder: bool = True,
        use_layer_norm: bool = True,
        normalize_memory_keys: bool = False,
        normalize_memory_queries: bool = False,
        tie_keys_values: bool = False,
        learnable_beta: bool = True,
        use_spectral_norm_constraint: bool = True,
        norm_mode: str = "per_layer",
        use_query_proj: bool = True,
        num_heads: int = 1,
    ):
        """
        Initialize the Graph Hopfield Network.

        Args:
            in_dim: Input feature dimension
            hidden_dim: Hidden dimension (and memory pattern dimension)
            out_dim: Number of output classes
            num_patterns: Number of memory patterns (K)
            beta: Inverse temperature for retrieval
            lambda_graph: Graph Laplacian regularization weight
            num_iterations: Hopfield iterations per layer
            num_layers: Number of GHN layers
            alpha: Update damping coefficient
            dropout: Dropout rate
            use_encoder: Use input encoder MLP
            use_layer_norm: Apply LayerNorm after iterations
            normalize_memory_keys: Normalize memory keys for stability
            normalize_memory_queries: Normalize queries for stability
            tie_keys_values: If True, use same matrix for keys and values (default: False)
            learnable_beta: If True, make beta learnable (default: True)
            use_spectral_norm_constraint: Constrain beta for convexity (default: True)
            norm_mode: When to apply LayerNorm - "none", "per_layer", "per_iteration"
            use_query_proj: If True, add a learnable query projection (default: True)
            num_heads: Number of attention heads for multi-head memory (default: 1)
        """
        super().__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers

        # Input encoder
        if use_encoder:
            self.encoder = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            ghn_in_dim = hidden_dim
        else:
            self.encoder = nn.Identity()
            ghn_in_dim = in_dim

        # Graph Hopfield layers
        self.ghn_layers = nn.ModuleList()
        for i in range(num_layers):
            layer_in_dim = ghn_in_dim if i == 0 else hidden_dim
            self.ghn_layers.append(
                GraphHopfieldLayer(
                    in_dim=layer_in_dim,
                    out_dim=hidden_dim,
                    num_patterns=num_patterns,
                    beta=beta,
                    lambda_graph=lambda_graph,
                    num_iterations=num_iterations,
                    alpha=alpha,
                    dropout=dropout,
                    use_layer_norm=use_layer_norm,
                    normalize_memory_keys=normalize_memory_keys,
                    normalize_memory_queries=normalize_memory_queries,
                    tie_keys_values=tie_keys_values,
                    learnable_beta=learnable_beta,
                    use_spectral_norm_constraint=use_spectral_norm_constraint,
                    norm_mode=norm_mode,
                    use_query_proj=use_query_proj,
                    num_heads=num_heads,
                )
            )

        # Output classifier
        self.classifier = nn.Linear(hidden_dim, out_dim)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        return_energy: bool = False,
        return_attention: bool = False,
    ) -> Tuple[Tensor, Optional[dict]]:
        """
        Forward pass through the network.

        Args:
            x: Node features [N, in_dim]
            edge_index: Graph edges [2, E]
            return_energy: Return energy values
            return_attention: Return attention weights
            
        Returns:
            logits: Classification logits [N, out_dim]
            info: Optional dict with energy/attention info
        """
        all_energies = []
        all_attentions = []
        
        # Encode input
        x = self.encoder(x)
        
        # Pass through GHN layers
        for layer in self.ghn_layers:
            x, info = layer(
                x, edge_index,
                return_energy=return_energy,
                return_attention=return_attention,
            )
            if info:
                if "energies" in info:
                    all_energies.extend(info["energies"])
                if "attentions" in info:
                    all_attentions.extend(info["attentions"])
        
        # Classify
        logits = self.classifier(x)
        
        # Collect info
        output_info = None
        if return_energy or return_attention:
            output_info = {}
            if return_energy:
                output_info["energies"] = all_energies
            if return_attention:
                output_info["attentions"] = all_attentions
        
        return logits, output_info
    
    def get_num_params(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class GraphHopfieldNetworkMinimal(nn.Module):
    """
    Minimal Graph Hopfield Network variant.
    
    Architecture: Input MLP -> Single GHN Block (T iterations) -> Classifier
    
    This is the simplest version for quick experiments.
    """
    
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_patterns: int = 32,
        beta: float = 1.0,
        lambda_graph: float = 0.1,
        num_iterations: int = 2,
        alpha: float = 0.5,
        dropout: float = 0.5,
        use_layer_norm: bool = True,
        normalize_memory_keys: bool = False,
        normalize_memory_queries: bool = False,
        tie_keys_values: bool = False,
        learnable_beta: bool = True,
        use_spectral_norm_constraint: bool = True,
        norm_mode: str = "per_layer",
        use_query_proj: bool = True,
        num_heads: int = 1,
    ):
        super().__init__()

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Single GHN layer with multiple iterations
        self.ghn = GraphHopfieldLayer(
            in_dim=hidden_dim,
            out_dim=hidden_dim,
            num_patterns=num_patterns,
            beta=beta,
            lambda_graph=lambda_graph,
            num_iterations=num_iterations,
            alpha=alpha,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
            normalize_memory_keys=normalize_memory_keys,
            normalize_memory_queries=normalize_memory_queries,
            tie_keys_values=tie_keys_values,
            learnable_beta=learnable_beta,
            use_spectral_norm_constraint=use_spectral_norm_constraint,
            norm_mode=norm_mode,
            use_query_proj=use_query_proj,
            num_heads=num_heads,
        )

        # Classifier
        self.classifier = nn.Linear(hidden_dim, out_dim)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        return_energy: bool = False,
        return_attention: bool = False,
    ) -> Tuple[Tensor, Optional[dict]]:
        x = self.input_proj(x)
        x, info = self.ghn(
            x, edge_index,
            return_energy=return_energy,
            return_attention=return_attention,
        )
        logits = self.classifier(x)
        return logits, info

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
