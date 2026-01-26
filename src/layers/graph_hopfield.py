"""Graph Hopfield Layer - Core implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.utils import degree, add_self_loops
from typing import Optional, Tuple

from .memory_bank import MemoryBank


class GraphHopfieldLayer(nn.Module):
    """
    Graph Hopfield Layer combining associative memory retrieval with graph structure.
    
    This layer implements the energy function:
        E_GH(X) = Σᵥ [-β⁻¹ lse(β, M^T Xᵥ) + ½||Xᵥ||²] + λ Σ_(u,v)∈E ||Xᵤ - Xᵥ||²
    
    The update dynamics combine:
        1. Hopfield retrieval: M^T · softmax(β · M · xᵥ)
        2. Graph Laplacian smoothing: -λ · L · X
    
    Attributes:
        in_dim (int): Input feature dimension
        out_dim (int): Output feature dimension
        num_patterns (int): Number of memory patterns K
        beta (float): Inverse temperature for retrieval sharpness
        lambda_graph (float): Weight for graph Laplacian regularization
        num_iterations (int): Number of update iterations T
        use_layer_norm (bool): Whether to apply LayerNorm after updates
        use_residual (bool): Whether to use residual connections
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_patterns: int,
        beta: float = 1.0,
        lambda_graph: float = 0.1,
        num_iterations: int = 1,
        alpha: float = 0.5,
        use_layer_norm: bool = True,
        use_residual: bool = True,
        dropout: float = 0.0,
        normalize_laplacian: bool = True,
        normalize_memory_keys: bool = False,
        normalize_memory_queries: bool = False,
    ):
        """
        Initialize the Graph Hopfield Layer.
        
        Args:
            in_dim: Input feature dimension
            out_dim: Output feature dimension (also memory pattern dimension)
            num_patterns: Number of memory prototype patterns (K)
            beta: Inverse temperature controlling retrieval sharpness
            lambda_graph: Weight for graph Laplacian regularization
            num_iterations: Number of fixed-point iterations (T)
            alpha: Damping/mixing coefficient for update stability
            use_layer_norm: Apply LayerNorm after each iteration
            use_residual: Use residual connection in updates
            dropout: Dropout rate
            normalize_laplacian: Use symmetric normalized Laplacian
            normalize_memory_keys: Normalize memory keys for numerical stability
            normalize_memory_queries: Normalize queries for numerical stability
        """
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_patterns = num_patterns
        self.beta = beta
        self.lambda_graph = lambda_graph
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.use_layer_norm = use_layer_norm
        self.use_residual = use_residual
        self.normalize_laplacian = normalize_laplacian
        
        # Input projection (if dimensions differ)
        if in_dim != out_dim:
            self.proj_in = nn.Linear(in_dim, out_dim)
        else:
            self.proj_in = nn.Identity()
        
        # Memory bank for Hopfield retrieval
        self.memory = MemoryBank(
            num_patterns=num_patterns,
            pattern_dim=out_dim,
            tie_keys_values=True,
            normalize_keys=normalize_memory_keys,
            normalize_queries=normalize_memory_queries,
        )
        
        # Optional layer normalization
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(out_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def _compute_laplacian_term(
        self,
        x: Tensor,
        edge_index: Tensor,
        num_nodes: int,
    ) -> Tensor:
        """
        Compute L @ X where L is the (normalized) graph Laplacian.
        
        For normalized Laplacian: L_sym = I - D^{-1/2} A D^{-1/2}
        For unnormalized: L = D - A
        
        Args:
            x: Node features [N, d]
            edge_index: Graph connectivity [2, E]
            num_nodes: Number of nodes N
            
        Returns:
            Laplacian term L @ X [N, d]
        """
        row, col = edge_index
        
        # Compute node degrees
        deg = degree(col, num_nodes, dtype=x.dtype)
        
        if self.normalize_laplacian:
            # Symmetric normalized Laplacian: L_sym = I - D^{-1/2} A D^{-1/2}
            # L_sym @ X = X - D^{-1/2} A D^{-1/2} X
            
            # Compute D^{-1/2}
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            
            # Normalize features: D^{-1/2} X
            norm_x = deg_inv_sqrt.unsqueeze(-1) * x
            
            # Compute A @ (D^{-1/2} X) via sparse aggregation
            # For each node v, sum normalized features of neighbors
            out = torch.zeros_like(x)
            out.index_add_(0, row, norm_x[col])
            
            # Apply D^{-1/2} again
            agg = deg_inv_sqrt.unsqueeze(-1) * out
            
            # L_sym @ X = X - agg
            return x - agg
        else:
            # Unnormalized Laplacian: L = D - A
            # L @ X = D @ X - A @ X
            
            # D @ X
            dx = deg.unsqueeze(-1) * x
            
            # A @ X via sparse aggregation
            ax = torch.zeros_like(x)
            ax.index_add_(0, row, x[col])
            
            return dx - ax
    
    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        return_energy: bool = False,
        return_attention: bool = False,
    ) -> Tuple[Tensor, Optional[dict]]:
        """
        Forward pass through the Graph Hopfield Layer.
        
        Args:
            x: Input node features [N, in_dim]
            edge_index: Graph edge indices [2, E]
            return_energy: If True, compute and return energy values
            return_attention: If True, return attention weights
            
        Returns:
            x: Output node features [N, out_dim]
            info: Optional dict with energy/attention info
        """
        num_nodes = x.size(0)
        info = {}
        
        # Project input to output dimension
        x = self.proj_in(x)
        
        # Store input for residual
        x_input = x
        
        energies = []
        attentions = []
        
        # Iterative update
        for t in range(self.num_iterations):
            # 1. Hopfield retrieval: R(X) = M^T @ softmax(β M X)
            retrieved, attn = self.memory.retrieve(
                x, beta=self.beta, return_attention=return_attention
            )
            
            if return_attention and attn is not None:
                attentions.append(attn)
            
            # 2. Graph Laplacian term: L @ X
            laplacian_term = self._compute_laplacian_term(x, edge_index, num_nodes)
            
            # 3. Combined update with damping
            # X^{t+1} = (1-α)X^t + α[R(X^t) - 2λ·L·X^t]
            # Note: Factor of 2 comes from gradient: ∇E = ... + 2λ(LX)_v
            # IMPORTANT: With this fix, lambda_graph now corresponds to λ in the math.
            # Previously (without factor of 2), lambda_graph=1.0 meant λ=0.5 effectively.
            # To match previous behavior, use lambda_graph=0.5 instead of 1.0.
            laplacian_coeff = 2.0 * self.lambda_graph
            if self.use_residual:
                x_new = (1 - self.alpha) * x + self.alpha * (
                    retrieved - laplacian_coeff * laplacian_term
                )
            else:
                x_new = retrieved - laplacian_coeff * laplacian_term
            
            # 4. Apply layer norm (for stability)
            if self.use_layer_norm:
                x_new = self.layer_norm(x_new)
            
            x = x_new
            
            # Compute energy if requested
            if return_energy:
                energy = self._compute_total_energy(x, edge_index, num_nodes)
                energies.append(energy.item())
        
        # Apply dropout after all iterations (not during, to preserve energy descent)
        x = self.dropout(x)
        
        # Collect info
        if return_energy:
            info["energies"] = energies
        if return_attention and attentions:
            info["attentions"] = attentions
        
        return x, info if (return_energy or return_attention) else None
    
    def _compute_total_energy(
        self,
        x: Tensor,
        edge_index: Tensor,
        num_nodes: int,
    ) -> Tensor:
        """
        Compute the total Graph Hopfield energy.
        
        E_GH(X) = Σᵥ [-β⁻¹ lse(β, M^T Xᵥ) + ½||Xᵥ||²] + λ tr(X^T L X)
        
        Args:
            x: Node features [N, d]
            edge_index: Graph edges [2, E]
            num_nodes: Number of nodes
            
        Returns:
            Total energy (scalar)
        """
        # Hopfield energy per node
        hopfield_energy = self.memory.compute_energy(x, beta=self.beta)
        
        # Graph Laplacian energy: λ tr(X^T L X) = λ Σ_(u,v)∈E ||xᵤ - xᵥ||²
        row, col = edge_index
        diff = x[row] - x[col]
        laplacian_energy = self.lambda_graph * (diff ** 2).sum()
        
        return hopfield_energy + laplacian_energy
    
    def extra_repr(self) -> str:
        return (
            f"in_dim={self.in_dim}, out_dim={self.out_dim}, "
            f"num_patterns={self.num_patterns}, beta={self.beta}, "
            f"lambda_graph={self.lambda_graph}, num_iterations={self.num_iterations}"
        )


class GraphHopfieldBlock(nn.Module):
    """
    A complete Graph Hopfield block with optional pre/post MLPs.
    
    Structure: [Optional Input MLP] -> GraphHopfieldLayer -> [Optional Output MLP]
    """
    
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_patterns: int,
        beta: float = 1.0,
        lambda_graph: float = 0.1,
        num_iterations: int = 1,
        alpha: float = 0.5,
        dropout: float = 0.0,
        use_input_mlp: bool = True,
        use_output_mlp: bool = False,
        use_layer_norm: bool = True,
        normalize_memory_keys: bool = False,
        normalize_memory_queries: bool = False,
    ):
        """
        Initialize the Graph Hopfield Block.
        
        Args:
            in_dim: Input dimension
            hidden_dim: Hidden/memory dimension
            out_dim: Output dimension
            num_patterns: Number of memory patterns
            beta: Inverse temperature
            lambda_graph: Graph smoothness weight
            num_iterations: Number of Hopfield iterations
            alpha: Update damping coefficient
            dropout: Dropout rate
            use_input_mlp: Use MLP before Hopfield layer
            use_output_mlp: Use MLP after Hopfield layer
            use_layer_norm: Apply LayerNorm after iterations
            normalize_memory_keys: Normalize memory keys for stability
            normalize_memory_queries: Normalize queries for stability
        """
        super().__init__()
        
        self.use_input_mlp = use_input_mlp
        self.use_output_mlp = use_output_mlp
        
        # Input MLP
        if use_input_mlp:
            self.input_mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            hopfield_in_dim = hidden_dim
        else:
            self.input_mlp = nn.Identity()
            hopfield_in_dim = in_dim
        
        # Graph Hopfield Layer
        self.hopfield = GraphHopfieldLayer(
            in_dim=hopfield_in_dim,
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
        )
        
        # Output MLP
        if use_output_mlp:
            self.output_mlp = nn.Sequential(
                nn.Linear(hidden_dim, out_dim),
            )
        else:
            self.output_mlp = nn.Identity()
            # Ensure dimensions match
            if hidden_dim != out_dim:
                self.output_mlp = nn.Linear(hidden_dim, out_dim)
    
    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        return_energy: bool = False,
        return_attention: bool = False,
    ) -> Tuple[Tensor, Optional[dict]]:
        """Forward pass through the block."""
        x = self.input_mlp(x)
        x, info = self.hopfield(
            x, edge_index,
            return_energy=return_energy,
            return_attention=return_attention,
        )
        x = self.output_mlp(x)
        return x, info
