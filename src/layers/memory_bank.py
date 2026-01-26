"""Memory Bank module for storing learnable prototype patterns."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MemoryBank(nn.Module):
    """
    Learnable memory bank of prototype patterns for Hopfield retrieval.
    
    The memory bank stores K prototype vectors of dimension d. During retrieval,
    queries are compared against all prototypes using softmax attention, and
    the output is a weighted combination of prototype values.
    
    Attributes:
        num_patterns (int): Number of prototype patterns K
        pattern_dim (int): Dimension of each pattern d
        keys (nn.Parameter): Memory keys for similarity computation [K, d]
        values (nn.Parameter): Memory values for retrieval [K, d] (can be tied to keys)
    """
    
    def __init__(
        self,
        num_patterns: int,
        pattern_dim: int,
        tie_keys_values: bool = True,
        init_std: float = 0.02,
        normalize_keys: bool = False,
        normalize_queries: bool = False,
    ):
        """
        Initialize the memory bank.
        
        Args:
            num_patterns: Number of prototype patterns to store (K)
            pattern_dim: Dimension of each pattern (d)
            tie_keys_values: If True, use same matrix for keys and values
            init_std: Standard deviation for weight initialization
            normalize_keys: Normalize memory keys to unit norm (for numerical stability)
            normalize_queries: Normalize queries to unit norm (for numerical stability)
        """
        super().__init__()
        self.num_patterns = num_patterns
        self.pattern_dim = pattern_dim
        self.tie_keys_values = tie_keys_values
        self.normalize_keys = normalize_keys
        self.normalize_queries = normalize_queries
        
        # Initialize memory patterns
        self.keys = nn.Parameter(torch.randn(num_patterns, pattern_dim) * init_std)
        
        if not tie_keys_values:
            self.values = nn.Parameter(torch.randn(num_patterns, pattern_dim) * init_std)
        else:
            self.register_parameter("values", None)
    
    def get_values(self) -> torch.Tensor:
        """Get the value matrix (may be tied to keys)."""
        if self.tie_keys_values:
            return self.keys
        return self.values
    
    def retrieve(
        self,
        queries: torch.Tensor,
        beta: float = 1.0,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Retrieve from memory using Hopfield/attention mechanism.
        
        Computes: output = softmax(beta * queries @ keys.T) @ values
        
        Args:
            queries: Query vectors [N, d] or [B, N, d]
            beta: Inverse temperature for softmax sharpness
            return_attention: If True, also return attention weights
            
        Returns:
            retrieved: Retrieved patterns [N, d] or [B, N, d]
            attention: Attention weights [N, K] or [B, N, K] (if return_attention)
        """
        keys = self.keys
        values = self.get_values()
        
        # Normalize for numerical stability (prevents overflow in exp(β * scores))
        if self.normalize_keys:
            keys = F.normalize(keys, p=2, dim=-1)
        if self.normalize_queries:
            queries = F.normalize(queries, p=2, dim=-1)
        
        # Compute attention scores: [N, K] or [B, N, K]
        scores = beta * torch.matmul(queries, keys.T)
        attention = F.softmax(scores, dim=-1)
        
        # Retrieve weighted combination of values
        retrieved = torch.matmul(attention, values)
        
        if return_attention:
            return retrieved, attention
        return retrieved, None
    
    def compute_energy(self, x: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        """
        Compute the Hopfield energy for given states.
        
        E_hopfield(x) = -β⁻¹ * log(Σᵢ exp(β * x^T mᵢ)) + ½||x||²
        
        Args:
            x: State vectors [N, d]
            beta: Inverse temperature
            
        Returns:
            energy: Scalar energy value
        """
        keys = self.keys
        
        # Compute log-sum-exp term: -β⁻¹ * lse(β * x @ M.T)
        scores = beta * torch.matmul(x, keys.T)  # [N, K]
        lse = torch.logsumexp(scores, dim=-1)  # [N]
        hopfield_energy = -lse.sum() / beta
        
        # Quadratic regularization: ½||x||²
        reg_energy = 0.5 * (x ** 2).sum()
        
        return hopfield_energy + reg_energy
    
    def extra_repr(self) -> str:
        return (
            f"num_patterns={self.num_patterns}, "
            f"pattern_dim={self.pattern_dim}, "
            f"tie_keys_values={self.tie_keys_values}"
        )
