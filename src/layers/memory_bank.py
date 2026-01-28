"""Memory Bank module for storing learnable prototype patterns."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


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
        init_std: float = None,
        normalize_keys: bool = False,
        normalize_queries: bool = False,
        learnable_beta: bool = False,
        initial_beta: float = 1.0,
        use_spectral_norm_constraint: bool = True,
        use_query_proj: bool = False,
    ):
        """
        Initialize the memory bank.

        Args:
            num_patterns: Number of prototype patterns to store (K)
            pattern_dim: Dimension of each pattern (d)
            tie_keys_values: If True, use same matrix for keys and values
            init_std: Standard deviation for weight initialization (default: 1/sqrt(pattern_dim))
            normalize_keys: Normalize memory keys to unit norm (for numerical stability)
            normalize_queries: Normalize queries to unit norm (for numerical stability)
            learnable_beta: If True, make beta (inverse temperature) learnable
            initial_beta: Initial value for beta (default: 1.0)
            use_spectral_norm_constraint: Constrain beta * ||M||^2 < 1 for convexity
            use_query_proj: If True, add a learnable query projection (default: False)
        """
        super().__init__()
        self.num_patterns = num_patterns
        self.pattern_dim = pattern_dim
        self.tie_keys_values = tie_keys_values
        self.normalize_keys = normalize_keys
        self.normalize_queries = normalize_queries
        self.learnable_beta = learnable_beta
        self.use_spectral_norm_constraint = use_spectral_norm_constraint
        self.use_query_proj = use_query_proj

        # Default initialization: 1/sqrt(pattern_dim) for better gradient flow
        if init_std is None:
            init_std = 1.0 / math.sqrt(pattern_dim)

        # Initialize memory patterns with orthogonal initialization
        keys_init = torch.empty(num_patterns, pattern_dim)
        nn.init.orthogonal_(keys_init)
        keys_init = keys_init * init_std * math.sqrt(pattern_dim)  # Scale to target std
        self.keys = nn.Parameter(keys_init)

        if not tie_keys_values:
            values_init = torch.empty(num_patterns, pattern_dim)
            nn.init.orthogonal_(values_init)
            values_init = values_init * init_std * math.sqrt(pattern_dim)
            self.values = nn.Parameter(values_init)
        else:
            self.register_parameter("values", None)

        # Learnable temperature (inverse beta stored as log for positivity)
        if learnable_beta:
            # Store log(beta) to ensure beta > 0
            self.log_beta = nn.Parameter(torch.tensor(math.log(initial_beta)))
        else:
            self.register_buffer("log_beta", torch.tensor(math.log(initial_beta)))

        # Query projection for better matching capability
        if use_query_proj:
            self.query_proj = nn.Linear(pattern_dim, pattern_dim, bias=False)
            # Initialize close to identity for stable training start
            nn.init.eye_(self.query_proj.weight)
        else:
            self.query_proj = None
    
    def get_values(self) -> torch.Tensor:
        """Get the value matrix (may be tied to keys)."""
        if self.tie_keys_values:
            return self.keys
        return self.values

    def get_beta(self, keys: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get the effective beta value, optionally constrained by spectral norm.

        For convexity of the Hopfield energy, we need: beta * ||M||^2 < 1
        where ||M|| is the spectral norm (largest singular value) of the key matrix.

        Args:
            keys: Optional key matrix (uses self.keys if not provided)

        Returns:
            Effective beta value (scalar tensor)
        """
        beta = torch.exp(self.log_beta)

        if self.use_spectral_norm_constraint:
            if keys is None:
                keys = self.keys
            # Compute spectral norm (largest singular value)
            # Use power iteration approximation for efficiency
            spectral_norm_sq = self._estimate_spectral_norm_sq(keys)
            # Constrain: beta * ||M||^2 < 1  =>  beta < 1 / ||M||^2
            # Use soft constraint with margin for stability
            max_beta = 0.99 / (spectral_norm_sq + 1e-6)
            beta = torch.minimum(beta, max_beta)

        return beta

    def _estimate_spectral_norm_sq(self, M: torch.Tensor, num_iters: int = 1) -> torch.Tensor:
        """
        Estimate ||M||^2 using power iteration.

        For M of shape [K, d], we compute the largest eigenvalue of M @ M.T
        which equals ||M||^2 (squared spectral norm).

        Args:
            M: Matrix of shape [K, d]
            num_iters: Number of power iterations

        Returns:
            Estimated squared spectral norm
        """
        # M @ M.T has shape [K, K]
        # Use power iteration on M @ M.T
        K = M.shape[0]
        # Initialize random vector
        v = torch.randn(K, device=M.device, dtype=M.dtype)
        v = v / v.norm()

        for _ in range(num_iters):
            # v = M @ M.T @ v / ||M @ M.T @ v||
            Mv = M.T @ v  # [d]
            MMv = M @ Mv  # [K]
            v = MMv / (MMv.norm() + 1e-8)

        # Rayleigh quotient gives eigenvalue estimate
        Mv = M.T @ v
        MMv = M @ Mv
        spectral_norm_sq = (v * MMv).sum()

        return spectral_norm_sq
    
    def retrieve(
        self,
        queries: torch.Tensor,
        beta: Optional[float] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Retrieve from memory using Hopfield/attention mechanism.

        Computes: output = softmax(beta * proj(queries) @ keys.T) @ values

        Args:
            queries: Query vectors [N, d] or [B, N, d]
            beta: Inverse temperature for softmax sharpness (if None, uses learnable beta)
            return_attention: If True, also return attention weights

        Returns:
            retrieved: Retrieved patterns [N, d] or [B, N, d]
            attention: Attention weights [N, K] or [B, N, K] (if return_attention)
        """
        keys = self.keys
        values = self.get_values()

        # Apply query projection if enabled
        if self.query_proj is not None:
            queries = self.query_proj(queries)

        # Normalize for numerical stability (prevents overflow in exp(β * scores))
        if self.normalize_keys:
            keys = F.normalize(keys, p=2, dim=-1)
        if self.normalize_queries:
            queries = F.normalize(queries, p=2, dim=-1)

        # Get effective beta (with optional spectral norm constraint)
        if beta is None:
            effective_beta = self.get_beta(keys)
        else:
            effective_beta = beta

        # Compute attention scores: [N, K] or [B, N, K]
        scores = effective_beta * torch.matmul(queries, keys.T)
        attention = F.softmax(scores, dim=-1)

        # Retrieve weighted combination of values
        retrieved = torch.matmul(attention, values)

        if return_attention:
            return retrieved, attention
        return retrieved, None
    
    def compute_energy(self, x: torch.Tensor, beta: Optional[float] = None) -> torch.Tensor:
        """
        Compute the Hopfield energy for given states.

        E_hopfield(x) = -β⁻¹ * log(Σᵢ exp(β * x^T mᵢ)) + ½||x||²

        Args:
            x: State vectors [N, d]
            beta: Inverse temperature (if None, uses learnable beta)

        Returns:
            energy: Scalar energy value
        """
        keys = self.keys

        # Get effective beta
        if beta is None:
            effective_beta = self.get_beta(keys)
        else:
            effective_beta = beta

        # Compute log-sum-exp term: -β⁻¹ * lse(β * x @ M.T)
        scores = effective_beta * torch.matmul(x, keys.T)  # [N, K]
        lse = torch.logsumexp(scores, dim=-1)  # [N]
        hopfield_energy = -lse.sum() / effective_beta

        # Quadratic regularization: ½||x||²
        reg_energy = 0.5 * (x ** 2).sum()

        return hopfield_energy + reg_energy
    
    def compute_diversity_loss(self, threshold: float = 0.5) -> torch.Tensor:
        """
        Compute diversity regularization loss to prevent pattern collapse.

        Penalizes pairs of patterns that are too similar (cosine similarity > threshold).
        This encourages the memory bank to maintain diverse, distinct patterns.

        Args:
            threshold: Similarity threshold above which patterns are penalized (default: 0.5)

        Returns:
            Diversity loss (scalar tensor). Lower means more diverse patterns.
        """
        # Normalize keys to unit vectors for cosine similarity
        M_norm = F.normalize(self.keys, p=2, dim=-1)  # [K, d]

        # Compute pairwise cosine similarities: [K, K]
        sim = torch.matmul(M_norm, M_norm.T)

        # Mask out diagonal (self-similarity = 1)
        K = self.num_patterns
        mask = ~torch.eye(K, dtype=torch.bool, device=sim.device)

        # Penalize similarities above threshold using squared hinge loss
        # Only off-diagonal elements
        excess_sim = F.relu(sim[mask] - threshold)
        diversity_loss = excess_sim.pow(2).mean()

        return diversity_loss

    def get_pattern_similarity_stats(self) -> dict:
        """
        Compute statistics about pattern similarity for monitoring.

        Returns:
            Dict with mean, max, and std of off-diagonal cosine similarities.
        """
        with torch.no_grad():
            M_norm = F.normalize(self.keys, p=2, dim=-1)
            sim = torch.matmul(M_norm, M_norm.T)

            K = self.num_patterns
            mask = ~torch.eye(K, dtype=torch.bool, device=sim.device)
            off_diag = sim[mask]

            return {
                "mean_similarity": off_diag.mean().item(),
                "max_similarity": off_diag.max().item(),
                "std_similarity": off_diag.std().item(),
            }

    def extra_repr(self) -> str:
        beta_str = f"learnable_beta={self.learnable_beta}"
        if self.learnable_beta:
            beta_str += f", initial_beta={math.exp(self.log_beta.item()):.3f}"
        return (
            f"num_patterns={self.num_patterns}, "
            f"pattern_dim={self.pattern_dim}, "
            f"tie_keys_values={self.tie_keys_values}, "
            f"use_query_proj={self.use_query_proj}, "
            f"{beta_str}"
        )


class MultiHeadMemoryBank(nn.Module):
    """
    Multi-head memory bank for increased capacity and expressiveness.

    Similar to multi-head attention in Transformers, this module uses multiple
    independent memory banks (heads) that operate on different subspaces of the
    input. Each head has K/num_heads patterns of dimension d/num_heads.

    This increases capacity (similar to GAT's multi-head attention) while
    maintaining parameter count.

    Attributes:
        num_heads (int): Number of attention heads
        num_patterns (int): Total number of patterns across all heads
        pattern_dim (int): Total dimension (each head uses pattern_dim/num_heads)
    """

    def __init__(
        self,
        num_patterns: int,
        pattern_dim: int,
        num_heads: int = 4,
        tie_keys_values: bool = False,
        normalize_keys: bool = False,
        normalize_queries: bool = False,
        learnable_beta: bool = True,
        initial_beta: float = 1.0,
        use_spectral_norm_constraint: bool = True,
        use_query_proj: bool = True,
    ):
        """
        Initialize the multi-head memory bank.

        Args:
            num_patterns: Total number of patterns (divided among heads)
            pattern_dim: Total dimension (must be divisible by num_heads)
            num_heads: Number of attention heads (default: 4)
            tie_keys_values: If True, use same matrix for keys and values
            normalize_keys: Normalize memory keys to unit norm
            normalize_queries: Normalize queries to unit norm
            learnable_beta: If True, make beta learnable (shared across heads)
            initial_beta: Initial value for beta
            use_spectral_norm_constraint: Constrain beta for convexity
            use_query_proj: Add learnable query projection
        """
        super().__init__()

        if pattern_dim % num_heads != 0:
            raise ValueError(f"pattern_dim ({pattern_dim}) must be divisible by num_heads ({num_heads})")

        self.num_heads = num_heads
        self.num_patterns = num_patterns
        self.pattern_dim = pattern_dim
        self.head_dim = pattern_dim // num_heads
        self.patterns_per_head = num_patterns // num_heads

        # Create individual memory banks for each head
        self.heads = nn.ModuleList([
            MemoryBank(
                num_patterns=self.patterns_per_head,
                pattern_dim=self.head_dim,
                tie_keys_values=tie_keys_values,
                normalize_keys=normalize_keys,
                normalize_queries=normalize_queries,
                learnable_beta=learnable_beta,
                initial_beta=initial_beta,
                use_spectral_norm_constraint=use_spectral_norm_constraint,
                use_query_proj=use_query_proj,
            )
            for _ in range(num_heads)
        ])

        # Output projection to combine heads
        self.output_proj = nn.Linear(pattern_dim, pattern_dim, bias=False)
        nn.init.eye_(self.output_proj.weight)  # Initialize close to identity

    def retrieve(
        self,
        queries: torch.Tensor,
        beta: Optional[float] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Retrieve from memory using multi-head Hopfield/attention mechanism.

        Args:
            queries: Query vectors [N, d] or [B, N, d]
            beta: Inverse temperature (if None, uses learnable beta)
            return_attention: If True, also return attention weights

        Returns:
            retrieved: Retrieved patterns [N, d] or [B, N, d]
            attention: Concatenated attention weights from all heads (if return_attention)
        """
        # Split queries into heads: [N, d] -> num_heads x [N, head_dim]
        query_heads = queries.split(self.head_dim, dim=-1)

        head_outputs = []
        all_attentions = []

        for i, (head, query_head) in enumerate(zip(self.heads, query_heads)):
            retrieved, attn = head.retrieve(query_head, beta=beta, return_attention=return_attention)
            head_outputs.append(retrieved)
            if return_attention and attn is not None:
                all_attentions.append(attn)

        # Concatenate head outputs: [N, head_dim] * num_heads -> [N, d]
        combined = torch.cat(head_outputs, dim=-1)

        # Apply output projection
        output = self.output_proj(combined)

        if return_attention:
            # Stack attentions: [num_heads, N, K_per_head]
            attention = torch.stack(all_attentions, dim=0) if all_attentions else None
            return output, attention
        return output, None

    def compute_energy(self, x: torch.Tensor, beta: Optional[float] = None) -> torch.Tensor:
        """
        Compute the total Hopfield energy across all heads.

        Args:
            x: State vectors [N, d]
            beta: Inverse temperature

        Returns:
            Total energy (sum across heads)
        """
        x_heads = x.split(self.head_dim, dim=-1)
        total_energy = 0.0

        for head, x_head in zip(self.heads, x_heads):
            total_energy = total_energy + head.compute_energy(x_head, beta=beta)

        return total_energy

    def compute_diversity_loss(self, threshold: float = 0.5) -> torch.Tensor:
        """
        Compute diversity loss across all heads.

        Args:
            threshold: Similarity threshold

        Returns:
            Average diversity loss across heads
        """
        total_loss = 0.0
        for head in self.heads:
            total_loss = total_loss + head.compute_diversity_loss(threshold)
        return total_loss / self.num_heads

    def get_pattern_similarity_stats(self) -> dict:
        """
        Get pattern similarity statistics across all heads.

        Returns:
            Dict with per-head and aggregate statistics
        """
        head_stats = [head.get_pattern_similarity_stats() for head in self.heads]

        # Aggregate
        mean_sims = [s["mean_similarity"] for s in head_stats]
        max_sims = [s["max_similarity"] for s in head_stats]

        return {
            "per_head": head_stats,
            "mean_similarity": sum(mean_sims) / len(mean_sims),
            "max_similarity": max(max_sims),
        }

    def extra_repr(self) -> str:
        return (
            f"num_heads={self.num_heads}, "
            f"num_patterns={self.num_patterns}, "
            f"pattern_dim={self.pattern_dim}, "
            f"head_dim={self.head_dim}"
        )
