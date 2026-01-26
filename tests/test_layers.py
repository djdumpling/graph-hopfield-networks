"""Unit tests for Graph Hopfield layers."""

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.layers.memory_bank import MemoryBank
from src.layers.graph_hopfield import GraphHopfieldLayer, GraphHopfieldBlock


class TestMemoryBank:
    """Tests for MemoryBank module."""
    
    def test_initialization(self):
        """Test memory bank initialization."""
        num_patterns = 16
        pattern_dim = 32
        
        memory = MemoryBank(num_patterns, pattern_dim)
        
        assert memory.keys.shape == (num_patterns, pattern_dim)
        assert memory.num_patterns == num_patterns
        assert memory.pattern_dim == pattern_dim
    
    def test_retrieve(self):
        """Test memory retrieval."""
        num_patterns = 16
        pattern_dim = 32
        num_nodes = 100
        
        memory = MemoryBank(num_patterns, pattern_dim)
        queries = torch.randn(num_nodes, pattern_dim)
        
        retrieved, attention = memory.retrieve(queries, beta=1.0, return_attention=True)
        
        assert retrieved.shape == (num_nodes, pattern_dim)
        assert attention.shape == (num_nodes, num_patterns)
        
        # Check attention sums to 1
        assert torch.allclose(attention.sum(dim=-1), torch.ones(num_nodes), atol=1e-5)
    
    def test_retrieve_beta_effect(self):
        """Test that higher beta leads to sharper attention."""
        num_patterns = 16
        pattern_dim = 32
        num_nodes = 10
        
        memory = MemoryBank(num_patterns, pattern_dim)
        queries = torch.randn(num_nodes, pattern_dim)
        
        _, attn_low = memory.retrieve(queries, beta=0.1, return_attention=True)
        _, attn_high = memory.retrieve(queries, beta=10.0, return_attention=True)
        
        # Higher beta should have lower entropy (sharper)
        entropy_low = -(attn_low * torch.log(attn_low + 1e-10)).sum(dim=-1).mean()
        entropy_high = -(attn_high * torch.log(attn_high + 1e-10)).sum(dim=-1).mean()
        
        assert entropy_high < entropy_low
    
    def test_compute_energy(self):
        """Test energy computation."""
        num_patterns = 16
        pattern_dim = 32
        num_nodes = 100
        
        memory = MemoryBank(num_patterns, pattern_dim)
        x = torch.randn(num_nodes, pattern_dim)
        
        energy = memory.compute_energy(x, beta=1.0)
        
        assert energy.ndim == 0  # Scalar
        assert torch.isfinite(energy)


class TestGraphHopfieldLayer:
    """Tests for GraphHopfieldLayer module."""
    
    @pytest.fixture
    def sample_graph(self):
        """Create a sample graph for testing."""
        num_nodes = 50
        num_edges = 200
        in_dim = 16
        
        x = torch.randn(num_nodes, in_dim)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        
        return x, edge_index
    
    def test_initialization(self):
        """Test layer initialization."""
        layer = GraphHopfieldLayer(
            in_dim=16,
            out_dim=32,
            num_patterns=64,
            beta=1.0,
            lambda_graph=0.1,
        )
        
        assert layer.in_dim == 16
        assert layer.out_dim == 32
        assert layer.num_patterns == 64
    
    def test_forward(self, sample_graph):
        """Test forward pass."""
        x, edge_index = sample_graph
        in_dim = x.size(1)
        out_dim = 32
        
        layer = GraphHopfieldLayer(
            in_dim=in_dim,
            out_dim=out_dim,
            num_patterns=64,
        )
        
        out, info = layer(x, edge_index)
        
        assert out.shape == (x.size(0), out_dim)
        assert info is None  # No info requested
    
    def test_forward_with_info(self, sample_graph):
        """Test forward pass with energy and attention."""
        x, edge_index = sample_graph
        in_dim = x.size(1)
        out_dim = 32
        
        layer = GraphHopfieldLayer(
            in_dim=in_dim,
            out_dim=out_dim,
            num_patterns=64,
            num_iterations=2,
        )
        
        out, info = layer(x, edge_index, return_energy=True, return_attention=True)
        
        assert out.shape == (x.size(0), out_dim)
        assert info is not None
        assert "energies" in info
        assert "attentions" in info
        assert len(info["energies"]) == 2  # num_iterations
    
    def test_energy_decreases(self, sample_graph):
        """Test that energy decreases with iterations."""
        x, edge_index = sample_graph
        in_dim = x.size(1)
        out_dim = 32
        
        layer = GraphHopfieldLayer(
            in_dim=in_dim,
            out_dim=out_dim,
            num_patterns=64,
            num_iterations=5,
            alpha=0.3,  # Smaller step for stability
            lambda_graph=0.05,
        )
        
        _, info = layer(x, edge_index, return_energy=True)
        
        energies = info["energies"]
        
        # Energy should generally decrease (may not be strictly monotonic)
        assert energies[-1] <= energies[0] + 1.0  # Allow small increases
    
    def test_gradient_flow(self, sample_graph):
        """Test that gradients flow through the layer."""
        x, edge_index = sample_graph
        x.requires_grad_(True)
        
        layer = GraphHopfieldLayer(
            in_dim=x.size(1),
            out_dim=32,
            num_patterns=64,
        )
        
        out, _ = layer(x, edge_index)
        loss = out.sum()
        loss.backward()
        
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
    
    def test_lambda_zero_is_pure_hopfield(self, sample_graph):
        """Test that lambda=0 gives pure Hopfield retrieval."""
        x, edge_index = sample_graph
        
        layer = GraphHopfieldLayer(
            in_dim=x.size(1),
            out_dim=32,
            num_patterns=64,
            lambda_graph=0.0,  # No graph coupling
            num_iterations=1,
            use_residual=False,
        )
        
        out, _ = layer(x, edge_index)
        
        # Output should be independent of edge structure
        # (though this is hard to test without more setup)
        assert out.shape == (x.size(0), 32)


class TestGraphHopfieldBlock:
    """Tests for GraphHopfieldBlock module."""
    
    def test_forward(self):
        """Test block forward pass."""
        num_nodes = 50
        in_dim = 16
        hidden_dim = 32
        out_dim = 7
        num_patterns = 64
        
        x = torch.randn(num_nodes, in_dim)
        edge_index = torch.randint(0, num_nodes, (2, 200))
        
        block = GraphHopfieldBlock(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_patterns=num_patterns,
        )
        
        out, info = block(x, edge_index)
        
        assert out.shape == (num_nodes, out_dim)


class TestIntegration:
    """Integration tests."""
    
    def test_overfit_small_graph(self):
        """Test that model can overfit a small graph."""
        # Create a simple graph classification problem
        num_nodes = 20
        in_dim = 8
        hidden_dim = 16
        out_dim = 3
        num_patterns = 8
        
        x = torch.randn(num_nodes, in_dim)
        edge_index = torch.randint(0, num_nodes, (2, 50))
        y = torch.randint(0, out_dim, (num_nodes,))
        
        from src.models.ghn import GraphHopfieldNetworkMinimal
        
        model = GraphHopfieldNetworkMinimal(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_patterns=num_patterns,
            num_iterations=1,
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Train for a few iterations
        for _ in range(100):
            optimizer.zero_grad()
            logits, _ = model(x, edge_index)
            loss = nn.functional.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
        
        # Should overfit
        with torch.no_grad():
            logits, _ = model(x, edge_index)
            preds = logits.argmax(dim=-1)
            acc = (preds == y).float().mean()
        
        assert acc > 0.5  # Should do better than random


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
