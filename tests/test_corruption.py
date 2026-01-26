"""Unit tests for corruption utilities."""

import pytest
import torch
from torch_geometric.data import Data

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.corruption import (
    add_feature_noise,
    mask_features,
    corrupt_edges,
    corrupt_labels,
    apply_corruption,
    CorruptionConfig,
)


class TestFeatureCorruption:
    """Tests for feature corruption utilities."""
    
    def test_add_feature_noise(self):
        """Test Gaussian noise addition."""
        x = torch.randn(100, 16)
        noise_std = 0.5
        
        x_noisy = add_feature_noise(x, noise_std, seed=42)
        
        # Should be different from original
        assert not torch.allclose(x, x_noisy)
        
        # Check noise magnitude is roughly correct
        noise = x_noisy - x
        actual_std = noise.std()
        assert abs(actual_std - noise_std) < 0.1
    
    def test_add_feature_noise_zero(self):
        """Test that zero noise returns original."""
        x = torch.randn(100, 16)
        
        x_noisy = add_feature_noise(x, 0.0)
        
        assert torch.allclose(x, x_noisy)
    
    def test_mask_features(self):
        """Test feature masking."""
        x = torch.ones(100, 16)
        mask_ratio = 0.5
        
        x_masked = mask_features(x, mask_ratio, seed=42)
        
        # Some values should be zero
        assert (x_masked == 0).any()
        
        # Roughly correct fraction should be masked
        actual_ratio = (x_masked == 0).float().mean()
        assert abs(actual_ratio - mask_ratio) < 0.1
    
    def test_mask_features_deterministic(self):
        """Test that masking is deterministic with seed."""
        x = torch.randn(100, 16)
        
        x_masked1 = mask_features(x, 0.3, seed=42)
        x_masked2 = mask_features(x, 0.3, seed=42)
        
        assert torch.allclose(x_masked1, x_masked2)


class TestEdgeCorruption:
    """Tests for edge corruption utilities."""
    
    def test_corrupt_edges_drop(self):
        """Test edge dropping."""
        num_nodes = 50
        num_edges = 200
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        
        corrupted = corrupt_edges(edge_index, num_nodes, drop_ratio=0.3, seed=42)
        
        # Should have fewer edges
        assert corrupted.size(1) < edge_index.size(1)
    
    def test_corrupt_edges_add(self):
        """Test edge addition."""
        num_nodes = 50
        num_edges = 100
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        
        corrupted = corrupt_edges(edge_index, num_nodes, add_ratio=0.5, seed=42)
        
        # Should have more edges (after deduplication)
        # Note: due to deduplication, may not be exactly 1.5x
        assert corrupted.size(1) >= edge_index.size(1)
    
    def test_corrupt_edges_no_corruption(self):
        """Test that no corruption returns original."""
        num_nodes = 50
        edge_index = torch.randint(0, num_nodes, (2, 100))
        
        corrupted = corrupt_edges(edge_index, num_nodes, drop_ratio=0.0, add_ratio=0.0)
        
        assert torch.equal(edge_index, corrupted)


class TestLabelCorruption:
    """Tests for label corruption utilities."""
    
    def test_corrupt_labels(self):
        """Test label flipping."""
        num_nodes = 100
        num_classes = 5
        
        y = torch.randint(0, num_classes, (num_nodes,))
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[:50] = True  # 50 training nodes
        
        y_corrupted = corrupt_labels(y, train_mask, num_classes, flip_ratio=0.5, seed=42)
        
        # Some training labels should change
        train_changed = (y[train_mask] != y_corrupted[train_mask]).sum()
        assert train_changed > 0
        
        # Test labels should not change
        test_mask = ~train_mask
        assert torch.equal(y[test_mask], y_corrupted[test_mask])
    
    def test_corrupt_labels_zero(self):
        """Test that zero flip ratio returns original."""
        y = torch.randint(0, 5, (100,))
        train_mask = torch.ones(100, dtype=torch.bool)
        
        y_corrupted = corrupt_labels(y, train_mask, 5, flip_ratio=0.0)
        
        assert torch.equal(y, y_corrupted)


class TestApplyCorruption:
    """Tests for combined corruption application."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample PyG data."""
        num_nodes = 100
        num_features = 16
        num_edges = 300
        num_classes = 5
        
        x = torch.randn(num_nodes, num_features)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        y = torch.randint(0, num_classes, (num_nodes,))
        
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        train_mask[:60] = True
        val_mask[60:80] = True
        test_mask[80:] = True
        
        return Data(
            x=x,
            edge_index=edge_index,
            y=y,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
        )
    
    def test_apply_corruption_features(self, sample_data):
        """Test applying feature corruption."""
        config = CorruptionConfig(feature_noise_std=0.5, seed=42)
        
        corrupted = apply_corruption(sample_data, config)
        
        assert not torch.allclose(sample_data.x, corrupted.x)
        assert torch.equal(sample_data.edge_index, corrupted.edge_index)
        assert torch.equal(sample_data.y, corrupted.y)
    
    def test_apply_corruption_combined(self, sample_data):
        """Test applying multiple corruptions."""
        config = CorruptionConfig(
            feature_noise_std=0.3,
            edge_drop_ratio=0.2,
            label_flip_ratio=0.1,
            seed=42,
        )
        
        corrupted = apply_corruption(sample_data, config)
        
        # Features should change
        assert not torch.allclose(sample_data.x, corrupted.x)
        
        # Edge count should change
        assert corrupted.edge_index.size(1) != sample_data.edge_index.size(1)
        
        # Some training labels should change
        train_changed = (
            sample_data.y[sample_data.train_mask] != 
            corrupted.y[corrupted.train_mask]
        ).sum()
        assert train_changed > 0
    
    def test_apply_corruption_preserves_masks(self, sample_data):
        """Test that corruption preserves data masks."""
        config = CorruptionConfig(feature_noise_std=0.5)
        
        corrupted = apply_corruption(sample_data, config)
        
        assert torch.equal(sample_data.train_mask, corrupted.train_mask)
        assert torch.equal(sample_data.val_mask, corrupted.val_mask)
        assert torch.equal(sample_data.test_mask, corrupted.test_mask)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
