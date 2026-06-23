#!/usr/bin/env python
"""
Tests for SparseTensorList and related functionality.

Run with:
    pytest tests/test_sparse_tensor_list.py -v
Or:
    python tests/test_sparse_tensor_list.py
"""

import torch
import sys
import os

# Try importing pytest, but make tests runnable without it
try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False
    # Create dummy pytest decorators
    class pytest:
        @staticmethod
        def fixture(func):
            return func

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch_sla import SparseTensor, SparseTensorList


class TestConnectedComponents:
    """Tests for SparseTensor.connected_components()."""
    
    def test_single_component(self):
        """Single connected graph."""
        # Simple 3-node graph: 0-1-2
        val = torch.ones(4)
        row = torch.tensor([0, 1, 1, 2])
        col = torch.tensor([1, 0, 2, 1])
        A = SparseTensor(val, row, col, (3, 3))
        
        labels, n_comp = A.connected_components()
        assert n_comp == 1
        assert labels.max() == 0  # All same label
    
    def test_two_components(self):
        """Two disconnected components."""
        # Component 1: nodes 0, 1
        # Component 2: nodes 2, 3
        val = torch.ones(4)
        row = torch.tensor([0, 1, 2, 3])
        col = torch.tensor([1, 0, 3, 2])
        A = SparseTensor(val, row, col, (4, 4))
        
        labels, n_comp = A.connected_components()
        assert n_comp == 2
        assert labels[0] == labels[1]
        assert labels[2] == labels[3]
        assert labels[0] != labels[2]
    
    def test_three_components(self):
        """Three disconnected components."""
        # 0-1, 2-3, 4-5
        val = torch.ones(6)
        row = torch.tensor([0, 1, 2, 3, 4, 5])
        col = torch.tensor([1, 0, 3, 2, 5, 4])
        A = SparseTensor(val, row, col, (6, 6))
        
        labels, n_comp = A.connected_components()
        assert n_comp == 3
    
    def test_has_isolated_components(self):
        """Test has_isolated_components()."""
        # Single component
        val = torch.ones(4)
        row = torch.tensor([0, 1, 1, 2])
        col = torch.tensor([1, 0, 2, 1])
        A = SparseTensor(val, row, col, (3, 3))
        assert not A.has_isolated_components()
        
        # Two components
        val = torch.ones(4)
        row = torch.tensor([0, 1, 2, 3])
        col = torch.tensor([1, 0, 3, 2])
        A = SparseTensor(val, row, col, (4, 4))
        assert A.has_isolated_components()


class TestToConnectedComponents:
    """Tests for SparseTensor.to_connected_components()."""
    
    def test_split_two_components(self):
        """Split block diagonal into components."""
        # Create block diagonal: 2x2 block + 3x3 block
        val = torch.tensor([1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
        row = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        col = torch.tensor([0, 1, 0, 1, 2, 3, 2, 4, 3, 4])
        A = SparseTensor(val, row, col, (5, 5))
        
        stl = A.to_connected_components()
        
        assert len(stl) == 2
        # First component: 2 nodes
        assert stl[0].sparse_shape == (2, 2)
        # Second component: 3 nodes
        assert stl[1].sparse_shape == (3, 3)
    
    def test_single_component_returns_list(self):
        """Single component still returns SparseTensorList."""
        val = torch.ones(4)
        row = torch.tensor([0, 1, 1, 2])
        col = torch.tensor([1, 0, 2, 1])
        A = SparseTensor(val, row, col, (3, 3))
        
        stl = A.to_connected_components()
        assert isinstance(stl, SparseTensorList)
        assert len(stl) == 1


class TestSparseTensorListOperations:
    """Tests for SparseTensorList arithmetic operations."""
    
    @pytest.fixture
    def sample_list(self):
        """Create sample SparseTensorList."""
        tensors = []
        for n in [3, 4, 5]:
            val = torch.randn(n * 2)
            row = torch.cat([torch.arange(n), torch.arange(n)])
            col = torch.cat([torch.arange(n), (torch.arange(n) + 1) % n])
            tensors.append(SparseTensor(val, row, col, (n, n)))
        return SparseTensorList(tensors)
    
    def test_matmul_list(self, sample_list):
        """Test __matmul__ with list of vectors."""
        x_list = [torch.randn(t.sparse_shape[1]) for t in sample_list]
        y_list = sample_list @ x_list
        
        assert len(y_list) == len(sample_list)
        for y, t in zip(y_list, sample_list):
            assert y.shape == (t.sparse_shape[0],)
    
    def test_matmul_broadcast(self, sample_list):
        """Test __matmul__ with broadcast."""
        # All same size for broadcast
        n = 5
        tensors = []
        for _ in range(3):
            val = torch.randn(n * 2)
            row = torch.cat([torch.arange(n), torch.arange(n)])
            col = torch.cat([torch.arange(n), (torch.arange(n) + 1) % n])
            tensors.append(SparseTensor(val, row, col, (n, n)))
        stl = SparseTensorList(tensors)
        
        x = torch.randn(n)
        y_list = stl @ x
        
        assert len(y_list) == 3
        for y in y_list:
            assert y.shape == (n,)
    
    def test_add(self, sample_list):
        """Test __add__ with SparseTensorList."""
        result = sample_list + sample_list
        assert len(result) == len(sample_list)
    
    def test_mul_scalar(self, sample_list):
        """Test __mul__ with scalar."""
        result = sample_list * 2.0
        assert len(result) == len(sample_list)
        
        for orig, scaled in zip(sample_list, result):
            assert torch.allclose(scaled.values, orig.values * 2.0)
    
    def test_neg(self, sample_list):
        """Test __neg__."""
        result = -sample_list
        for orig, neg in zip(sample_list, result):
            assert torch.allclose(neg.values, -orig.values)
    
    def test_sum(self, sample_list):
        """Test sum()."""
        sums = sample_list.sum()
        assert len(sums) == len(sample_list)
        for s in sums:
            assert s.dim() == 0  # Scalar


class TestBlockDiagonal:
    """Tests for block diagonal conversion."""
    
    def test_to_block_diagonal(self):
        """Test to_block_diagonal()."""
        # Create two small matrices
        A1 = SparseTensor(torch.ones(2), torch.tensor([0, 1]), torch.tensor([0, 1]), (2, 2))
        A2 = SparseTensor(torch.ones(3), torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2]), (3, 3))
        
        stl = SparseTensorList([A1, A2])
        block_diag = stl.to_block_diagonal()
        
        assert block_diag.sparse_shape == (5, 5)
        assert block_diag.nnz == 5
    
    def test_from_block_diagonal(self):
        """Test from_block_diagonal()."""
        # Create block diagonal
        val = torch.tensor([1.0, 1.0, 2.0, 2.0, 2.0])
        row = torch.tensor([0, 1, 2, 3, 4])
        col = torch.tensor([0, 1, 2, 3, 4])
        block_diag = SparseTensor(val, row, col, (5, 5))
        
        stl = SparseTensorList.from_block_diagonal(block_diag, [(2, 2), (3, 3)])
        
        assert len(stl) == 2
        assert stl[0].sparse_shape == (2, 2)
        assert stl[1].sparse_shape == (3, 3)
    
    def test_roundtrip(self):
        """Test to_block_diagonal -> from_block_diagonal roundtrip."""
        A1 = SparseTensor(torch.randn(4), torch.tensor([0, 0, 1, 1]), 
                          torch.tensor([0, 1, 0, 1]), (2, 2))
        A2 = SparseTensor(torch.randn(6), torch.tensor([0, 0, 1, 1, 2, 2]), 
                          torch.tensor([0, 1, 0, 1, 1, 2]), (3, 3))
        
        stl = SparseTensorList([A1, A2])
        sizes = stl.block_sizes
        
        block_diag = stl.to_block_diagonal()
        recovered = SparseTensorList.from_block_diagonal(block_diag, sizes)
        
        assert len(recovered) == len(stl)
        for orig, rec in zip(stl, recovered):
            assert orig.sparse_shape == rec.sparse_shape
            assert orig.nnz == rec.nnz


class TestSparseTensorListProperties:
    """Tests for SparseTensorList properties."""
    
    def test_block_sizes(self):
        """Test block_sizes property."""
        A1 = SparseTensor(torch.ones(1), torch.tensor([0]), torch.tensor([0]), (3, 3))
        A2 = SparseTensor(torch.ones(1), torch.tensor([0]), torch.tensor([0]), (5, 5))
        
        stl = SparseTensorList([A1, A2])
        assert stl.block_sizes == [(3, 3), (5, 5)]
    
    def test_total_nnz(self):
        """Test total_nnz property."""
        A1 = SparseTensor(torch.ones(3), torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2]), (3, 3))
        A2 = SparseTensor(torch.ones(5), torch.tensor([0, 1, 2, 3, 4]), 
                          torch.tensor([0, 1, 2, 3, 4]), (5, 5))
        
        stl = SparseTensorList([A1, A2])
        assert stl.total_nnz == 8
    
    def test_total_shape(self):
        """Test total_shape property."""
        A1 = SparseTensor(torch.ones(1), torch.tensor([0]), torch.tensor([0]), (3, 3))
        A2 = SparseTensor(torch.ones(1), torch.tensor([0]), torch.tensor([0]), (5, 5))
        
        stl = SparseTensorList([A1, A2])
        assert stl.total_shape == (8, 8)


def run_tests_manually():
    """Run all tests without pytest."""
    print("=" * 70)
    print("Running SparseTensorList tests")
    print("=" * 70)
    
    # TestConnectedComponents
    print("\n--- TestConnectedComponents ---")
    test_cc = TestConnectedComponents()
    test_cc.test_single_component()
    print("  ✓ test_single_component")
    test_cc.test_two_components()
    print("  ✓ test_two_components")
    test_cc.test_three_components()
    print("  ✓ test_three_components")
    test_cc.test_has_isolated_components()
    print("  ✓ test_has_isolated_components")
    
    # TestToConnectedComponents
    print("\n--- TestToConnectedComponents ---")
    test_tcc = TestToConnectedComponents()
    test_tcc.test_split_two_components()
    print("  ✓ test_split_two_components")
    test_tcc.test_single_component_returns_list()
    print("  ✓ test_single_component_returns_list")
    
    # TestSparseTensorListOperations
    print("\n--- TestSparseTensorListOperations ---")
    test_ops = TestSparseTensorListOperations()
    sample = test_ops.sample_list()
    test_ops.test_matmul_list(sample)
    print("  ✓ test_matmul_list")
    test_ops.test_matmul_broadcast(sample)
    print("  ✓ test_matmul_broadcast")
    test_ops.test_add(sample)
    print("  ✓ test_add")
    test_ops.test_mul_scalar(sample)
    print("  ✓ test_mul_scalar")
    test_ops.test_neg(sample)
    print("  ✓ test_neg")
    test_ops.test_sum(sample)
    print("  ✓ test_sum")
    
    # TestBlockDiagonal
    print("\n--- TestBlockDiagonal ---")
    test_bd = TestBlockDiagonal()
    test_bd.test_to_block_diagonal()
    print("  ✓ test_to_block_diagonal")
    test_bd.test_from_block_diagonal()
    print("  ✓ test_from_block_diagonal")
    test_bd.test_roundtrip()
    print("  ✓ test_roundtrip")
    
    # TestSparseTensorListProperties
    print("\n--- TestSparseTensorListProperties ---")
    test_props = TestSparseTensorListProperties()
    test_props.test_block_sizes()
    print("  ✓ test_block_sizes")
    test_props.test_total_nnz()
    print("  ✓ test_total_nnz")
    test_props.test_total_shape()
    print("  ✓ test_total_shape")
    
    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)


if __name__ == "__main__":
    if HAS_PYTEST:
        pytest.main([__file__, "-v"])
    else:
        run_tests_manually()

