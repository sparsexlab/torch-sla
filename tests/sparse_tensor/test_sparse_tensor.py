"""
Tests for SparseTensor class with batched and block support.

Tests cover:
- Basic SparseTensor creation and properties
- Batched SparseTensor operations
- SparseTensorList operations
- Matrix multiplication (Sparse@Dense, Dense@Sparse, Sparse@Sparse)
- Sparse gradient for Sparse@Sparse
- Property detection (symmetry, positive definiteness)
- Eigenvalue and SVD operations
"""

import pytest
import torch
import numpy as np
from itertools import product
import sys

sys.path.insert(0, "..")
from torch_sla import (
    SparseTensor,
    SparseTensorList,
    LUFactorization,
)


def create_spd_coo(n: int, dtype=torch.float64, device='cpu'):
    """Create SPD matrix in COO format."""
    A = torch.rand(n, n, dtype=dtype, device=device)
    A = A @ A.T + torch.eye(n, dtype=dtype, device=device) * n
    A[A.abs() < 0.3] = 0
    A_sparse = A.to_sparse_coo()
    return (
        A_sparse._values(),
        A_sparse._indices()[0],
        A_sparse._indices()[1],
        (n, n)
    )


def create_tridiagonal_spd(n: int, dtype=torch.float64, device='cpu'):
    """Create tridiagonal SPD matrix (like 1D Poisson)."""
    rows, cols, vals = [], [], []
    for i in range(n):
        # Diagonal
        rows.append(i)
        cols.append(i)
        vals.append(4.0)
        # Off-diagonals
        if i > 0:
            rows.append(i)
            cols.append(i - 1)
            vals.append(-1.0)
        if i < n - 1:
            rows.append(i)
            cols.append(i + 1)
            vals.append(-1.0)
    
    return (
        torch.tensor(vals, dtype=dtype, device=device),
        torch.tensor(rows, device=device),
        torch.tensor(cols, device=device),
        (n, n)
    )


# ============================================================================
# Basic SparseTensor Tests
# ============================================================================

class TestSparseTensorBasic:
    """Test basic SparseTensor functionality."""
    
    def test_create_2d(self):
        """Test creating a simple 2D SparseTensor."""
        val, row, col, shape = create_tridiagonal_spd(10)
        A = SparseTensor(val, row, col, shape)
        
        assert A.shape == (10, 10)
        assert A.sparse_shape == (10, 10)
        assert A.batch_shape == ()
        assert A.block_shape == ()
        assert A.is_batched == False
        assert A.is_block == False
        assert A.is_square == True
        assert A.batch_size == 1
    
    def test_create_from_dense(self):
        """Test creating SparseTensor from dense matrix."""
        A_dense = torch.randn(10, 10, dtype=torch.float64)
        A_dense[A_dense.abs() < 0.5] = 0
        
        A = SparseTensor.from_dense(A_dense)
        
        assert A.shape == (10, 10)
        assert A.nnz > 0
        
        # Verify values match
        A_reconstructed = A.to_dense()
        torch.testing.assert_close(A_reconstructed, A_dense)
    
    def test_create_from_torch_sparse(self):
        """Test creating SparseTensor from PyTorch sparse tensor."""
        A_dense = torch.randn(10, 10, dtype=torch.float64)
        A_dense[A_dense.abs() < 0.5] = 0
        A_coo = A_dense.to_sparse_coo()

        A = SparseTensor.from_torch_sparse(A_coo)

        assert A.shape == (10, 10)
        torch.testing.assert_close(A.to_dense(), A_dense)

    def test_from_dense_preserves_grad(self):
        """from_dense must keep the autograd graph so downstream .solve() is differentiable."""
        Es = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True)
        dense = torch.stack([
            torch.stack([Es[0],           torch.tensor(-1.0, dtype=torch.float64), torch.tensor(0.0, dtype=torch.float64)]),
            torch.stack([torch.tensor(-1.0, dtype=torch.float64), Es[1],           torch.tensor(-1.0, dtype=torch.float64)]),
            torch.stack([torch.tensor(0.0, dtype=torch.float64), torch.tensor(-1.0, dtype=torch.float64), Es[2]]),
        ])

        A = SparseTensor.from_dense(dense)
        assert A.values.requires_grad, "from_dense dropped the autograd graph"

        b = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        x = A.solve(b)
        x.sum().backward()
        assert Es.grad is not None
        assert torch.isfinite(Es.grad).all()

    def test_from_torch_sparse_preserves_grad(self):
        """from_torch_sparse must keep the autograd graph (regression test for GitHub issue)."""
        Es = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True)
        dense = torch.stack([
            torch.stack([Es[0],           torch.tensor(-1.0, dtype=torch.float64), torch.tensor(0.0, dtype=torch.float64)]),
            torch.stack([torch.tensor(-1.0, dtype=torch.float64), Es[1],           torch.tensor(-1.0, dtype=torch.float64)]),
            torch.stack([torch.tensor(0.0, dtype=torch.float64), torch.tensor(-1.0, dtype=torch.float64), Es[2]]),
        ])
        coo = dense.to_sparse()

        A = SparseTensor.from_torch_sparse(coo)
        assert A.values.requires_grad, "from_torch_sparse dropped the autograd graph"

        b = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        x = A.solve(b)
        x.sum().backward()
        assert Es.grad is not None
        assert torch.isfinite(Es.grad).all()

    def test_to_device_and_dtype(self):
        """Test moving tensor to device and changing dtype."""
        val, row, col, shape = create_tridiagonal_spd(10)
        A = SparseTensor(val, row, col, shape)
        
        # Test dtype conversion
        A_float32 = A.to(dtype=torch.float32)
        assert A_float32.dtype == torch.float32
        
        # Test float/double methods
        A_float = A.float()
        assert A_float.dtype == torch.float32
        
        A_double = A.double()
        assert A_double.dtype == torch.float64
        
        # Test to with both device and dtype
        A_cpu_f32 = A.to('cpu', torch.float32)
        assert A_cpu_f32.device == torch.device('cpu')
        assert A_cpu_f32.dtype == torch.float32
    
    def test_transpose(self):
        """Test matrix transpose."""
        val, row, col, shape = create_tridiagonal_spd(10)
        A = SparseTensor(val, row, col, shape)
        
        A_T = A.T()
        assert A_T.shape == (10, 10)
        
        # Verify transpose
        torch.testing.assert_close(A.to_dense().T, A_T.to_dense())
    
    def test_solve_basic(self):
        """Test basic solve functionality."""
        val, row, col, shape = create_tridiagonal_spd(10)
        A = SparseTensor(val, row, col, shape)
        
        b = torch.randn(10, dtype=torch.float64)
        x = A.solve(b)
        
        # Verify solution
        residual = A @ x - b
        assert residual.norm() / b.norm() < 1e-6
    
    def test_norm(self):
        """Test matrix norm computation."""
        val, row, col, shape = create_tridiagonal_spd(10)
        A = SparseTensor(val, row, col, shape)
        
        # Frobenius norm
        norm_fro = A.norm('fro')
        assert norm_fro > 0
        
        # Compare with torch
        expected = torch.linalg.norm(A.to_dense(), 'fro')
        torch.testing.assert_close(norm_fro, expected)
    
    def test_matvec(self):
        """Test matrix-vector multiplication."""
        val, row, col, shape = create_tridiagonal_spd(10)
        A = SparseTensor(val, row, col, shape)
        
        x = torch.randn(10, dtype=torch.float64)
        y = A @ x
        
        # Compare with dense
        y_expected = A.to_dense() @ x
        torch.testing.assert_close(y, y_expected)


# ============================================================================
# Matrix Multiplication Tests
# ============================================================================

class TestMatmul:
    """Test matrix multiplication operations."""
    
    def test_sparse_dense_vector(self):
        """Test Sparse @ Dense vector."""
        val, row, col, shape = create_tridiagonal_spd(10)
        A = SparseTensor(val, row, col, shape)
        
        x = torch.randn(10, dtype=torch.float64)
        y = A @ x
        y_expected = A.to_dense() @ x
        
        torch.testing.assert_close(y, y_expected)
    
    def test_sparse_dense_matrix(self):
        """Test Sparse @ Dense matrix."""
        val, row, col, shape = create_tridiagonal_spd(10)
        A = SparseTensor(val, row, col, shape)
        
        X = torch.randn(10, 5, dtype=torch.float64)
        Y = A @ X
        Y_expected = A.to_dense() @ X
        
        torch.testing.assert_close(Y, Y_expected)
    
    def test_dense_sparse_vector(self):
        """Test Dense @ Sparse (vector)."""
        val, row, col, shape = create_tridiagonal_spd(10)
        A = SparseTensor(val, row, col, shape)
        
        x = torch.randn(10, dtype=torch.float64)
        y = x @ A
        y_expected = x @ A.to_dense()
        
        torch.testing.assert_close(y, y_expected)
    
    def test_dense_sparse_matrix(self):
        """Test Dense @ Sparse (matrix)."""
        val, row, col, shape = create_tridiagonal_spd(10)
        A = SparseTensor(val, row, col, shape)
        
        X = torch.randn(5, 10, dtype=torch.float64)
        Y = X @ A
        Y_expected = X @ A.to_dense()
        
        torch.testing.assert_close(Y, Y_expected)
    
    def test_sparse_sparse(self):
        """Test Sparse @ Sparse."""
        val, row, col, shape = create_tridiagonal_spd(10)
        A = SparseTensor(val, row, col, shape)
        B = SparseTensor(val.clone(), row.clone(), col.clone(), shape)
        
        C = A @ B
        
        assert isinstance(C, SparseTensor)
        
        C_expected = A.to_dense() @ B.to_dense()
        torch.testing.assert_close(C.to_dense(), C_expected)
    
    def test_sparse_sparse_gradient(self):
        """Test Sparse @ Sparse with sparse gradient."""
        val_a = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64, requires_grad=True)
        row_a = torch.tensor([0, 0, 1, 1])
        col_a = torch.tensor([0, 1, 0, 1])
        A = SparseTensor(val_a, row_a, col_a, (2, 2))
        
        val_b = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
        row_b = torch.tensor([0, 1])
        col_b = torch.tensor([0, 1])
        B = SparseTensor(val_b, row_b, col_b, (2, 2))
        
        C = A @ B
        
        # Compute loss
        loss = C.values.sum()
        loss.backward()
        
        # Check that gradients exist and have the right shape
        assert val_a.grad is not None
        assert val_a.grad.shape == val_a.shape
        assert val_b.grad is not None
        assert val_b.grad.shape == val_b.shape
    
    def test_sparse_sparse_gradient_memory(self):
        """Test that Sparse @ Sparse gradient is memory efficient (sparse)."""
        N = 100
        density = 0.1
        nnz = int(N * N * density)
        
        # Create random sparse matrices
        val_a = torch.randn(nnz, dtype=torch.float64, requires_grad=True)
        row_a = torch.randint(0, N, (nnz,))
        col_a = torch.randint(0, N, (nnz,))
        A = SparseTensor(val_a, row_a, col_a, (N, N))
        
        val_b = torch.randn(nnz, dtype=torch.float64, requires_grad=True)
        row_b = torch.randint(0, N, (nnz,))
        col_b = torch.randint(0, N, (nnz,))
        B = SparseTensor(val_b, row_b, col_b, (N, N))
        
        C = A @ B
        loss = C.values.sum()
        loss.backward()
        
        # Gradients should be sparse (same size as values)
        assert val_a.grad.shape == (nnz,)  # Sparse gradient
        assert val_b.grad.shape == (nnz,)


# ============================================================================
# Batched Operations Tests
# ============================================================================

class TestBatchedOperations:
    """Test batched Sparse @ Dense operations."""
    
    def test_batched_sparse_dense_vector(self):
        """Test batched Sparse @ Dense with shared vector."""
        val, row, col, shape = create_tridiagonal_spd(10)
        batch_size = 4
        
        val_batch = val.unsqueeze(0).expand(batch_size, -1).clone()
        A = SparseTensor(val_batch, row, col, (batch_size, 10, 10))
        
        x = torch.randn(10, dtype=torch.float64)
        y = A @ x
        
        assert y.shape == (batch_size, 10)
        
        for i in range(batch_size):
            A_i = SparseTensor(val_batch[i], row, col, (10, 10))
            y_expected = A_i @ x
            torch.testing.assert_close(y[i], y_expected)
    
    def test_batched_sparse_dense_batched(self):
        """Test batched Sparse @ batched Dense."""
        val, row, col, shape = create_tridiagonal_spd(10)
        batch_size = 4
        
        val_batch = val.unsqueeze(0).expand(batch_size, -1).clone()
        A = SparseTensor(val_batch, row, col, (batch_size, 10, 10))
        
        x = torch.randn(batch_size, 10, dtype=torch.float64)
        y = A @ x
        
        assert y.shape == (batch_size, 10)
    
    def test_batched_dense_sparse(self):
        """Test batched Dense @ Sparse."""
        val, row, col, shape = create_tridiagonal_spd(10)
        batch_size = 4
        
        val_batch = val.unsqueeze(0).expand(batch_size, -1).clone()
        A = SparseTensor(val_batch, row, col, (batch_size, 10, 10))
        
        x = torch.randn(batch_size, 10, dtype=torch.float64)
        y = x @ A
        
        assert y.shape == (batch_size, 10)


# ============================================================================
# Batched SparseTensor Tests
# ============================================================================

class TestSparseTensorBatched:
    """Test batched SparseTensor functionality."""
    
    def test_batched_creation(self):
        """Test creating batched SparseTensor."""
        val, row, col, shape = create_tridiagonal_spd(10)
        batch_size = 4
        
        val_batch = val.unsqueeze(0).expand(batch_size, -1).clone()
        A = SparseTensor(val_batch, row, col, (batch_size, 10, 10))
        
        assert A.shape == (batch_size, 10, 10)
        assert A.batch_shape == (batch_size,)
        assert A.sparse_shape == (10, 10)
        assert A.is_batched == True
        assert A.batch_size == batch_size
    
    def test_multi_batch_creation(self):
        """Test creating multi-dimensional batched SparseTensor."""
        val, row, col, shape = create_tridiagonal_spd(10)
        
        val_batch = val.unsqueeze(0).unsqueeze(0).expand(2, 3, -1).clone()
        A = SparseTensor(val_batch, row, col, (2, 3, 10, 10))
        
        assert A.shape == (2, 3, 10, 10)
        assert A.batch_shape == (2, 3)
        assert A.batch_size == 6
    
    def test_batched_solve(self):
        """Test batched solve."""
        val, row, col, shape = create_tridiagonal_spd(10)
        batch_size = 4
        
        val_batch = val.unsqueeze(0).expand(batch_size, -1).clone()
        # Vary values slightly
        for i in range(batch_size):
            val_batch[i] = val * (1 + 0.1 * i)
        
        A = SparseTensor(val_batch, row, col, (batch_size, 10, 10))
        
        b = torch.randn(batch_size, 10, dtype=torch.float64)
        x = A.solve(b)
        
        assert x.shape == (batch_size, 10)
        
        # Verify each solution
        for i in range(batch_size):
            A_i = SparseTensor(val_batch[i], row, col, (10, 10))
            residual = A_i @ x[i] - b[i]
            assert residual.norm() / b[i].norm() < 1e-6
    
    def test_batched_norm(self):
        """Test batched norm computation."""
        val, row, col, shape = create_tridiagonal_spd(10)
        batch_size = 4
        
        val_batch = val.unsqueeze(0).expand(batch_size, -1).clone()
        for i in range(batch_size):
            val_batch[i] = val * (1 + 0.1 * i)
        
        A = SparseTensor(val_batch, row, col, (batch_size, 10, 10))
        
        norms = A.norm('fro')
        
        assert norms.shape == (batch_size,)
        assert (norms > 0).all()
    
    def test_batched_eigsh(self):
        """Test batched eigenvalue computation."""
        val, row, col, shape = create_tridiagonal_spd(10)
        batch_size = 3
        
        val_batch = val.unsqueeze(0).expand(batch_size, -1).clone()
        for i in range(batch_size):
            val_batch[i] = val * (1 + 0.1 * i)
        
        A = SparseTensor(val_batch, row, col, (batch_size, 10, 10))
        
        eigenvalues, eigenvectors = A.eigsh(k=3, which='LM')
        
        assert eigenvalues.shape == (batch_size, 3)
        assert eigenvectors.shape == (batch_size, 10, 3)


# ============================================================================
# Property Detection Tests
# ============================================================================

class TestPropertyDetection:
    """Test symmetry and positive definiteness detection."""
    
    def test_is_symmetric_true(self):
        """Test symmetric matrix detection."""
        val, row, col, shape = create_tridiagonal_spd(10)
        A = SparseTensor(val, row, col, shape)
        
        result = A.is_symmetric()
        assert isinstance(result, torch.Tensor)
        assert result.item() == True
    
    def test_is_symmetric_false(self):
        """Test non-symmetric matrix detection."""
        # Create non-symmetric matrix
        val = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        row = torch.tensor([0, 0, 1, 1])
        col = torch.tensor([0, 1, 0, 1])
        
        A = SparseTensor(val, row, col, (2, 2))
        
        # A[0,1] = 2, A[1,0] = 3, so not symmetric
        result = A.is_symmetric()
        assert isinstance(result, torch.Tensor)
        assert result.item() == False
    
    def test_is_positive_definite_gershgorin(self):
        """Test positive definite check using Gershgorin."""
        val, row, col, shape = create_tridiagonal_spd(10)
        A = SparseTensor(val, row, col, shape)
        
        # Tridiagonal with 4 on diagonal and -1 off-diagonal is diagonally dominant
        result = A.is_positive_definite(method='gershgorin')
        assert isinstance(result, torch.Tensor)
        assert result.item() == True
    
    def test_is_positive_definite_cholesky(self):
        """Test positive definite check using Cholesky."""
        val, row, col, shape = create_tridiagonal_spd(10)
        A = SparseTensor(val, row, col, shape)
        
        result = A.is_positive_definite(method='cholesky')
        assert isinstance(result, torch.Tensor)
        assert result.item() == True
    
    def test_is_positive_definite_false(self):
        """Test indefinite matrix detection."""
        # Create symmetric but indefinite matrix
        val = torch.tensor([1.0, -2.0, -2.0, 1.0], dtype=torch.float64)
        row = torch.tensor([0, 0, 1, 1])
        col = torch.tensor([0, 1, 0, 1])
        
        A = SparseTensor(val, row, col, (2, 2))
        
        result_gershgorin = A.is_positive_definite(method='gershgorin')
        result_cholesky = A.is_positive_definite(method='cholesky')
        
        assert result_gershgorin.item() == False
        assert result_cholesky.item() == False
    
    def test_batched_is_symmetric(self):
        """Test batched symmetry detection."""
        val, row, col, shape = create_tridiagonal_spd(10)
        batch_size = 4
        
        val_batch = val.unsqueeze(0).expand(batch_size, -1).clone()
        A = SparseTensor(val_batch, row, col, (batch_size, 10, 10))
        
        result = A.is_symmetric()
        
        assert result.shape == (batch_size,)
        assert result.all().item() == True
    
    def test_batched_is_positive_definite(self):
        """Test batched positive definiteness detection."""
        val, row, col, shape = create_tridiagonal_spd(10)
        batch_size = 4
        
        val_batch = val.unsqueeze(0).expand(batch_size, -1).clone()
        A = SparseTensor(val_batch, row, col, (batch_size, 10, 10))
        
        result = A.is_positive_definite()
        
        assert result.shape == (batch_size,)
        assert result.all().item() == True
    
    def test_is_symmetric_caching(self):
        """Test that is_symmetric results are cached."""
        val, row, col, shape = create_tridiagonal_spd(10)
        A = SparseTensor(val, row, col, shape)
        
        # First call computes
        result1 = A.is_symmetric()
        # Second call should use cache
        result2 = A.is_symmetric()
        
        assert result1.item() == result2.item()
        
        # Force recompute
        result3 = A.is_symmetric(force_recompute=True)
        assert result1.item() == result3.item()


# ============================================================================
# Eigenvalue and SVD Tests
# ============================================================================

class TestEigenvalueSVD:
    """Test eigenvalue and SVD operations."""
    
    def test_eigsh_basic(self):
        """Test basic eigsh computation."""
        val, row, col, shape = create_tridiagonal_spd(20)
        A = SparseTensor(val, row, col, shape)
        
        eigenvalues, eigenvectors = A.eigsh(k=5, which='LM')
        
        assert eigenvalues.shape == (5,)
        assert eigenvectors.shape == (20, 5)
        
        # Eigenvalues should be positive for SPD matrix
        assert (eigenvalues > 0).all()
    
    def test_svd_basic(self):
        """Test basic SVD computation."""
        val, row, col, shape = create_tridiagonal_spd(20)
        A = SparseTensor(val, row, col, shape)

        U, S, Vt = A.svd(k=5)

        assert U.shape == (20, 5)
        assert S.shape == (5,)
        assert Vt.shape == (5, 20)

        # Singular values should be positive
        assert (S > 0).all()

    def test_svd_sigma_backward(self):
        """``SvdAdjoint`` propagates ∂L/∂σ_i = u_i v_i^T at A's nnz pattern.

        Uses ``Synthetic['anisotropic_2d_64_eps_001']`` from
        torch_sla.datasets -- a real PDE matrix with non-degenerate
        spectrum so the comparison against dense torch.linalg.svd is
        not muddied by U/V rotations in degenerate subspaces.
        """
        from torch_sla.datasets import Synthetic

        bench = Synthetic["anisotropic_2d_64_eps_001"]
        n = int(bench.shape[0])
        k = 5

        # Ours: sparse SVD with SvdAdjoint backward
        val_g = bench.val.detach().clone().requires_grad_(True)
        A_sp = SparseTensor(val_g, bench.row, bench.col, bench.shape)
        _, S, _ = A_sp.svd(k=k)
        S.sum().backward()
        grad_ours = val_g.grad.clone()

        # Reference: dense torch.linalg.svd backward (sum of top-k σ).
        val_d = bench.val.detach().clone().requires_grad_(True)
        A_dense = torch.zeros(n, n, dtype=bench.val.dtype)
        A_dense[bench.row, bench.col] = val_d
        _, S_ref, _ = torch.linalg.svd(A_dense, full_matrices=False)
        S_ref[:k].sum().backward()
        grad_ref = val_d.grad.clone()

        # When σ values are non-degenerate, fp64 eps; gradient direction
        # is the meaningful invariant even in degenerate subspaces, so
        # cosine similarity > 0.999 is the right test.
        cos = torch.nn.functional.cosine_similarity(
            grad_ours.unsqueeze(0), grad_ref.unsqueeze(0)
        ).item()
        assert cos > 0.999, (
            f"σ gradient direction on {bench.name!r} differs from "
            f"dense ref: cos={cos:.4f}"
        )
    
    def test_condition_number(self):
        """Test condition number estimation."""
        val, row, col, shape = create_tridiagonal_spd(20)
        A = SparseTensor(val, row, col, shape)
        
        cond = A.condition_number(ord=2)
        
        assert cond > 1  # Condition number >= 1
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_eigsh_cuda(self):
        """Test eigsh on CUDA (uses LOBPCG)."""
        val, row, col, shape = create_tridiagonal_spd(50)
        A = SparseTensor(val.cuda(), row.cuda(), col.cuda(), shape)
        
        eigenvalues, eigenvectors = A.eigsh(k=5, which='LM')
        
        assert eigenvalues.device.type == 'cuda'
        assert eigenvalues.shape == (5,)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_svd_cuda(self):
        """Test SVD on CUDA (uses power iteration)."""
        val, row, col, shape = create_tridiagonal_spd(50)
        A = SparseTensor(val.cuda(), row.cuda(), col.cuda(), shape)
        
        U, S, Vt = A.svd(k=5)
        
        assert S.device.type == 'cuda'
        assert (S > 0).all()


# ============================================================================
# SparseTensorList Tests
# ============================================================================

class TestSparseTensorList:
    """Test SparseTensorList operations."""
    
    def test_creation(self):
        """Test creating SparseTensorList."""
        tensors = []
        for n in [5, 10, 15]:
            val, row, col, shape = create_tridiagonal_spd(n)
            tensors.append(SparseTensor(val, row, col, shape))
        
        lst = SparseTensorList(tensors)
        
        assert len(lst) == 3
        assert lst[0].shape == (5, 5)
        assert lst[-1].shape == (15, 15)
    
    def test_from_coo_list(self):
        """Test creating from COO list."""
        matrices = []
        for n in [5, 10, 15]:
            val, row, col, shape = create_tridiagonal_spd(n)
            matrices.append((val, row, col, shape))
        
        lst = SparseTensorList.from_coo_list(matrices)
        
        shapes = [t.shape for t in lst]
        assert shapes == [(5, 5), (10, 10), (15, 15)]
    
    def test_solve_batch(self):
        """Test batch solve with different layouts."""
        tensors = []
        b_list = []
        
        for n in [5, 10, 15]:
            val, row, col, shape = create_tridiagonal_spd(n)
            tensors.append(SparseTensor(val, row, col, shape))
            b_list.append(torch.randn(n, dtype=torch.float64))
        
        lst = SparseTensorList(tensors)
        x_list = lst.solve(b_list)
        
        assert len(x_list) == 3
        
        # Verify each
        for i, (A, x, b) in enumerate(zip(lst, x_list, b_list)):
            residual = A @ x - b
            assert residual.norm() / b.norm() < 1e-6
    
    def test_device_movement(self):
        """Test moving list to device."""
        tensors = []
        for n in [5, 10]:
            val, row, col, shape = create_tridiagonal_spd(n)
            tensors.append(SparseTensor(val, row, col, shape))
        
        lst = SparseTensorList(tensors)
        
        assert lst.device == torch.device('cpu')
        
        lst_cpu = lst.cpu()
        assert lst_cpu.device == torch.device('cpu')
    
    def test_norm_batch(self):
        """Test batch norm computation."""
        tensors = []
        for n in [5, 10, 15]:
            val, row, col, shape = create_tridiagonal_spd(n)
            tensors.append(SparseTensor(val, row, col, shape))
        
        lst = SparseTensorList(tensors)
        norms = lst.norm('fro')
        
        assert len(norms) == 3
        assert all(n > 0 for n in norms)
    
    def test_is_symmetric_batch(self):
        """Test batch symmetry detection."""
        tensors = []
        for n in [5, 10]:
            val, row, col, shape = create_tridiagonal_spd(n)
            tensors.append(SparseTensor(val, row, col, shape))
        
        lst = SparseTensorList(tensors)
        results = lst.is_symmetric()
        
        assert len(results) == 2
        assert all(r.item() == True for r in results)
    
    def test_is_positive_definite_batch(self):
        """Test batch positive definiteness detection."""
        tensors = []
        for n in [5, 10]:
            val, row, col, shape = create_tridiagonal_spd(n)
            tensors.append(SparseTensor(val, row, col, shape))
        
        lst = SparseTensorList(tensors)
        results = lst.is_positive_definite()
        
        assert len(results) == 2
        assert all(r.item() == True for r in results)


# ============================================================================
# LU Factorization Tests
# ============================================================================

class TestLUFactorization:
    """Test LU factorization for repeated solves."""
    
    def test_lu_factorization(self):
        """Test LU factorization creation and solve."""
        val, row, col, shape = create_tridiagonal_spd(10)
        A = SparseTensor(val, row, col, shape)
        
        lu = A.lu()
        
        assert isinstance(lu, LUFactorization)
        
        # Solve with LU
        b = torch.randn(10, dtype=torch.float64)
        x = lu.solve(b)
        
        # Verify
        residual = A @ x - b
        assert residual.norm() / b.norm() < 1e-6
    
    def test_lu_repeated_solves(self):
        """Test multiple solves with same LU factorization."""
        val, row, col, shape = create_tridiagonal_spd(10)
        A = SparseTensor(val, row, col, shape)
        
        lu = A.lu()
        
        # Solve multiple times
        for _ in range(5):
            b = torch.randn(10, dtype=torch.float64)
            x = lu.solve(b)
            
            residual = A @ x - b
            assert residual.norm() / b.norm() < 1e-6


# ============================================================================
# solve_batch Tests
# ============================================================================

class TestSolveBatch:
    """Test solve_batch for same-structure different-values solving."""
    
    def test_solve_batch_basic(self):
        """Test basic solve_batch."""
        val, row, col, shape = create_tridiagonal_spd(10)
        A = SparseTensor(val, row, col, shape)
        
        batch_size = 4
        val_batch = val.unsqueeze(0).expand(batch_size, -1).clone()
        for i in range(batch_size):
            val_batch[i] = val * (1 + 0.1 * i)
        
        b_batch = torch.randn(batch_size, 10, dtype=torch.float64)
        
        x_batch = A.solve_batch(val_batch, b_batch)
        
        assert x_batch.shape == (batch_size, 10)
        
        # Verify each solution
        for i in range(batch_size):
            A_i = SparseTensor(val_batch[i], row, col, (10, 10))
            residual = A_i @ x_batch[i] - b_batch[i]
            assert residual.norm() / b_batch[i].norm() < 1e-6


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_list_error(self):
        """Test that empty SparseTensorList raises error."""
        with pytest.raises(ValueError):
            SparseTensorList([])
    
    def test_non_square_solve_error(self):
        """Test that solve on non-square matrix raises error."""
        val = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        row = torch.tensor([0, 1, 2])
        col = torch.tensor([0, 0, 1])
        
        A = SparseTensor(val, row, col, (3, 2))
        
        with pytest.raises(ValueError):
            A.solve(torch.randn(3))
    
    def test_small_matrix(self):
        """Test operations on small matrix."""
        val = torch.tensor([2.0, -1.0, -1.0, 2.0], dtype=torch.float64)
        row = torch.tensor([0, 0, 1, 1])
        col = torch.tensor([0, 1, 0, 1])
        
        A = SparseTensor(val, row, col, (2, 2))
        
        # Test solve
        b = torch.tensor([1.0, 1.0], dtype=torch.float64)
        x = A.solve(b)
        assert x.shape == (2,)
        
        # Test symmetry
        is_sym = A.is_symmetric()
        assert is_sym.item() == True
        
        # Test positive definiteness
        is_pd = A.is_positive_definite()
        assert is_pd.item() == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
