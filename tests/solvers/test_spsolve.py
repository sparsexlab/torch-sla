"""
Tests for torch-sla sparse linear solvers

Tests all available backends:
- CPU: cg, bicgstab
"""

import pytest
import torch
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import cg as sp_cg
from itertools import product
import sys

sys.path.insert(0, "..")
from torch_sla import (
    spsolve,
    spsolve_coo,
    spsolve_csr,
    get_available_backends,
    is_cudss_available,
)


# ============================================================================
# CPU Backend Tests
# ============================================================================

@pytest.mark.parametrize(
    ['n', 'method'],
    product([16, 64, 256], ['cg', 'bicgstab'])
    )
def test_spsolve_cpu(n, method):
    """Test CPU iterative solvers"""
    # Create SPD matrix
    A = torch.rand(n, n).double()
    A = A @ A.T + torch.eye(n).double() * n  # Ensure positive definite
    A[A.abs() < 0.3] = 0
    A = A.to_sparse_coo()

    b = torch.randn(n).double()
    x = spsolve(
        A.values(), A.indices()[0], A.indices()[1], A.shape, b,
        method=method, atol=1e-10, maxiter=10000
    )

    # Compare with scipy
    A_scipy = csc_matrix(A.to_dense().numpy())
    b_scipy = b.numpy()
    x_ref, _ = sp_cg(A_scipy, b_scipy, atol=1e-10, maxiter=10000)
    x_ref = torch.tensor(x_ref)

    torch.testing.assert_close(x, x_ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize(
    ['n', 'method'],
    product([16, 64], ['cg', 'bicgstab'])
    )
def test_spsolve_gradient_cpu(n, method):
    """Test gradient computation for CPU solvers"""
    # Create SPD matrix
    A_dense = torch.rand(n, n).double()
    A_dense = A_dense @ A_dense.T + torch.eye(n).double() * n
    A_dense[A_dense.abs() < 0.3] = 0
    A = A_dense.to_sparse_coo()

    b = torch.randn(n).double()
    b_dense = b.clone()

    val = A.values().clone()
    val.requires_grad_(True)
    b.requires_grad_(True)
    A_dense.requires_grad_(True)
    b_dense.requires_grad_(True)

    # Sparse solve
    x = spsolve(
        val, A.indices()[0], A.indices()[1], A.shape, b,
        method=method, atol=1e-10, maxiter=10000
    )
    x.sum().backward()

    # Dense solve for reference
    x2 = torch.linalg.solve(A_dense, b_dense)
    x2.sum().backward()

    # Compare gradients
    A_grad = torch.sparse_coo_tensor(A.indices(), val.grad, A.shape).to_dense()

    torch.testing.assert_close(x, x2, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(A_dense.grad, A_grad, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(b_dense.grad, b.grad, rtol=1e-3, atol=1e-3)


# ============================================================================
# cuDSS Backend Tests
# ============================================================================

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize(
    ['n', 'method'],
    product([16, 64, 256], ['cudss', 'cudss_lu', 'cudss_cholesky', 'cudss_ldlt'])
)
def test_spsolve_cudss(n, method):
    """Test cuDSS direct solvers"""
    if not is_cudss_available():
        pytest.skip("cuDSS backend not available")

    # Create SPD matrix (required for Cholesky/LDLT)
    A = torch.rand(n, n, dtype=torch.float64, device='cuda')
    A = A @ A.T + torch.eye(n, dtype=torch.float64, device='cuda') * n
    A[A.abs() < 0.3] = 0
    A = A.to_sparse_coo()

    b = torch.randn(n, dtype=torch.float64, device='cuda')

    matrix_type = "general"
    if method in ['cudss_cholesky']:
        matrix_type = "spd"
    elif method in ['cudss_ldlt']:
        matrix_type = "symmetric"

    x = spsolve(
        A.values(), A.indices()[0], A.indices()[1], A.shape, b,
        method=method, matrix_type=matrix_type
    )

    # Verify solution: Ax ≈ b
    A_dense = A.to_dense()
    residual = torch.mv(A_dense, x) - b
    relative_error = residual.norm() / b.norm()

    assert relative_error < 1e-6, f"Relative error too large: {relative_error}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize('method', ['cudss_lu', 'cudss_cholesky', 'cudss_ldlt'])
def test_spsolve_gradient_cudss(method):
    """Test gradient computation for cuDSS solvers"""
    if not is_cudss_available():
        pytest.skip("cuDSS backend not available")

    n = 32

    # Create SPD matrix
    A_dense = torch.rand(n, n, dtype=torch.float64, device='cuda')
    A_dense = A_dense @ A_dense.T + torch.eye(n, dtype=torch.float64, device='cuda') * n
    A_dense[A_dense.abs() < 0.3] = 0
    A = A_dense.to_sparse_coo()

    b = torch.randn(n, dtype=torch.float64, device='cuda')
    b_dense = b.clone()

    val = A.values().clone()
    val.requires_grad_(True)
    b.requires_grad_(True)
    A_dense.requires_grad_(True)
    b_dense.requires_grad_(True)

    # Sparse solve
    x = spsolve(val, A.indices()[0], A.indices()[1], A.shape, b, method=method)
    x.sum().backward()

    # Dense solve for reference
    x2 = torch.linalg.solve(A_dense, b_dense)
    x2.sum().backward()

    torch.testing.assert_close(x, x2, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(b_dense.grad, b.grad, rtol=1e-3, atol=1e-3)


# ============================================================================
# Convenience Function Tests
# ============================================================================

def test_spsolve_coo():
    """Test spsolve_coo convenience function"""
    n = 32
    A = torch.rand(n, n).double()
    A = A @ A.T + torch.eye(n).double() * n
    A[A.abs() < 0.3] = 0
    A_sparse = A.to_sparse_coo()

    b = torch.randn(n).double()

    x = spsolve_coo(A_sparse, b, method='bicgstab')

    # Verify
    residual = torch.mv(A, x) - b
    relative_error = residual.norm() / b.norm()
    assert relative_error < 1e-3


def test_spsolve_csr():
    """Test spsolve_csr convenience function"""
    n = 32
    A = torch.rand(n, n).double()
    A = A @ A.T + torch.eye(n).double() * n
    A[A.abs() < 0.3] = 0
    A_csr = A.to_sparse_csr()

    b = torch.randn(n).double()

    x = spsolve_csr(A_csr, b, method='bicgstab')

    # Verify
    residual = torch.mv(A, x) - b
    relative_error = residual.norm() / b.norm()
    assert relative_error < 1e-3


# ============================================================================
# Multi-RHS Tests
# ============================================================================

def _make_spd_sparse(n):
    """Create a sparse SPD matrix and return (val, row, col, shape, A_dense)."""
    A_dense = torch.rand(n, n).double()
    A_dense = A_dense @ A_dense.T + torch.eye(n).double() * n
    A_dense[A_dense.abs() < 0.3] = 0
    A = A_dense.to_sparse_coo().coalesce()
    return A.values(), A.indices()[0], A.indices()[1], A.shape, A_dense


@pytest.mark.parametrize('method', ['lu', 'cg', 'bicgstab'])
def test_spsolve_multi_rhs(method):
    """Test spsolve with 2D b (multiple right-hand sides)"""
    n, k = 32, 5
    val, row, col, shape, A_dense = _make_spd_sparse(n)
    b = torch.randn(n, k).double()

    x = spsolve(val, row, col, shape, b, backend='scipy', method=method, atol=1e-10)

    assert x.shape == (n, k), f"Expected shape ({n}, {k}), got {x.shape}"

    # Verify each column: Ax_k ≈ b_k
    for j in range(k):
        residual = A_dense @ x[:, j] - b[:, j]
        relative_error = residual.norm() / b[:, j].norm()
        assert relative_error < 1e-3, f"Column {j}: relative error {relative_error}"


@pytest.mark.parametrize('method', ['lu', 'cg'])
def test_spsolve_multi_rhs_gradient(method):
    """Test gradient computation for multi-RHS solve"""
    n, k = 16, 3
    _, row, col, shape, _ = _make_spd_sparse(n)
    # Re-create with fresh values for grad
    A_dense = torch.rand(n, n).double()
    A_dense = A_dense @ A_dense.T + torch.eye(n).double() * n
    A_dense[A_dense.abs() < 0.3] = 0
    A = A_dense.to_sparse_coo().coalesce()

    val = A.values().clone().requires_grad_(True)
    b = torch.randn(n, k).double().requires_grad_(True)
    A_dense2 = A_dense.clone().requires_grad_(True)
    b2 = b.detach().clone().requires_grad_(True)

    # Sparse multi-RHS solve
    x = spsolve(val, A.indices()[0], A.indices()[1], A.shape, b,
                backend='scipy', method=method, atol=1e-10)
    x.sum().backward()

    # Dense reference
    x_ref = torch.linalg.solve(A_dense2, b2)
    x_ref.sum().backward()

    torch.testing.assert_close(x, x_ref, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(b.grad, b2.grad, rtol=1e-3, atol=1e-3)


def test_sparse_tensor_multi_rhs():
    """Test SparseTensor.solve() with 2D b"""
    from torch_sla import SparseTensor

    n, k = 32, 4
    A_dense = torch.rand(n, n).double()
    A_dense = A_dense @ A_dense.T + torch.eye(n).double() * n
    A_dense[A_dense.abs() < 0.3] = 0
    A_coo = A_dense.to_sparse_coo().coalesce()

    A = SparseTensor(A_coo.values(), A_coo.indices()[0], A_coo.indices()[1], A_coo.shape)
    b = torch.randn(n, k).double()

    x = A.solve(b)
    assert x.shape == (n, k)

    # Verify
    for j in range(k):
        residual = A_dense @ x[:, j] - b[:, j]
        assert residual.norm() / b[:, j].norm() < 1e-3


# ============================================================================
# Backend Discovery Tests
# ============================================================================

def test_get_available_backends():
    """Test backend discovery"""
    backends = get_available_backends()
    assert isinstance(backends, list)
    # CPU backends should always be available
    # (assuming compilation succeeded)


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    print("Available backends:", get_available_backends())
    print("cuDSS available:", is_cudss_available())

    # Run a simple test
    test_spsolve_cpu(32, 'bicgstab')
    print("CPU test passed!")

    test_spsolve_gradient_cpu(32, 'cg')
    print("CPU gradient test passed!")


    if torch.cuda.is_available() and is_cudss_available():
        test_spsolve_cudss(32, 'cudss_lu')
        print("cuDSS test passed!")
