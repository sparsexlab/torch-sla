"""
Tests for batch sparse linear solvers
"""

import pytest
import torch
import numpy as np
from itertools import product
import sys

sys.path.insert(0, "..")
from torch_sla import (
    spsolve_batch_same_layout,
    spsolve_batch_different_layout,
    spsolve_batch_coo_same_layout,
    spsolve_batch_coo_different_layout,
    ParallelBatchSolver,
    is_cudss_available,
)


def create_spd_sparse(n: int, density: float = 0.3, device: str = 'cpu'):
    """Create a sparse SPD matrix"""
    A = torch.rand(n, n, dtype=torch.float64, device=device)
    A = A @ A.T + torch.eye(n, dtype=torch.float64, device=device) * n
    A[A.abs() < (1 - density)] = 0
    return A.to_sparse_coo()


# ============================================================================
# Same Layout Batch Tests
# ============================================================================

@pytest.mark.parametrize('batch_size', [2, 5, 10])
@pytest.mark.parametrize('n', [16, 32])
def test_batch_same_layout_cpu(batch_size, n):
    """Test batch solve with same sparsity pattern on CPU"""
    # Create template matrix
    A_template = create_spd_sparse(n, density=0.3)
    row = A_template._indices()[0]
    col = A_template._indices()[1]
    nnz = A_template._nnz()
    shape = tuple(A_template.shape)
    
    # Create batch of values and RHS
    val_batch = torch.randn(batch_size, nnz, dtype=torch.float64)
    # Make each matrix SPD
    for i in range(batch_size):
        A_dense = torch.zeros(n, n, dtype=torch.float64)
        A_dense[row, col] = val_batch[i]
        A_dense = A_dense @ A_dense.T + torch.eye(n, dtype=torch.float64) * n
        val_batch[i] = A_dense[row, col]
    
    b_batch = torch.randn(batch_size, n, dtype=torch.float64)
    
    # Solve
    x_batch = spsolve_batch_same_layout(
        val_batch, row, col, shape, b_batch,
        method='bicgstab', atol=1e-10, maxiter=10000
    )
    
    # Verify each solution
    for i in range(batch_size):
        A_dense = torch.zeros(n, n, dtype=torch.float64)
        A_dense[row, col] = val_batch[i]
        residual = torch.mv(A_dense, x_batch[i]) - b_batch[i]
        relative_error = residual.norm() / b_batch[i].norm()
        assert relative_error < 1e-3, f"Batch {i}: relative error {relative_error}"


@pytest.mark.parametrize('batch_size', [2, 5])
def test_batch_same_layout_gradient(batch_size):
    """Test gradient computation for batch solve"""
    n = 16
    
    A_template = create_spd_sparse(n, density=0.3)
    row = A_template._indices()[0]
    col = A_template._indices()[1]
    nnz = A_template._nnz()
    shape = tuple(A_template.shape)
    
    # Create batch with gradient tracking
    val_batch = torch.randn(batch_size, nnz, dtype=torch.float64, requires_grad=True)
    b_batch = torch.randn(batch_size, n, dtype=torch.float64, requires_grad=True)
    
    # Forward
    x_batch = spsolve_batch_same_layout(
        val_batch, row, col, shape, b_batch,
        method='cg', atol=1e-10, maxiter=10000
    )
    
    # Backward
    loss = x_batch.sum()
    loss.backward()
    
    # Check gradients exist
    assert val_batch.grad is not None
    assert b_batch.grad is not None
    assert val_batch.grad.shape == val_batch.shape
    assert b_batch.grad.shape == b_batch.shape


# ============================================================================
# Different Layout Batch Tests
# ============================================================================

def test_batch_different_layout():
    """Test batch solve with different sparsity patterns"""
    matrices = []
    b_list = []
    
    for n in [16, 24, 32]:
        A = create_spd_sparse(n, density=0.3)
        val = A._values()
        row = A._indices()[0]
        col = A._indices()[1]
        shape = tuple(A.shape)
        matrices.append((val, row, col, shape))
        b_list.append(torch.randn(n, dtype=torch.float64))
    
    # Solve
    x_list = spsolve_batch_different_layout(
        matrices, b_list,
        method='bicgstab', atol=1e-10, maxiter=10000
    )
    
    # Verify
    assert len(x_list) == len(matrices)
    for i, ((val, row, col, shape), b, x) in enumerate(zip(matrices, b_list, x_list)):
        n = shape[0]
        A_dense = torch.zeros(n, n, dtype=torch.float64)
        A_dense[row, col] = val
        residual = torch.mv(A_dense, x) - b
        relative_error = residual.norm() / b.norm()
        assert relative_error < 1e-3, f"Matrix {i}: relative error {relative_error}"


# ============================================================================
# COO Interface Tests
# ============================================================================

def test_batch_coo_same_layout():
    """Test batch solve using COO tensor interface"""
    n = 32
    batch_size = 5
    
    A_template = create_spd_sparse(n, density=0.3)
    nnz = A_template._nnz()
    
    # Create batch of values
    val_batch = torch.randn(batch_size, nnz, dtype=torch.float64)
    b_batch = torch.randn(batch_size, n, dtype=torch.float64)
    
    # Solve
    x_batch = spsolve_batch_coo_same_layout(
        A_template, val_batch, b_batch,
        method='bicgstab', atol=1e-10
    )
    
    assert x_batch.shape == (batch_size, n)


def test_batch_coo_different_layout():
    """Test batch solve using COO tensor interface with different layouts"""
    A_list = []
    b_list = []
    
    for n in [16, 24, 32]:
        A_list.append(create_spd_sparse(n, density=0.3))
        b_list.append(torch.randn(n, dtype=torch.float64))
    
    x_list = spsolve_batch_coo_different_layout(
        A_list, b_list,
        method='bicgstab', atol=1e-10
    )
    
    assert len(x_list) == len(A_list)


# ============================================================================
# ParallelBatchSolver Tests
# ============================================================================

def test_parallel_batch_solver():
    """Test ParallelBatchSolver class"""
    n = 32
    batch_size = 5
    
    A_template = create_spd_sparse(n, density=0.3)
    row = A_template._indices()[0]
    col = A_template._indices()[1]
    nnz = A_template._nnz()
    shape = tuple(A_template.shape)
    
    # Create solver
    solver = ParallelBatchSolver(row, col, shape, method='bicgstab', device='cpu')
    
    # Solve multiple batches
    for _ in range(3):
        val_batch = torch.randn(batch_size, nnz, dtype=torch.float64)
        b_batch = torch.randn(batch_size, n, dtype=torch.float64)
        
        x_batch = solver(val_batch, b_batch, atol=1e-10)
        
        assert x_batch.shape == (batch_size, n)


# ============================================================================
# CUDA Batch Tests
# ============================================================================

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize('method', ['cudss_lu', 'cudss_cholesky', 'cudss_ldlt'])
def test_batch_same_layout_cudss(method):
    """Test batch solve with cuDSS backends"""
    if not is_cudss_available():
        pytest.skip("cuDSS not available")
    
    n = 32
    batch_size = 3
    device = 'cuda'
    
    A_template = create_spd_sparse(n, density=0.3, device=device)
    row = A_template._indices()[0]
    col = A_template._indices()[1]
    nnz = A_template._nnz()
    shape = tuple(A_template.shape)
    
    val_batch = torch.randn(batch_size, nnz, dtype=torch.float64, device=device)
    b_batch = torch.randn(batch_size, n, dtype=torch.float64, device=device)
    
    x_batch = spsolve_batch_same_layout(
        val_batch, row, col, shape, b_batch,
        method=method
    )
    
    assert x_batch.shape == (batch_size, n)


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    print("Testing batch solvers...")
    
    test_batch_same_layout_cpu(5, 32)
    print("✓ Batch same layout CPU test passed")
    
    test_batch_same_layout_gradient(3)
    print("✓ Batch gradient test passed")
    
    test_batch_different_layout()
    print("✓ Batch different layout test passed")
    
    test_batch_coo_same_layout()
    print("✓ Batch COO same layout test passed")
    
    test_batch_coo_different_layout()
    print("✓ Batch COO different layout test passed")
    
    test_parallel_batch_solver()
    print("✓ ParallelBatchSolver test passed")
    
    print("\nAll batch tests passed!")



# ============================================================================
# Device-agnostic pytorch-native methods (run on CPU / CUDA / ROCm alike)
# ============================================================================

@pytest.mark.parametrize('method', ['cg', 'bicgstab', 'gmres', 'minres', 'lsmr'])
def test_batch_same_layout_pytorch_methods(method):
    """All pytorch-native methods work through the batch path (these are the
    device-agnostic solvers that also cover ROCm). Checks solution + gradients."""
    n, batch_size = 16, 3
    A_template = create_spd_sparse(n, density=0.4)
    row, col = A_template._indices()[0], A_template._indices()[1]
    nnz = A_template._nnz()

    val_batch = torch.randn(batch_size, nnz, dtype=torch.float64)
    # make each matrix SPD-ish by reusing the template values + noise on diagonal
    base = A_template._values()
    val_batch = base.unsqueeze(0).repeat(batch_size, 1).clone()
    val_batch.requires_grad_(True)
    b_batch = torch.randn(batch_size, n, dtype=torch.float64)

    x = spsolve_batch_same_layout(val_batch, row, col, (n, n), b_batch,
                                  method=method, atol=1e-12, maxiter=5000)
    assert x.shape == (batch_size, n)

    # reference: dense solve per item
    ref = []
    for j in range(batch_size):
        A = torch.zeros(n, n, dtype=torch.float64)
        A[row, col] = val_batch.detach()[j]
        ref.append(torch.linalg.solve(A, b_batch[j]))
    ref = torch.stack(ref)
    # all iterative -> relative tolerance ~rtol (default 1e-6)
    assert (x - ref).abs().max().item() < 1e-5, f"{method}: {(x - ref).abs().max().item():.2e}"

    g = torch.autograd.grad(x.sum(), val_batch)[0]
    assert torch.isfinite(g).all()
