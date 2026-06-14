"""Autograd ``Function`` classes for differentiable sparse ops.

* :class:`DetAdjoint`            -- ``det(A)`` (implicit-diff gradient).
* :class:`EigshAdjoint`          -- ``eigsh`` (implicit-diff gradient).
* :class:`SparseSolveFunction`   -- scipy-backed differentiable solve.
* :class:`SparseSparseMatmulFunction` -- ``A @ B`` sparse-sparse with sparse gradients.

Lifted from ``sparse_tensor.core`` as part of the file split. All four are
used internally by :class:`~torch_sla.sparse_tensor.SparseTensor` methods.
"""
from __future__ import annotations

from typing import Tuple

import torch
from torch.autograd.function import Function


# =============================================================================
# Adjoint Determinant Solver
# =============================================================================

class DetAdjoint(Function):
    """
    Adjoint-based differentiable determinant computation.
    
    Uses implicit differentiation to compute gradients:
    For matrix A with determinant d = det(A):
        ∂d/∂A = d * (A^{-1})^T
    
    This means:
        ∂d/∂A_ij = d * (A^{-1})_ji
    
    The gradient computation requires solving a linear system,
    which is done efficiently using the existing solve infrastructure.
    """
    
    @staticmethod
    def forward(ctx, val, row, col, shape, device, is_cuda):
        """Forward pass via the DetConfig dispatcher.

        Routing (CPU=SuperLU LU, CUDA=copy-to-CPU+SuperLU by default,
        SPD=Cholesky if scikit-sparse is installed, disconnected matrices
        factored per-component) lives in :mod:`torch_sla.det`.
        """
        from ..sparse_tensor import SparseTensor
        from ..det import _det_dispatch
        # Build a no-grad SparseTensor view so the dispatcher can look
        # at properties (is_pd / connected_components / ...).
        A = SparseTensor(val.detach(), row, col, shape)
        det_val = _det_dispatch(A)
        if not torch.is_tensor(det_val):
            det_val = torch.tensor(det_val, dtype=val.dtype, device=device)
        elif det_val.device != device:
            det_val = det_val.to(device)
        ctx.save_for_backward(val, row, col, det_val)
        ctx.shape = shape
        ctx.device = device
        ctx.is_cuda = is_cuda
        return det_val
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: compute gradient using adjoint method.
        
        Gradient formula: ∂L/∂A_ij = ∂L/∂d * d * (A^{-1})_ji
        """
        val, row, col, det_val = ctx.saved_tensors
        shape = ctx.shape
        device = ctx.device
        is_cuda = ctx.is_cuda
        
        # If determinant is zero, gradient is undefined
        if abs(det_val.item()) < 1e-15:
            # Return zero gradient for numerical stability
            return torch.zeros_like(val), None, None, None, None, None
        
        # Compute A^{-1} using sparse solve
        # We need (A^{-1})_ji for each nonzero A_ij
        # 
        # Formula: ∂d/∂A_ij = d * (A^{-1})^T_ij = d * (A^{-1})_ji
        # 
        # Strategy: For each unique row index i in the sparsity pattern,
        # solve A @ x = e_i to get the i-th column of A^{-1}
        # Then (A^{-1})_ji is the j-th element of this column
        
        # Build sparse matrix
        indices = torch.stack([row, col], dim=0).to(device)
        sparse_coo = torch.sparse_coo_tensor(indices, val, shape, device=device)
        
        # Get unique row indices (for each row i, we need column i of A^{-1})
        unique_rows = torch.unique(row)
        
        # Solve for each column of A^{-1}
        A_inv_cols = {}
        for i in unique_rows:
            # Create unit vector e_i
            e_i = torch.zeros(shape[0], dtype=val.dtype, device=device)
            e_i[i] = 1.0
            
            # Solve A @ x = e_i to get i-th column of A^{-1}
            if is_cuda:
                # Use dense solve for CUDA
                dense = sparse_coo.to_dense()
                x = torch.linalg.solve(dense, e_i)
            else:
                # Use scipy backend for CPU
                from ..backends.scipy_backend import scipy_solve
                x = scipy_solve(val, row, col, shape, e_i, method='lu')
            
            A_inv_cols[i.item()] = x
        
        # Compute gradient for each nonzero element
        # ∂d/∂A_ij = d * (A^{-1})_ji
        grad_val = torch.zeros_like(val)
        for k in range(len(val)):
            i = row[k].item()
            j = col[k].item()
            # (A^{-1})_ji is the j-th element of the i-th column of A^{-1}
            grad_val[k] = det_val * A_inv_cols[i][j]
        
        # Multiply by upstream gradient
        grad_val = grad_val * grad_output
        
        return grad_val, None, None, None, None, None


# =============================================================================
# Adjoint Eigenvalue Solver
# =============================================================================

class EigshAdjoint(Function):
    """
    Adjoint-based differentiable eigenvalue solver.
    
    Uses implicit differentiation to compute gradients with O(1) graph nodes,
    regardless of the number of iterations in the forward solve.
    
    For symmetric matrix A with eigenvalue λ and eigenvector v:
        A @ v = λ * v
    
    The gradient is:
        ∂λ/∂A = v @ v.T  (outer product)
        ∂v/∂A requires solving a linear system (more complex)
    """
    
    @staticmethod
    def forward(ctx, val, row, col, shape, k, which, return_eigenvectors, device):
        """Forward pass: compute eigenvalues using LOBPCG or dense fallback."""
        n = shape[0]
        
        # Detach for forward computation
        val_detached = val.detach()
        
        # Build sparse matrix for matvec
        indices = torch.stack([row, col], dim=0).to(device)
        sparse_coo = torch.sparse_coo_tensor(indices, val_detached, shape, device=device)
        
        def matvec(x):
            if x.dim() == 1:
                return torch.sparse.mm(sparse_coo, x.unsqueeze(1)).squeeze(1)
            return torch.sparse.mm(sparse_coo, x)
        
        # Compute eigenvalues
        if device.type == 'cuda':
            # Use LOBPCG on CUDA
            largest = which in ('LM', 'LA')
            eigenvalues, eigenvectors = _lobpcg_eigsh(
                matvec, n, k, val.dtype, device, largest=largest
            )
        else:
            # Use dense fallback on CPU (SciPy breaks gradient)
            A_dense = torch.zeros(n, n, dtype=val.dtype, device=device)
            for i in range(len(row)):
                A_dense[row[i], col[i]] = val_detached[i]
            
            eigenvalues_all, eigenvectors_all = torch.linalg.eigh(A_dense)
            
            if which in ('LM', 'LA'):
                # Largest eigenvalues
                eigenvalues = eigenvalues_all[-k:]
                eigenvectors = eigenvectors_all[:, -k:]
            else:
                # Smallest eigenvalues
                eigenvalues = eigenvalues_all[:k]
                eigenvectors = eigenvectors_all[:, :k]
        
        # Save for backward
        ctx.save_for_backward(val, eigenvalues, eigenvectors)
        ctx.row = row
        ctx.col = col
        ctx.shape = shape
        ctx.k = k
        ctx.return_eigenvectors = return_eigenvectors
        
        if return_eigenvectors:
            return eigenvalues, eigenvectors
        return eigenvalues, None
    
    @staticmethod
    def backward(ctx, grad_eigenvalues, grad_eigenvectors):
        """
        Backward pass using adjoint method.
        
        For eigenvalue λ_i with eigenvector v_i:
            ∂L/∂A[j,k] = Σ_i (∂L/∂λ_i) * v_i[j] * v_i[k]
        
        This gives us O(1) graph nodes.
        """
        val, eigenvalues, eigenvectors = ctx.saved_tensors
        row = ctx.row
        col = ctx.col
        k = ctx.k
        
        if grad_eigenvalues is None:
            return None, None, None, None, None, None, None, None
        
        # Compute gradient w.r.t. values
        # ∂L/∂A[i,j] = Σ_m (∂L/∂λ_m) * v_m[i] * v_m[j]
        # For sparse format: ∂L/∂val[idx] = Σ_m (∂L/∂λ_m) * v_m[row[idx]] * v_m[col[idx]]
        
        grad_val = torch.zeros_like(val)
        
        for m in range(k):
            if grad_eigenvalues[m] != 0:
                # v_m[row] * v_m[col] for each sparse entry
                v_m = eigenvectors[:, m]
                grad_val += grad_eigenvalues[m] * v_m[row] * v_m[col]
        
        # Handle eigenvector gradients if needed (more complex, skip for now)
        # The eigenvector gradient requires solving (A - λI) @ dv = ...
        
        return grad_val, None, None, None, None, None, None, None



# =============================================================================
# Autograd Functions
# =============================================================================

class SparseSolveFunction(Function):
    """
    Differentiable sparse solve using scipy for CPU.
    
    Solves Ax = b and computes gradients for both A's values and b.
    """
    
    @staticmethod
    def forward(ctx, val, row, col, shape, b, method, atol, maxiter):
        u = scipy_solve(val, row, col, shape, b, method=method, atol=atol, maxiter=maxiter)
        ctx.save_for_backward(val, row, col, u, b)
        ctx.shape = shape
        ctx.method = method
        ctx.atol = atol
        ctx.maxiter = maxiter
        return u
    
    @staticmethod
    def backward(ctx, grad_u):
        val, row, col, u, b = ctx.saved_tensors
        shape = ctx.shape
        method = ctx.method
        atol = ctx.atol
        maxiter = ctx.maxiter
        # Adjoint system uses the conjugate transpose A^H (= conj(A)^T), not
        # just A^T. For real matrices .conj() is a no-op so this is unchanged;
        # for complex it makes the Wirtinger gradient correct.
        grad_b = scipy_solve(val.conj(), col, row, (shape[1], shape[0]), grad_u,
                            method=method, atol=atol, maxiter=maxiter)
        grad_val = -grad_b[row] * u[col].conj()
        return grad_val, None, None, None, grad_b, None, None, None


class SparseSparseMatmulFunction(Function):
    """
    Differentiable Sparse @ Sparse multiplication with SPARSE gradients.
    
    Forward: C = A @ B where A is [M, K], B is [K, N], C is [M, N]
    
    Backward:
    - grad_A_values = (grad_C @ B^T)[A_row, A_col]  (sparse gradient at A's positions)
    - grad_B_values = (A^T @ grad_C)[B_row, B_col]  (sparse gradient at B's positions)
    
    The gradients are computed only at the original non-zero positions,
    keeping memory usage proportional to nnz rather than M*N.
    """
    
    @staticmethod
    def forward(ctx, val_a, row_a, col_a, shape_a, val_b, row_b, col_b, shape_b):
        M, K = shape_a
        K2, N = shape_b
        assert K == K2, f"Inner dimensions must match: {K} vs {K2}"
        
        # Create torch sparse tensors for multiplication
        A_coo = torch.sparse_coo_tensor(
            torch.stack([row_a, col_a]), val_a, (M, K)
        ).coalesce()
        B_coo = torch.sparse_coo_tensor(
            torch.stack([row_b, col_b]), val_b, (K, N)
        ).coalesce()
        
        # Sparse @ Sparse -> Sparse
        with torch.no_grad():
            C_coo = torch.sparse.mm(A_coo, B_coo).coalesce()
        
        # Extract result
        C_indices = C_coo._indices()
        C_values = C_coo._values()
        
        # Save for backward
        ctx.save_for_backward(val_a, row_a, col_a, val_b, row_b, col_b, 
                              C_indices[0], C_indices[1], C_values)
        ctx.shape_a = shape_a
        ctx.shape_b = shape_b
        
        return C_values, C_indices[0], C_indices[1]
    
    @staticmethod
    def backward(ctx, grad_C_values, grad_row_c, grad_col_c):
        (val_a, row_a, col_a, val_b, row_b, col_b, 
         row_c, col_c, val_c) = ctx.saved_tensors
        M, K = ctx.shape_a
        K2, N = ctx.shape_b
        
        grad_val_a = None
        grad_val_b = None
        
        if ctx.needs_input_grad[0]:
            # grad_A = grad_C @ B^T
            grad_C_coo = torch.sparse_coo_tensor(
                torch.stack([row_c, col_c]), grad_C_values, (M, N)
            ).coalesce()
            B_T_coo = torch.sparse_coo_tensor(
                torch.stack([col_b, row_b]), val_b, (N, K)
            ).coalesce()
            grad_A_dense = torch.sparse.mm(grad_C_coo, B_T_coo).to_dense()
            grad_val_a = grad_A_dense[row_a, col_a]
        
        if ctx.needs_input_grad[4]:
            # grad_B = A^T @ grad_C
            A_T_coo = torch.sparse_coo_tensor(
                torch.stack([col_a, row_a]), val_a, (K, M)
            ).coalesce()
            grad_C_coo = torch.sparse_coo_tensor(
                torch.stack([row_c, col_c]), grad_C_values, (M, N)
            ).coalesce()
            grad_B_dense = torch.sparse.mm(A_T_coo, grad_C_coo).to_dense()
            grad_val_b = grad_B_dense[row_b, col_b]
        
        return grad_val_a, None, None, None, grad_val_b, None, None, None


def _sparse_sparse_matmul_with_sparse_grad(
    val_a: torch.Tensor, row_a: torch.Tensor, col_a: torch.Tensor, shape_a: Tuple[int, int],
    val_b: torch.Tensor, row_b: torch.Tensor, col_b: torch.Tensor, shape_b: Tuple[int, int]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[int, int]]:
    """
    Sparse @ Sparse with sparse gradients.
    
    Parameters
    ----------
    val_a, row_a, col_a : torch.Tensor
        COO representation of matrix A.
    shape_a : Tuple[int, int]
        Shape of matrix A (M, K).
    val_b, row_b, col_b : torch.Tensor
        COO representation of matrix B.
    shape_b : Tuple[int, int]
        Shape of matrix B (K, N).
    
    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[int, int]]
        (values, row_indices, col_indices, shape) of result C = A @ B.
    """
    M, K = shape_a
    K2, N = shape_b
    
    C_values, C_row, C_col = SparseSparseMatmulFunction.apply(
        val_a, row_a, col_a, shape_a,
        val_b, row_b, col_b, shape_b
    )
    
    return C_values, C_row, C_col, (M, N)


# =============================================================================
# LOBPCG and Power Iteration for CUDA
# =============================================================================

def _lobpcg_eigsh(
    A_matvec,
    n: int,
    k: int,
    dtype: torch.dtype,
    device: torch.device,
    largest: bool = True,
    maxiter: int = 1000,
    tol: float = 1e-8
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    LOBPCG eigenvalue solver for sparse matrices on any device.
    
    Uses subspace iteration with Rayleigh-Ritz procedure to find
    the k largest or smallest eigenvalues.
    
    Parameters
    ----------
    A_matvec : callable
        Function that computes A @ x for input x of shape [n] or [n, m].
    n : int
        Matrix dimension.
    k : int
        Number of eigenvalues to compute.
    dtype : torch.dtype
        Data type.
    device : torch.device
        Device to compute on.
    largest : bool, optional
        If True, compute largest eigenvalues. Default: True.
    maxiter : int, optional
        Maximum iterations. Default: 1000.
    tol : float, optional
        Convergence tolerance. Default: 1e-8.
    
    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        (eigenvalues, eigenvectors) with shapes [k] and [n, k].
    """
    m = min(2 * k, n)
    X = torch.randn(n, m, dtype=dtype, device=device)
    X, _ = torch.linalg.qr(X)
    
    eigenvalues_prev = None
    
    for iteration in range(maxiter):
        AX = A_matvec(X)
        H = X.T @ AX
        eigenvalues, eigenvectors = torch.linalg.eigh(H)
        
        if largest:
            idx = eigenvalues.argsort(descending=True)
        else:
            idx = eigenvalues.argsort()
        
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        X = X @ eigenvectors
        
        if eigenvalues_prev is not None:
            diff = (eigenvalues[:k] - eigenvalues_prev[:k]).abs()
            if (diff < tol * eigenvalues[:k].abs().clamp(min=1e-10)).all():
                break
        eigenvalues_prev = eigenvalues.clone()
        
        if iteration < maxiter - 1:
            AX = A_matvec(X)
            residual = AX - X * eigenvalues.unsqueeze(0)
            combined = torch.cat([X[:, :k], residual[:, :k]], dim=1)
            X, _ = torch.linalg.qr(combined)
            if X.size(1) < m:
                extra = torch.randn(n, m - X.size(1), dtype=dtype, device=device)
                X = torch.cat([X, extra], dim=1)
                X, _ = torch.linalg.qr(X)
    
    return eigenvalues[:k], X[:, :k]

