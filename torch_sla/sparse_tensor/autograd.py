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

        if abs(det_val.item()) < 1e-15:
            return torch.zeros_like(val), None, None, None, None, None

        # Jacobi: ∂det/∂A_ij = det * (A^-1)_ji, so grad_val[k] = det *
        # (A^-1)[col[k], row[k]] * grad_output. We need only the entries of
        # A^-1 at the nonzero positions, transposed.
        #
        # Strategy (was: per-unique-row Python LU + per-nnz Python gather):
        #   1. One sparse LU factorisation of A^T (CPU via SuperLU).
        #   2. Batched solve A^T Y = E, where E's columns are e_c for each
        #      unique c in col[:]. Y[:, k] = (A^-T)[:, c_k] = (A^-1)[c_k, :].
        #   3. Vectorised gather: grad_val[k] = det * Y[row[k], col_inv[k]].
        #
        # For nnz ~ N this drops the Python loop overhead from O(N) to 0
        # and replaces N LU solves with one batched solve (SuperLU shares
        # the factorisation).

        # On CUDA we fall back to dense for now -- a sparse batched solve
        # path on CUDA would need cuDSS-style multi-RHS support.
        if is_cuda:
            indices = torch.stack([row, col], dim=0).to(device)
            sparse_coo = torch.sparse_coo_tensor(
                indices, val, shape, device=device)
            dense = sparse_coo.to_dense()
            A_inv = torch.linalg.inv(dense)
            grad_val = det_val * A_inv[col.to(torch.int64),
                                         row.to(torch.int64)] * grad_output
            return grad_val, None, None, None, None, None

        # CPU sparse path -- single SuperLU + batched solve.
        from ..backends.scipy_backend import torch_coo_to_scipy_csr
        import scipy.sparse.linalg as spla
        import numpy as np

        A_T_csc = torch_coo_to_scipy_csr(val, row, col, shape).T.tocsc()
        unique_cols, col_inv = torch.unique(col, return_inverse=True)
        unique_cols_np = unique_cols.cpu().numpy()
        n_uc = unique_cols_np.shape[0]
        E = np.zeros((shape[0], n_uc), dtype=val.detach().cpu().numpy().dtype)
        E[unique_cols_np, np.arange(n_uc)] = 1.0
        lu = spla.splu(A_T_csc)
        Y_np = lu.solve(E)                       # (N, n_uc)
        Y = torch.from_numpy(Y_np).to(dtype=val.dtype, device=device)
        gathered = Y[row.to(torch.int64), col_inv.to(torch.int64)]
        grad_val = det_val * gathered * grad_output
        return grad_val, None, None, None, None, None


# =============================================================================
# Adjoint Eigenvalue Solver
# =============================================================================

def _eigh_dense_topk(sparse_coo, k, largest):
    """Small-n CPU fallback: dense eigh + slice top/bottom k."""
    A_dense = sparse_coo.to_dense()
    evals_all, evecs_all = torch.linalg.eigh(A_dense)
    if largest:
        return evals_all[-k:], evecs_all[:, -k:]
    return evals_all[:k], evecs_all[:, :k]


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
            # CPU path. For n > 1024 dispatch to scipy.sparse.linalg.eigsh
            # (ARPACK Lanczos) -- truly O(nnz*k*iter), no dense n*n alloc.
            # For small n we still use torch.linalg.eigh on the dense
            # tensor: LAPACK is faster than ARPACK's setup cost, and
            # scipy isn't a hard dep at that scale.
            largest = which in ('LM', 'LA')
            if n > 1024:
                from ..backends.scipy_backend import SCIPY_AVAILABLE, scipy_eigsh
                if SCIPY_AVAILABLE:
                    which_arpack = {'LM': 'LM', 'SM': 'SM', 'LA': 'LA', 'SA': 'SA'}[which]
                    eigenvalues, eigenvectors = scipy_eigsh(
                        val_detached, row, col, shape,
                        k=k, which=which_arpack, sigma=None,
                        return_eigenvectors=True,
                    )
                    eigenvalues = eigenvalues.to(device=device)
                    eigenvectors = eigenvectors.to(device=device)
                else:
                    eigenvalues, eigenvectors = _eigh_dense_topk(
                        sparse_coo, k, largest)
            else:
                eigenvalues, eigenvectors = _eigh_dense_topk(
                    sparse_coo, k, largest)
        
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

