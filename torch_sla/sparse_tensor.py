"""
SparseTensor wrapper class for PyTorch sparse tensors.

Supports batched and block sparse tensors with shape [...batch, M, N, ...block]:
- Leading dimensions: batch dimensions [B1, B2, ...]
- Matrix dimensions: (M, N) at positions (sparse_dim[0], sparse_dim[1]), default (-2, -1)
- Trailing dimensions: block dimensions [K1, K2, ...]

Key Features:
- Automatic symmetry and positive definiteness detection
- Sparse linear equation solving with gradient support
- Sparse-sparse multiplication with sparse gradients
- Batched operations for all methods
- CUDA support with LOBPCG for eigenvalue computation

Examples
--------
>>> # Create a simple sparse matrix
>>> val = torch.tensor([4.0, -1.0, -1.0, 4.0])
>>> row = torch.tensor([0, 0, 1, 1])
>>> col = torch.tensor([0, 1, 0, 1])
>>> A = SparseTensor(val, row, col, (2, 2))
>>>
>>> # Check properties (returns boolean tensor for batched)
>>> is_sym = A.is_symmetric()  # tensor(True)
>>> is_pd = A.is_positive_definite()  # tensor(True)
>>>
>>> # Solve linear system
>>> b = torch.tensor([1.0, 2.0])
>>> x = A.solve(b)
>>>
>>> # Matrix operations
>>> y = A @ x  # Sparse @ Dense
>>> C = A @ A  # Sparse @ Sparse (sparse gradient)
"""

import os
import torch
from torch.autograd.function import Function
from typing import Tuple, Optional, Union, Literal, List, Dict
import warnings
import math

from .backends import (
    is_scipy_available,
    is_eigen_available,
    is_cupy_available,
    is_cudss_available,
    select_backend,
    select_method,
    BackendType,
    MethodType,
)
from .backends.scipy_backend import (
    scipy_solve,
    scipy_eigs,
    scipy_eigsh,
    scipy_svds,
    scipy_norm,
    scipy_lu,
    scipy_det,
)


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
        """Forward pass: compute determinant."""
        from .backends.scipy_backend import scipy_det
        
        if is_cuda:
            # For CUDA, convert to dense and use torch.linalg.det
            # NOTE: This is inefficient for sparse matrices due to O(n²) memory
            # and O(n³) computation. cuDSS doesn't expose determinant
            # computation for sparse matrices directly.
            # 
            # Performance: ~100x slower than CPU for sparse matrices
            # Recommendation: Use .cpu().det() for better performance
            val_detached = val.detach()
            indices = torch.stack([row, col], dim=0).to(device)
            sparse_coo = torch.sparse_coo_tensor(indices, val_detached, shape, device=device)
            dense = sparse_coo.to_dense()
            det_val = torch.linalg.det(dense)
        else:
            # For CPU, use scipy backend
            det_val = scipy_det(val.detach(), row, col, shape)
        
        # Save for backward
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
                from .backends.scipy_backend import scipy_solve
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
# Utility Functions
# =============================================================================

def estimate_direct_solver_memory(nnz: int, n: int, dtype: torch.dtype) -> int:
    """
    Estimate memory required for direct sparse solver.
    
    Parameters
    ----------
    nnz : int
        Number of non-zero elements.
    n : int
        Matrix dimension.
    dtype : torch.dtype
        Data type of the matrix.
    
    Returns
    -------
    int
        Estimated memory in bytes.
    """
    bytes_per_element = 8 if dtype == torch.float64 else 4
    fill_factor = min(10, max(2, n / 100))
    factor_memory = int(nnz * fill_factor * bytes_per_element)
    workspace_memory = n * bytes_per_element * 10
    return factor_memory + workspace_memory


def get_available_gpu_memory() -> int:
    """
    Get available GPU memory in bytes.
    
    Returns
    -------
    int
        Available GPU memory in bytes, or 0 if CUDA is not available.
    """
    if not torch.cuda.is_available():
        return 0
    try:
        free_memory, total_memory = torch.cuda.mem_get_info()
        return free_memory
    except Exception:
        return torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()


def auto_select_method(
    nnz: int, n: int, dtype: torch.dtype, is_cuda: bool, is_spd: bool = False,
    memory_threshold: float = 0.8
) -> Tuple[str, str]:
    """
    Automatically select the best backend and method.
    
    Parameters
    ----------
    nnz : int
        Number of non-zero elements.
    n : int
        Matrix dimension.
    dtype : torch.dtype
        Data type of the matrix.
    is_cuda : bool
        Whether the matrix is on CUDA.
    is_spd : bool, optional
        Whether the matrix is symmetric positive definite. Default: False.
    memory_threshold : float, optional
        Fraction of GPU memory to use. Default: 0.8.
        
    Returns
    -------
    Tuple[str, str]
        (backend, method) tuple.
    """
    if not is_cuda:
        if is_scipy_available():
            return ("scipy", "lu")
        elif is_eigen_available():
            return ("eigen", "cg" if is_spd else "bicgstab")
        else:
            raise RuntimeError("No CPU backend available")
    
    estimated_memory = estimate_direct_solver_memory(nnz, n, dtype)
    available_memory = get_available_gpu_memory()
    
    if available_memory > 0 and estimated_memory < available_memory * memory_threshold:
        if is_cudss_available():
            return ("cudss", "cholesky" if is_spd else "lu")
        elif is_cupy_available():
            return ("cupy", "lu")
    
    if is_scipy_available():
        return ("scipy", "lu")
    
    raise RuntimeError("No suitable backend available")


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


def _power_iteration_svd(
    A_matvec,
    At_matvec,
    m: int,
    n: int,
    k: int,
    dtype: torch.dtype,
    device: torch.device,
    maxiter: int = 100,
    tol: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Power iteration based SVD for sparse matrices on any device.
    
    Parameters
    ----------
    A_matvec : callable
        Function that computes A @ x.
    At_matvec : callable
        Function that computes A^T @ x.
    m, n : int
        Matrix dimensions (m rows, n columns).
    k : int
        Number of singular values to compute.
    dtype : torch.dtype
        Data type.
    device : torch.device
        Device to compute on.
    maxiter : int, optional
        Maximum iterations. Default: 100.
    tol : float, optional
        Convergence tolerance. Default: 1e-6.
    
    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        (U, S, Vt) with shapes [m, k], [k], [k, n].
    """
    V = torch.randn(n, k, dtype=dtype, device=device)
    V, _ = torch.linalg.qr(V)
    
    for _ in range(maxiter):
        U = A_matvec(V)
        U, R = torch.linalg.qr(U)
        V_new = At_matvec(U)
        S = V_new.norm(dim=0)
        V_new = V_new / S.unsqueeze(0).clamp(min=1e-10)
        diff = (V_new - V).norm()
        V = V_new
        if diff < tol:
            break
    
    return U, S, V.T


# =============================================================================
# SparseTensor Class
# =============================================================================

class SparseTensor:
    """
    Wrapper class for PyTorch sparse tensors with batched and block support.
    
    Supports tensors with shape [...batch, M, N, ...block] where:
    - Leading dimensions [...batch] are batch dimensions
    - (M, N) are the sparse matrix dimensions (at sparse_dim positions)
    - Trailing dimensions [...block] are block dimensions
    
    Parameters
    ----------
    values : torch.Tensor
        Non-zero values with shape:
        - Simple: [nnz]
        - Batched: [...batch, nnz] 
        - Block: [nnz, *block_shape]
        - Batched+Block: [...batch, nnz, *block_shape]
    row_indices : torch.Tensor
        Row indices with shape [nnz]. Must be on the same device as values.
    col_indices : torch.Tensor
        Column indices with shape [nnz]. Must be on the same device as values.
    shape : Tuple[int, ...]
        Full tensor shape [...batch, M, N, *block_shape].
    sparse_dim : Tuple[int, int], optional
        Which dimensions are sparse (M, N). Default: (-2, -1) meaning last two
        before any block dimensions.
    
    Attributes
    ----------
    values : torch.Tensor
        The non-zero values.
    row_indices : torch.Tensor
        Row indices of non-zeros.
    col_indices : torch.Tensor
        Column indices of non-zeros.
    shape : Tuple[int, ...]
        Full tensor shape.
    sparse_shape : Tuple[int, int]
        The (M, N) dimensions.
    batch_shape : Tuple[int, ...]
        The batch dimensions.
    block_shape : Tuple[int, ...]
        The block dimensions.
    
    Examples
    --------
    **1. Simple 2D Sparse Matrix [M, N]**
    
    >>> import torch
    >>> from torch_sla import SparseTensor
    >>> 
    >>> # Create a 3x3 tridiagonal matrix in COO format
    >>> val = torch.tensor([4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0])
    >>> row = torch.tensor([0, 0, 1, 1, 1, 2, 2])
    >>> col = torch.tensor([0, 1, 0, 1, 2, 1, 2])
    >>> A = SparseTensor(val, row, col, (3, 3))
    >>> print(A)
    SparseTensor(shape=(3, 3), sparse=(3, 3), nnz=7, dtype=torch.float64, device=cpu)
    >>> 
    >>> # Solve Ax = b
    >>> b = torch.tensor([1.0, 2.0, 3.0])
    >>> x = A.solve(b)
    
    **2. Batched Sparse Matrices [B, M, N]**
    
    Same sparsity pattern, different values for each batch.
    
    >>> # 4 matrices, each 3x3, same structure
    >>> batch_size = 4
    >>> val_batch = val.unsqueeze(0).expand(batch_size, -1).clone()  # [4, 7]
    >>> for i in range(batch_size):
    ...     val_batch[i] = val * (1.0 + 0.1 * i)  # Scale each matrix
    >>> 
    >>> A_batch = SparseTensor(val_batch, row, col, (4, 3, 3))
    >>> print(A_batch.batch_shape)  # (4,)
    >>> print(A_batch.sparse_shape)  # (3, 3)
    >>> 
    >>> # Batched solve
    >>> b_batch = torch.randn(4, 3)
    >>> x_batch = A_batch.solve(b_batch)  # [4, 3]
    
    **3. Multi-Dimensional Batch [B1, B2, M, N]**
    
    >>> B1, B2 = 2, 3  # e.g., 2 materials x 3 temperatures
    >>> val_batch = val.unsqueeze(0).unsqueeze(0).expand(B1, B2, -1).clone()  # [2, 3, 7]
    >>> A_multi = SparseTensor(val_batch, row, col, (B1, B2, 3, 3))
    >>> print(A_multi.batch_shape)  # (2, 3)
    >>> 
    >>> b_multi = torch.randn(B1, B2, 3)
    >>> x_multi = A_multi.solve(b_multi)  # [2, 3, 3]
    
    **4. Block Sparse Matrix [M, N, K, K] (Block Size K)**
    
    Each non-zero entry is a KxK dense block instead of a scalar.
    
    >>> # 2x2 block matrix with 2x2 blocks = 4x4 total
    >>> block_size = 2
    >>> nnz = 3  # 3 non-zero blocks
    >>> 
    >>> # Values: [nnz, K, K] = [3, 2, 2]
    >>> val_block = torch.randn(nnz, block_size, block_size)
    >>> row_block = torch.tensor([0, 0, 1])  # Block row indices
    >>> col_block = torch.tensor([0, 1, 1])  # Block col indices
    >>> 
    >>> # Shape: (num_block_rows, num_block_cols, block_size, block_size)
    >>> A_block = SparseTensor(val_block, row_block, col_block, (2, 2, 2, 2))
    >>> print(A_block.block_shape)  # (2, 2)
    >>> print(A_block.sparse_shape)  # (2, 2) - number of blocks
    >>> print(A_block.shape)  # (2, 2, 2, 2) - full shape
    
    **5. Batched Block Sparse [B, M, N, K, K]**
    
    >>> batch_size = 4
    >>> val_batch_block = torch.randn(batch_size, nnz, block_size, block_size)  # [4, 3, 2, 2]
    >>> A_batch_block = SparseTensor(val_batch_block, row_block, col_block, (4, 2, 2, 2, 2))
    >>> print(A_batch_block.batch_shape)  # (4,)
    >>> print(A_batch_block.block_shape)  # (2, 2)
    
    **6. Create from Dense Matrix**
    
    >>> A_dense = torch.randn(100, 100)
    >>> A_dense[A_dense.abs() < 0.5] = 0  # Sparsify
    >>> A = SparseTensor.from_dense(A_dense)
    
    **7. Create from PyTorch Sparse Tensor**
    
    >>> A_torch = torch.randn(100, 100).to_sparse_coo()
    >>> A = SparseTensor.from_torch_sparse(A_torch)
    
    **8. Property Detection**
    
    >>> A = SparseTensor(val, row, col, (3, 3))
    >>> A.is_symmetric()  # tensor(True) - returns tensor for batch support
    >>> A.is_positive_definite()  # tensor(True)
    >>> A.is_positive_definite('cholesky')  # Use Cholesky factorization check
    
    **9. Matrix Operations**
    
    >>> # Matrix-vector multiply
    >>> y = A @ x  # SparseTensor @ dense vector
    >>> 
    >>> # Sparse-sparse multiply (returns SparseTensor with sparse gradients)
    >>> C = A @ A
    >>> 
    >>> # Norms
    >>> A.norm('fro')  # Frobenius norm
    >>> 
    >>> # Eigenvalues (symmetric matrices)
    >>> eigenvalues, eigenvectors = A.eigsh(k=2, which='LM')
    
    **10. CUDA Support**
    
    >>> A_cuda = A.cuda()
    >>> x = A_cuda.solve(b.cuda())  # Uses cuDSS or CuPy
    """
    
    def __init__(
        self,
        values: torch.Tensor,
        row_indices: torch.Tensor,
        col_indices: torch.Tensor,
        shape: Tuple[int, ...],
        sparse_dim: Tuple[int, int] = (-2, -1),
    ):
        self.values = values
        self.row_indices = row_indices
        self.col_indices = col_indices
        self._shape = tuple(shape)
        self._sparse_dim = self._normalize_sparse_dim(sparse_dim, len(shape))
        
        # Cache for computed properties
        self._is_symmetric_cache = None
        self._is_positive_definite_cache = None
        
        self._validate()
    
    def _normalize_sparse_dim(self, sparse_dim: Tuple[int, int], ndim: int) -> Tuple[int, int]:
        """Normalize negative indices in sparse_dim."""
        dim_m = sparse_dim[0] if sparse_dim[0] >= 0 else ndim + sparse_dim[0]
        dim_n = sparse_dim[1] if sparse_dim[1] >= 0 else ndim + sparse_dim[1]
        return (dim_m, dim_n)
    
    def _validate(self):
        """Validate tensor dimensions and indices."""
        ndim = len(self._shape)
        dim_m, dim_n = self._sparse_dim
        if ndim < 2:
            raise ValueError(f"Shape must have at least 2 dimensions, got {ndim}")
        if not (0 <= dim_m < ndim and 0 <= dim_n < ndim):
            raise ValueError(f"sparse_dim {self._sparse_dim} out of range for shape {self._shape}")
        if dim_m == dim_n:
            raise ValueError(f"sparse_dim dimensions must be different")
    
    # =========================================================================
    # Class Methods
    # =========================================================================
    
    @classmethod
    def from_dense(
        cls, 
        A: torch.Tensor, 
        sparse_dim: Tuple[int, int] = (-2, -1)
    ) -> "SparseTensor":
        """
        Create SparseTensor from dense tensor.
        
        Parameters
        ----------
        A : torch.Tensor
            Dense tensor with shape [...batch, M, N, ...block].
        sparse_dim : Tuple[int, int], optional
            Which dimensions are sparse. Default: (-2, -1).
        
        Returns
        -------
        SparseTensor
            Sparse representation of A.
        
        Examples
        --------
        >>> A_dense = torch.randn(3, 3)
        >>> A_dense[A_dense.abs() < 0.5] = 0
        >>> A = SparseTensor.from_dense(A_dense)
        """
        ndim = A.dim()
        dim_m = sparse_dim[0] if sparse_dim[0] >= 0 else ndim + sparse_dim[0]
        dim_n = sparse_dim[1] if sparse_dim[1] >= 0 else ndim + sparse_dim[1]
        
        if ndim == 2 and dim_m == 0 and dim_n == 1:
            A_sparse = A.to_sparse_coo().coalesce()
            indices = A_sparse.indices()
            values = A_sparse.values()
            return cls(values, indices[0], indices[1], tuple(A.shape), sparse_dim=sparse_dim)
        
        perm = [i for i in range(ndim) if i not in (dim_m, dim_n)] + [dim_m, dim_n]
        A_perm = A.permute(*perm)
        batch_shape = A_perm.shape[:-2]
        M, N = A_perm.shape[-2], A_perm.shape[-1]
        A_flat = A_perm.reshape(-1, M, N)
        
        A_2d = A_flat[0].to_sparse_coo()
        indices = A_2d._indices()
        row = indices[0]
        col = indices[1]
        nnz = row.size(0)
        
        values = A_flat[:, row, col]
        if len(batch_shape) > 0:
            values = values.reshape(*batch_shape, nnz)
        else:
            values = values.squeeze(0)
        
        return cls(values, row, col, tuple(A.shape), sparse_dim=sparse_dim)
    
    @classmethod
    def from_torch_sparse(cls, A: torch.Tensor) -> "SparseTensor":
        """
        Create SparseTensor from PyTorch sparse tensor.
        
        Parameters
        ----------
        A : torch.Tensor
            PyTorch sparse COO or CSR tensor (2D only).
        
        Returns
        -------
        SparseTensor
            SparseTensor representation.
        
        Examples
        --------
        >>> A_coo = torch.randn(3, 3).to_sparse_coo()
        >>> A = SparseTensor.from_torch_sparse(A_coo)
        """
        if A.layout == torch.sparse_csr:
            A = A.to_sparse_coo()
        A = A.coalesce()
        indices = A.indices()
        values = A.values()
        return cls(values, indices[0], indices[1], tuple(A.shape))
    
    # =========================================================================
    # Properties
    # =========================================================================
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Full tensor shape [...batch, M, N, ...block]."""
        return self._shape
    
    @property
    def sparse_shape(self) -> Tuple[int, int]:
        """The (M, N) sparse matrix dimensions."""
        dim_m, dim_n = self._sparse_dim
        return (self._shape[dim_m], self._shape[dim_n])
    
    @property
    def batch_shape(self) -> Tuple[int, ...]:
        """The batch dimensions before the sparse dimensions."""
        dim_m, dim_n = self._sparse_dim
        min_dim = min(dim_m, dim_n)
        return self._shape[:min_dim]
    
    @property
    def block_shape(self) -> Tuple[int, ...]:
        """The block dimensions after the sparse dimensions."""
        dim_m, dim_n = self._sparse_dim
        max_dim = max(dim_m, dim_n)
        return self._shape[max_dim + 1:]
    
    @property
    def sparse_dim(self) -> Tuple[int, int]:
        """The dimensions that are sparse (M, N)."""
        return self._sparse_dim
    
    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return len(self._shape)
    
    @property
    def nnz(self) -> int:
        """Number of non-zero elements (per batch/block)."""
        return self.row_indices.size(0)
    
    @property
    def dtype(self) -> torch.dtype:
        """Data type of the values."""
        return self.values.dtype
    
    @property
    def device(self) -> torch.device:
        """Device of the tensor."""
        return self.values.device
    
    @property
    def is_cuda(self) -> bool:
        """Whether the tensor is on CUDA."""
        return self.values.is_cuda
    
    @property
    def is_batched(self) -> bool:
        """Whether the tensor has batch dimensions."""
        return len(self.batch_shape) > 0
    
    @property
    def is_block(self) -> bool:
        """Whether the tensor has block dimensions."""
        return len(self.block_shape) > 0
    
    @property
    def batch_size(self) -> int:
        """Total number of batch elements (product of batch_shape)."""
        return math.prod(self.batch_shape) if self.batch_shape else 1
    
    @property
    def is_square(self) -> bool:
        """Whether the sparse dimensions are square (M == N)."""
        M, N = self.sparse_shape
        return M == N
    
    # =========================================================================
    # Device and Type Management
    # =========================================================================
    
    def to(
        self, 
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None
    ) -> "SparseTensor":
        """
        Move tensor to device and/or convert dtype.
        
        Parameters
        ----------
        device : str or torch.device, optional
            Target device (e.g., 'cuda', 'cpu', 'cuda:0').
        dtype : torch.dtype, optional
            Target data type (e.g., torch.float32, torch.float64).
        
        Returns
        -------
        SparseTensor
            New SparseTensor on the target device/dtype.
        
        Examples
        --------
        >>> A = SparseTensor(val, row, col, shape)
        >>> A_cuda = A.to('cuda')
        >>> A_float32 = A.to(dtype=torch.float32)
        >>> A_cuda_float32 = A.to('cuda', torch.float32)
        """
        new_values = self.values
        new_row = self.row_indices
        new_col = self.col_indices
        
        if device is not None:
            new_values = new_values.to(device)
            new_row = new_row.to(device)
            new_col = new_col.to(device)
        
        if dtype is not None:
            new_values = new_values.to(dtype)
        
        result = SparseTensor(
            new_values, new_row, new_col, self._shape,
            sparse_dim=self._sparse_dim
        )
        return result
    
    def cuda(self, device: Optional[int] = None) -> "SparseTensor":
        """
        Move tensor to CUDA device.
        
        Parameters
        ----------
        device : int, optional
            CUDA device index. Default: current device.
        
        Returns
        -------
        SparseTensor
            Tensor on CUDA.
        """
        if device is None:
            return self.to('cuda')
        return self.to(f'cuda:{device}')
    
    def cpu(self) -> "SparseTensor":
        """
        Move tensor to CPU.
        
        Returns
        -------
        SparseTensor
            Tensor on CPU.
        """
        return self.to('cpu')
    
    def float(self) -> "SparseTensor":
        """Convert to float32."""
        return self.to(dtype=torch.float32)
    
    def double(self) -> "SparseTensor":
        """Convert to float64."""
        return self.to(dtype=torch.float64)
    
    def half(self) -> "SparseTensor":
        """Convert to float16."""
        return self.to(dtype=torch.float16)
    
    # =========================================================================
    # Conversion Methods
    # =========================================================================
    
    def to_torch_sparse(self, batch_idx: Optional[Tuple[int, ...]] = None) -> torch.Tensor:
        """
        Convert to PyTorch sparse COO tensor.
        
        Parameters
        ----------
        batch_idx : Tuple[int, ...], optional
            For batched tensors, which batch element to convert.
            Default: (0, 0, ...) for first batch element.
        
        Returns
        -------
        torch.Tensor
            PyTorch sparse COO tensor.
        """
        if self.is_batched:
            if batch_idx is None:
                batch_idx = (0,) * len(self.batch_shape)
            vals = self.values[batch_idx]
        else:
            vals = self.values
        
        M, N = self.sparse_shape
        indices = torch.stack([self.row_indices, self.col_indices], dim=0)
        
        if self.is_block:
            return torch.sparse_coo_tensor(indices, vals, (M, N) + self.block_shape)
        else:
            return torch.sparse_coo_tensor(indices, vals, (M, N))
    
    def to_dense(self, batch_idx: Optional[Tuple[int, ...]] = None) -> torch.Tensor:
        """
        Convert to dense tensor.
        
        Parameters
        ----------
        batch_idx : Tuple[int, ...], optional
            For batched tensors, which batch element to convert.
        
        Returns
        -------
        torch.Tensor
            Dense tensor.
        """
        return self.to_torch_sparse(batch_idx).to_dense()
    
    def to_csr(self, batch_idx: Optional[Tuple[int, ...]] = None) -> torch.Tensor:
        """
        Convert to CSR format.
        
        Parameters
        ----------
        batch_idx : Tuple[int, ...], optional
            For batched tensors, which batch element to convert.
        
        Returns
        -------
        torch.Tensor
            PyTorch sparse CSR tensor.
        """
        return self.to_torch_sparse(batch_idx).to_sparse_csr()
    
    def partition(
        self,
        num_partitions: int,
        coords: Optional[torch.Tensor] = None,
        partition_method: str = 'auto',
        verbose: bool = False
    ) -> "DSparseTensor":
        """
        Partition into a distributed sparse tensor.
        
        Creates a DSparseTensor with automatic domain decomposition.
        This is useful for distributed computing and parallel solvers.
        
        Parameters
        ----------
        num_partitions : int
            Number of partitions to create
        coords : torch.Tensor, optional
            Node coordinates for geometric partitioning [num_nodes, dim].
            Required for 'rcb' and 'slicing' methods.
        partition_method : str
            Partitioning method:
            - 'auto': Auto-select (uses 'rcb' if coords provided, else 'metis')
            - 'metis': Graph-based partitioning (requires pymetis)
            - 'rcb': Recursive Coordinate Bisection (requires coords)
            - 'slicing': Simple coordinate slicing (requires coords)
            - 'simple': Simple 1D partitioning by node index
        verbose : bool
            Whether to print partition info
            
        Returns
        -------
        DSparseTensor
            Distributed sparse tensor with the specified partitions
            
        Example
        -------
        >>> A = SparseTensor(val, row, col, shape)
        >>> D = A.partition(num_partitions=4)
        >>> for i in range(4):
        ...     partition = D[i]
        ...     y = partition.matvec(x_local)
        
        Notes
        -----
        - Use `D.to_sparse_tensor()` to gather back to a SparseTensor
        - For distributed training, use `partition_for_rank()` instead
        """
        from .distributed import DSparseTensor
        
        if self.is_batched:
            raise ValueError("partition() does not support batched SparseTensor. "
                           "Use a 2D SparseTensor.")
        
        return DSparseTensor(
            self.values,
            self.row_indices,
            self.col_indices,
            self.sparse_shape,
            num_partitions=num_partitions,
            coords=coords,
            partition_method=partition_method,
            device=self.device,
            verbose=verbose
        )
    
    def partition_for_rank(
        self,
        rank: int,
        world_size: int,
        coords: Optional[torch.Tensor] = None,
        partition_method: str = 'simple',
        verbose: bool = False
    ) -> "DSparseMatrix":
        """
        Get partition for a specific rank in distributed environment.
        
        This is the recommended API for multi-process distributed computing.
        Each rank calls this method with its own rank ID to get its local
        partition. The partitioning is deterministic and consistent across
        all ranks.
        
        Parameters
        ----------
        rank : int
            This process's rank (0 to world_size-1)
        world_size : int
            Total number of processes
        coords : torch.Tensor, optional
            Node coordinates for geometric partitioning
        partition_method : str
            Partitioning method ('simple', 'metis', 'rcb', 'slicing')
        verbose : bool
            Print partition info
            
        Returns
        -------
        DSparseMatrix
            Local partition for this rank
            
        Example
        -------
        >>> # In multi-process code:
        >>> A = SparseTensor(val, row, col, shape)
        >>> partition = A.partition_for_rank(rank, world_size)
        >>> y_local = partition.matvec(x_local)
        
        Notes
        -----
        - This uses `DSparseTensor.from_global_distributed()` internally,
          which broadcasts partition IDs from rank 0 for consistency.
        - Requires `torch.distributed` to be initialized.
        """
        from .distributed import DSparseTensor
        
        if self.is_batched:
            raise ValueError("partition_for_rank() does not support batched SparseTensor.")
        
        return DSparseTensor.from_global_distributed(
            self.values,
            self.row_indices,
            self.col_indices,
            self.sparse_shape,
            rank=rank,
            world_size=world_size,
            coords=coords,
            partition_method=partition_method,
            verbose=verbose
        )
    
    def T(self) -> "SparseTensor":
        """
        Transpose the sparse dimensions.
        
        Returns
        -------
        SparseTensor
            Transposed tensor with row/col indices swapped.
        """
        new_shape = list(self._shape)
        dim_m, dim_n = self._sparse_dim
        new_shape[dim_m], new_shape[dim_n] = new_shape[dim_n], new_shape[dim_m]
        
        result = SparseTensor(
            self.values,
            self.col_indices,  # Swap row and col
            self.row_indices,
            tuple(new_shape),
            sparse_dim=self._sparse_dim
        )
        return result

    def conj(self) -> "SparseTensor":
        """
        Element-wise complex conjugate (same sparsity pattern).

        For real tensors this is a no-op view-equivalent (``values`` are
        returned unchanged); for complex tensors every stored value is
        conjugated. Gradients flow through ``torch.conj``.
        """
        return SparseTensor(
            self.values.conj(),
            self.row_indices,
            self.col_indices,
            tuple(self._shape),
            sparse_dim=self._sparse_dim,
        )

    def H(self) -> "SparseTensor":
        """
        Conjugate (Hermitian) transpose ``A^H = conj(A)^T``.

        For real matrices this equals :meth:`T`. For complex matrices it is
        the proper adjoint, e.g. ``A.H()`` of a Hermitian matrix returns ``A``.
        Use this (not :meth:`T`) for inner products / normal equations on
        complex systems.
        """
        new_shape = list(self._shape)
        dim_m, dim_n = self._sparse_dim
        new_shape[dim_m], new_shape[dim_n] = new_shape[dim_n], new_shape[dim_m]

        return SparseTensor(
            self.values.conj(),       # conjugate the stored values
            self.col_indices,         # swap row and col (transpose)
            self.row_indices,
            tuple(new_shape),
            sparse_dim=self._sparse_dim,
        )

    def flatten_blocks(self) -> "SparseTensor":
        """
        Flatten block dimensions into the sparse (M, N) dimensions.
        
        For a block-sparse tensor with shape [...batch, M, N, *block_shape],
        this creates a new tensor with shape [...batch, M*block_M, N*block_N]
        where each block entry becomes multiple scalar entries.
        
        Returns
        -------
        SparseTensor
            Flattened tensor without block dimensions.
            
        Example
        -------
        >>> # Block sparse: shape (10, 10, 2, 2), block_shape=(2, 2)
        >>> A = SparseTensor(val, row, col, (10, 10, 2, 2))
        >>> A_flat = A.flatten_blocks()
        >>> print(A_flat.shape)  # (20, 20)
        >>> print(A_flat.nnz)    # nnz * 4 (each block has 4 elements)
        
        Notes
        -----
        - Only works for 2D block shapes (block_M, block_N).
        - Use `unflatten_blocks(block_shape)` to reverse this operation.
        - The flattened tensor's sparsity pattern may have duplicates that
          need to be coalesced.
        """
        if not self.is_block:
            return self  # No blocks, return as is
        
        block_shape = self.block_shape
        if len(block_shape) != 2:
            raise ValueError(f"flatten_blocks only supports 2D blocks, got {block_shape}")
        
        block_M, block_N = block_shape
        M, N = self.sparse_shape
        batch_shape = self.batch_shape
        nnz = self.nnz
        
        # New sparse shape
        new_M = M * block_M
        new_N = N * block_N
        
        # Expand block entries into individual entries
        # Original: values shape [...batch, nnz, block_M, block_N]
        # New: values shape [...batch, nnz * block_M * block_N]
        
        # Create new row/col indices
        # For each (row, col) block at position (i, j), create indices:
        # (i*block_M + bi, j*block_N + bj) for bi in [0, block_M), bj in [0, block_N)
        
        row = self.row_indices  # [nnz]
        col = self.col_indices  # [nnz]
        
        # Create block offsets
        block_offsets = torch.arange(block_M * block_N, device=self.device)
        bi = block_offsets // block_N  # [block_M * block_N]
        bj = block_offsets % block_N   # [block_M * block_N]
        
        # Expand row/col to new indices
        # new_row[k * block_M * block_N + offset] = row[k] * block_M + bi[offset]
        new_row = (row.unsqueeze(-1) * block_M + bi.unsqueeze(0)).reshape(-1)  # [nnz * block_size]
        new_col = (col.unsqueeze(-1) * block_N + bj.unsqueeze(0)).reshape(-1)  # [nnz * block_size]
        
        # Flatten values
        if len(batch_shape) > 0:
            # [...batch, nnz, block_M, block_N] -> [...batch, nnz * block_M * block_N]
            vals = self.values.reshape(*batch_shape, nnz * block_M * block_N)
        else:
            # [nnz, block_M, block_N] -> [nnz * block_M * block_N]
            vals = self.values.reshape(nnz * block_M * block_N)
        
        new_shape = batch_shape + (new_M, new_N)
        
        return SparseTensor(
            vals, new_row, new_col, new_shape,
            sparse_dim=self._sparse_dim
        )
    
    def unflatten_blocks(self, block_shape: Tuple[int, int]) -> "SparseTensor":
        """
        Restore block structure from a flattened tensor.
        
        This is the inverse of `flatten_blocks()`. It groups scalar entries
        back into block entries.
        
        Parameters
        ----------
        block_shape : Tuple[int, int]
            The (block_M, block_N) dimensions to create.
            M and N must be divisible by block_M and block_N respectively.
        
        Returns
        -------
        SparseTensor
            Block-sparse tensor with the specified block shape.
            
        Example
        -------
        >>> A_flat = SparseTensor(val, row, col, (20, 20))
        >>> A_block = A_flat.unflatten_blocks((2, 2))
        >>> print(A_block.shape)  # (10, 10, 2, 2)
        >>> print(A_block.block_shape)  # (2, 2)
        
        Notes
        -----
        - Requires that the sparsity pattern is block-aligned.
        - All block entries must be present (dense within each block).
        - For sparse blocks, use `to_block_sparse()` instead.
        """
        if self.is_block:
            raise ValueError("Tensor already has block structure. Use flatten_blocks first.")
        
        if len(block_shape) != 2:
            raise ValueError(f"block_shape must be 2D, got {block_shape}")
        
        block_M, block_N = block_shape
        M, N = self.sparse_shape
        batch_shape = self.batch_shape
        
        if M % block_M != 0 or N % block_N != 0:
            raise ValueError(
                f"Sparse shape ({M}, {N}) not divisible by block_shape ({block_M}, {block_N})"
            )
        
        new_M = M // block_M
        new_N = N // block_N
        block_size = block_M * block_N
        
        row = self.row_indices
        col = self.col_indices
        nnz = self.nnz
        
        if nnz % block_size != 0:
            raise ValueError(
                f"Number of non-zeros ({nnz}) not divisible by block size ({block_size}). "
                "The sparsity pattern may not be block-aligned."
            )
        
        # Compute block indices
        block_row = row // block_M  # Which block row
        block_col = col // block_N  # Which block col
        local_row = row % block_M   # Position within block
        local_col = col % block_N   # Position within block
        
        # Group entries by (block_row, block_col)
        # Create a unique block ID for sorting
        block_id = block_row * new_N + block_col
        
        # Sort by block_id, then by local position
        local_offset = local_row * block_N + local_col
        sort_key = block_id * block_size + local_offset
        sort_idx = torch.argsort(sort_key)
        
        sorted_block_id = block_id[sort_idx]
        sorted_local_offset = local_offset[sort_idx]
        
        # Extract unique blocks
        unique_blocks, counts = torch.unique_consecutive(sorted_block_id, return_counts=True)
        
        if not torch.all(counts == block_size):
            raise ValueError(
                "Not all blocks are complete. Each block must have exactly "
                f"{block_size} entries."
            )
        
        num_blocks = unique_blocks.size(0)
        new_row_indices = unique_blocks // new_N
        new_col_indices = unique_blocks % new_N
        
        # Reshape values to include block dimensions
        if len(batch_shape) > 0:
            # Sort values: [...batch, nnz] -> [...batch, num_blocks * block_size]
            sorted_vals = self.values[..., sort_idx]
            new_vals = sorted_vals.reshape(*batch_shape, num_blocks, block_M, block_N)
        else:
            sorted_vals = self.values[sort_idx]
            new_vals = sorted_vals.reshape(num_blocks, block_M, block_N)
        
        new_shape = batch_shape + (new_M, new_N, block_M, block_N)
        
        return SparseTensor(
            new_vals, new_row_indices, new_col_indices, new_shape,
            sparse_dim=self._sparse_dim
        )
    
    # =========================================================================
    # Property Detection (returns tensor for batched)
    # =========================================================================
    
    def is_symmetric(
        self, 
        atol: float = 1e-8, 
        rtol: float = 1e-5,
        force_recompute: bool = False
    ) -> torch.Tensor:
        """
        Check if the matrix is symmetric (A == A^T).
        
        For batched tensors, checks each matrix independently and returns
        a boolean tensor with shape matching the batch dimensions.
        
        Parameters
        ----------
        atol : float, optional
            Absolute tolerance for comparison. Default: 1e-8.
        rtol : float, optional
            Relative tolerance for comparison. Default: 1e-5.
        force_recompute : bool, optional
            If True, recompute even if cached. Default: False.
        
        Returns
        -------
        torch.Tensor
            Boolean tensor with shape:
            - [] (scalar) for non-batched tensors
            - [*batch_shape] for batched tensors
        
        Examples
        --------
        >>> A = SparseTensor(val, row, col, (3, 3))
        >>> A.is_symmetric()  # tensor(True) or tensor(False)
        
        >>> A_batch = SparseTensor(val_batch, row, col, (4, 3, 3))
        >>> A_batch.is_symmetric()  # tensor([True, True, True, True])
        """
        if self._is_symmetric_cache is not None and not force_recompute:
            return self._is_symmetric_cache
        
        if not self.is_square:
            result = torch.tensor(False, device=self.device)
            if self.is_batched:
                result = result.expand(self.batch_shape)
            self._is_symmetric_cache = result
            return result
        
        row = self.row_indices
        col = self.col_indices
        M, N = self.sparse_shape
        
        # Create hash for (row, col) pairs
        forward_hash = row * N + col
        transpose_hash = col * N + row
        
        # Sort both to align
        forward_order = forward_hash.argsort()
        transpose_order = transpose_hash.argsort()
        
        sorted_forward_hash = forward_hash[forward_order]
        sorted_transpose_hash = transpose_hash[transpose_order]
        
        # Check sparsity pattern
        if not torch.equal(sorted_forward_hash, sorted_transpose_hash):
            result = torch.tensor(False, device=self.device)
            if self.is_batched:
                result = result.expand(self.batch_shape).clone()
            self._is_symmetric_cache = result
            return result
        
        # Compare values
        if self.is_batched:
            B = self.batch_size
            vals_flat = self.values.reshape(B, self.nnz)
            vals_forward = vals_flat[:, forward_order]
            vals_transpose = vals_flat[:, transpose_order]
            
            diff = (vals_forward - vals_transpose).abs()
            threshold = atol + rtol * vals_forward.abs()
            is_sym = (diff <= threshold).all(dim=-1)
            result = is_sym.reshape(self.batch_shape)
        else:
            vals_forward = self.values[forward_order]
            vals_transpose = self.values[transpose_order]
            
            diff = (vals_forward - vals_transpose).abs()
            threshold = atol + rtol * vals_forward.abs()
            result = torch.tensor((diff <= threshold).all().item(), device=self.device)
        
        self._is_symmetric_cache = result
        return result
    
    def is_positive_definite(
        self, 
        method: Literal["gershgorin", "cholesky", "eigenvalue"] = "gershgorin",
        force_recompute: bool = False
    ) -> torch.Tensor:
        """
        Check if the matrix is positive definite.
        
        For batched tensors, checks each matrix independently and returns
        a boolean tensor with shape matching the batch dimensions.
        
        Parameters
        ----------
        method : {"gershgorin", "cholesky", "eigenvalue"}, optional
            Method for checking:
            - "gershgorin": Fast check using Gershgorin circles (sufficient but not necessary)
            - "cholesky": Try Cholesky decomposition (necessary and sufficient, slower)
            - "eigenvalue": Check smallest eigenvalues (necessary and sufficient, slowest)
            Default: "gershgorin".
        force_recompute : bool, optional
            If True, recompute even if cached. Default: False.
        
        Returns
        -------
        torch.Tensor
            Boolean tensor with shape:
            - [] (scalar) for non-batched tensors
            - [*batch_shape] for batched tensors
        
        Examples
        --------
        >>> A = SparseTensor(val, row, col, (3, 3))
        >>> A.is_positive_definite()  # tensor(True) or tensor(False)
        >>> A.is_positive_definite(method="cholesky")  # More accurate check
        
        >>> A_batch = SparseTensor(val_batch, row, col, (4, 3, 3))
        >>> A_batch.is_positive_definite()  # tensor([True, True, True, True])
        """
        if self._is_positive_definite_cache is not None and not force_recompute:
            return self._is_positive_definite_cache
        
        if not self.is_square:
            result = torch.tensor(False, device=self.device)
            if self.is_batched:
                result = result.expand(self.batch_shape).clone()
            self._is_positive_definite_cache = result
            return result
        
        row = self.row_indices
        col = self.col_indices
        M, N = self.sparse_shape
        
        if method == "gershgorin":
            result = self._check_pd_gershgorin()
        elif method == "cholesky":
            result = self._check_pd_cholesky()
        else:  # eigenvalue
            result = self._check_pd_eigenvalue()
        
        self._is_positive_definite_cache = result
        return result
    
    def _check_pd_gershgorin(self) -> torch.Tensor:
        """Check positive definiteness using Gershgorin circles."""
        row = self.row_indices
        col = self.col_indices
        M, N = self.sparse_shape
        is_diag = (row == col)
        
        if self.is_batched:
            B = self.batch_size
            vals_flat = self.values.reshape(B, self.nnz)
            
            # Gershgorin works on real magnitudes; for complex (Hermitian)
            # matrices the diagonal is real, so use real parts / |a_ij|.
            real_dtype = self.values.abs().dtype

            # Diagonal elements (real part)
            diag_rows = row[is_diag]
            diag_vals = vals_flat[:, is_diag].real  # [B, num_diag]

            diag = torch.zeros(B, M, dtype=real_dtype, device=self.device)
            diag.scatter_(1, diag_rows.unsqueeze(0).expand(B, -1), diag_vals)

            # Off-diagonal sum
            is_offdiag = ~is_diag
            offdiag_rows = row[is_offdiag]
            offdiag_vals = vals_flat[:, is_offdiag].abs()  # [B, num_offdiag]

            offdiag_sum = torch.zeros(B, M, dtype=real_dtype, device=self.device)
            offdiag_sum.scatter_add_(1, offdiag_rows.unsqueeze(0).expand(B, -1), offdiag_vals)
            
            # Check: diag > offdiag_sum AND diag > 0
            is_pd = ((diag > offdiag_sum) & (diag > 0)).all(dim=-1)
            return is_pd.reshape(self.batch_shape)
        else:
            real_dtype = self.values.abs().dtype

            diag_rows = row[is_diag]
            diag_vals = self.values[is_diag].real

            diag = torch.zeros(M, dtype=real_dtype, device=self.device)
            diag.scatter_(0, diag_rows, diag_vals)

            is_offdiag = ~is_diag
            offdiag_rows = row[is_offdiag]
            offdiag_vals = self.values[is_offdiag].abs()

            offdiag_sum = torch.zeros(M, dtype=real_dtype, device=self.device)
            offdiag_sum.scatter_add_(0, offdiag_rows, offdiag_vals)
            
            is_pd = ((diag > offdiag_sum) & (diag > 0)).all()
            return torch.tensor(is_pd.item(), device=self.device)
    
    def _check_pd_cholesky(self) -> torch.Tensor:
        """Check positive definiteness using Cholesky decomposition."""
        if self.is_batched:
            results = []
            for idx in self._batch_indices():
                try:
                    A_dense = self.to_dense(idx)
                    torch.linalg.cholesky(A_dense)
                    results.append(True)
                except RuntimeError:
                    results.append(False)
            return torch.tensor(results, device=self.device).reshape(self.batch_shape)
        else:
            try:
                A_dense = self.to_dense()
                torch.linalg.cholesky(A_dense)
                return torch.tensor(True, device=self.device)
            except RuntimeError:
                return torch.tensor(False, device=self.device)
    
    def _check_pd_eigenvalue(self) -> torch.Tensor:
        """Check positive definiteness using eigenvalue computation."""
        if self.is_batched:
            results = []
            for idx in self._batch_indices():
                try:
                    A_dense = self.to_dense(idx)
                    eigenvalues = torch.linalg.eigvalsh(A_dense)
                    results.append((eigenvalues > 0).all().item())
                except Exception:
                    results.append(False)
            return torch.tensor(results, device=self.device).reshape(self.batch_shape)
        else:
            try:
                A_dense = self.to_dense()
                eigenvalues = torch.linalg.eigvalsh(A_dense)
                return torch.tensor((eigenvalues > 0).all().item(), device=self.device)
            except Exception:
                return torch.tensor(False, device=self.device)
    
    def _batch_indices(self):
        """Generate all batch index tuples."""
        import itertools
        ranges = [range(s) for s in self.batch_shape]
        return itertools.product(*ranges)
    
    # =========================================================================
    # Graph / Connected Components
    # =========================================================================
    
    def connected_components(self) -> Tuple[torch.Tensor, int]:
        """
        Find connected components of the graph represented by this sparse matrix.
        
        Uses union-find algorithm for efficiency. Treats the matrix as an
        undirected graph adjacency matrix.
        
        Returns
        -------
        labels : torch.Tensor
            Component label for each node, shape [N]. Labels are in range [0, num_components).
        num_components : int
            Number of connected components.
            
        Notes
        -----
        - Only works for non-batched 2D matrices
        - Matrix is treated as undirected (edges in either direction count)
        - Self-loops are ignored for connectivity
        
        Examples
        --------
        >>> # Block diagonal matrix with 3 components
        >>> A = SparseTensor(val, row, col, (100, 100))
        >>> labels, num_comp = A.connected_components()
        >>> print(f"Found {num_comp} components")
        """
        if self.is_batched:
            raise NotImplementedError("connected_components not supported for batched tensors")
        
        M, N = self.sparse_shape
        if M != N:
            raise ValueError("connected_components requires square matrix")
        
        # Union-Find with path compression and union by rank
        parent = torch.arange(N, device=self.device, dtype=torch.long)
        rank = torch.zeros(N, device=self.device, dtype=torch.long)
        
        def find(x: int) -> int:
            """Find root with path compression."""
            root = x
            while parent[root].item() != root:
                root = parent[root].item()
            # Path compression
            while parent[x].item() != root:
                next_x = parent[x].item()
                parent[x] = root
                x = next_x
            return root
        
        def union(x: int, y: int):
            """Union by rank."""
            rx, ry = find(x), find(y)
            if rx == ry:
                return
            if rank[rx] < rank[ry]:
                rx, ry = ry, rx
            parent[ry] = rx
            if rank[rx] == rank[ry]:
                rank[rx] += 1
        
        # Process all edges
        row = self.row_indices.cpu()
        col = self.col_indices.cpu()
        
        for i in range(len(row)):
            r, c = row[i].item(), col[i].item()
            if r != c:  # Skip self-loops
                union(r, c)
        
        # Find all roots and relabel
        labels = torch.zeros(N, dtype=torch.long, device=self.device)
        for i in range(N):
            labels[i] = find(i)
        
        # Relabel to consecutive integers starting from 0
        unique_labels = labels.unique()
        num_components = len(unique_labels)
        
        label_map = torch.zeros(N, dtype=torch.long, device=self.device)
        for new_label, old_label in enumerate(unique_labels):
            label_map[labels == old_label] = new_label
        
        return label_map, num_components
    
    def has_isolated_components(self) -> bool:
        """
        Check if the matrix has multiple connected components.
        
        Returns
        -------
        bool
            True if matrix has more than one connected component.
            
        Examples
        --------
        >>> A = SparseTensor(val, row, col, (100, 100))
        >>> if A.has_isolated_components():
        ...     components = A.to_connected_components()
        """
        _, num_components = self.connected_components()
        return num_components > 1
    
    def to_connected_components(self) -> "SparseTensorList":
        """
        Split the matrix into a list of connected component subgraphs.
        
        Each component becomes a separate SparseTensor with reindexed nodes.
        
        Returns
        -------
        SparseTensorList
            List of SparseTensors, one per connected component.
            
        Notes
        -----
        - Each component's nodes are reindexed from 0
        - Original node indices can be recovered from the mapping
        
        Examples
        --------
        >>> A = SparseTensor(val, row, col, (100, 100))
        >>> components = A.to_connected_components()
        >>> print(f"Split into {len(components)} components")
        >>> for i, comp in enumerate(components):
        ...     print(f"  Component {i}: {comp.shape}")
        """
        if self.is_batched:
            raise NotImplementedError("to_connected_components not supported for batched tensors")
        
        labels, num_components = self.connected_components()
        
        if num_components == 1:
            # Single component, return list with self
            return SparseTensorList([self])
        
        # Split into components
        components = []
        row = self.row_indices
        col = self.col_indices
        val = self.values
        
        for comp_id in range(num_components):
            # Find nodes in this component
            node_mask = (labels == comp_id)
            comp_nodes = torch.where(node_mask)[0]
            num_comp_nodes = len(comp_nodes)
            
            # Create mapping from old to new indices
            old_to_new = torch.full((self.sparse_shape[0],), -1, dtype=torch.long, device=self.device)
            old_to_new[comp_nodes] = torch.arange(num_comp_nodes, device=self.device)
            
            # Find edges within this component
            row_in_comp = node_mask[row]
            col_in_comp = node_mask[col]
            edge_mask = row_in_comp & col_in_comp
            
            # Extract and remap edges
            comp_row = old_to_new[row[edge_mask]]
            comp_col = old_to_new[col[edge_mask]]
            comp_val = val[edge_mask]
            
            comp_sparse = SparseTensor(
                comp_val, comp_row, comp_col,
                (num_comp_nodes, num_comp_nodes)
            )
            components.append(comp_sparse)
        
        return SparseTensorList(components)
    
    # =========================================================================
    # Matrix Multiplication
    # =========================================================================
    
    def _spmv_coo(self, x: torch.Tensor) -> torch.Tensor:
        """
        Sparse matrix-vector/matrix multiply using COO format with scatter_add.
        
        Computes A @ x where A is this sparse tensor and x is dense.
        Works on any device without explicit CSR conversion.
        
        Parameters
        ----------
        x : torch.Tensor
            Dense tensor to multiply. Shape depends on batching:
            - Non-batched: [N] or [N, K]
            - Batched: [B, N] or [B, N, K]
        
        Returns
        -------
        torch.Tensor
            Result of A @ x.
        """
        row = self.row_indices
        col = self.col_indices
        M, N = self.sparse_shape
        
        if self.is_batched:
            batch_shape = self.batch_shape
            B = self.batch_size
            vals_flat = self.values.reshape(B, self.nnz)
            # Determine output dtype via type promotion
            out_dtype = torch.result_type(self.values, x)
            
            if x.dim() == 1:
                # x: [N] - same for all batches -> result [B, M]
                x_gathered = x[col]
                products = vals_flat * x_gathered
                result = torch.zeros(B, M, dtype=out_dtype, device=self.device)
                row_expanded = row.unsqueeze(0).expand(B, -1)
                result.scatter_add_(1, row_expanded, products)
                return result.reshape(*batch_shape, M)
            
            elif x.dim() == len(batch_shape) + 1:
                # x: [...batch, N] -> result [...batch, M]
                x_flat = x.reshape(B, N)
                x_gathered = x_flat[:, col]
                products = vals_flat * x_gathered
                result = torch.zeros(B, M, dtype=out_dtype, device=self.device)
                row_expanded = row.unsqueeze(0).expand(B, -1)
                result.scatter_add_(1, row_expanded, products)
                return result.reshape(*batch_shape, M)
            else:
                # x: [...batch, N, K] -> result [...batch, M, K]
                K = x.size(-1)
                x_flat = x.reshape(B, N, K)
                x_gathered = x_flat[:, col, :]
                products = vals_flat.unsqueeze(-1) * x_gathered
                result = torch.zeros(B, M, K, dtype=out_dtype, device=self.device)
                row_expanded = row.unsqueeze(0).unsqueeze(-1).expand(B, -1, K)
                result.scatter_add_(1, row_expanded, products)
                return result.reshape(*batch_shape, M, K)
        else:
            # Determine output dtype via type promotion (handles float32 @ float64, etc.)
            out_dtype = torch.result_type(self.values, x)
            
            if x.dim() == 1:
                x_gathered = x[col]
                products = self.values * x_gathered
                result = torch.zeros(M, dtype=out_dtype, device=self.device)
                result.scatter_add_(0, row, products)
                return result
            else:
                K = x.size(1)
                x_gathered = x[col]
                products = self.values.unsqueeze(1) * x_gathered
                result = torch.zeros(M, K, dtype=out_dtype, device=self.device)
                row_expanded = row.unsqueeze(1).expand(-1, K)
                result.scatter_add_(0, row_expanded, products)
                return result
    
    def _dense_sparse_mm(self, X: torch.Tensor) -> torch.Tensor:
        """
        Dense @ Sparse: X @ A where X is [..., M] or [..., K, M], A is [..., M, N].
        
        Parameters
        ----------
        X : torch.Tensor
            Dense tensor.
        
        Returns
        -------
        torch.Tensor
            Result of X @ A.
        """
        row = self.row_indices
        col = self.col_indices
        M, N = self.sparse_shape
        
        if self.is_batched:
            batch_shape = self.batch_shape
            B = self.batch_size
            vals_flat = self.values.reshape(B, self.nnz)
            # Determine output dtype via type promotion
            out_dtype = torch.result_type(self.values, X)
            
            if X.dim() == 1:
                X_gathered = X[row]
                products = vals_flat * X_gathered
                result = torch.zeros(B, N, dtype=out_dtype, device=self.device)
                col_expanded = col.unsqueeze(0).expand(B, -1)
                result.scatter_add_(1, col_expanded, products)
                return result.reshape(*batch_shape, N)
            
            elif X.dim() == len(batch_shape) + 1:
                X_flat = X.reshape(B, M)
                X_gathered = X_flat[:, row]
                products = vals_flat * X_gathered
                result = torch.zeros(B, N, dtype=out_dtype, device=self.device)
                col_expanded = col.unsqueeze(0).expand(B, -1)
                result.scatter_add_(1, col_expanded, products)
                return result.reshape(*batch_shape, N)
            
            else:
                K = X.size(-2)
                X_flat = X.reshape(B, K, M)
                X_gathered = X_flat[:, :, row]
                products = vals_flat.unsqueeze(1) * X_gathered
                result = torch.zeros(B, K, N, dtype=out_dtype, device=self.device)
                col_expanded = col.unsqueeze(0).unsqueeze(0).expand(B, K, -1)
                result.scatter_add_(2, col_expanded, products)
                return result.reshape(*batch_shape, K, N)
        else:
            # Determine output dtype via type promotion
            out_dtype = torch.result_type(self.values, X)
            
            if X.dim() == 1:
                X_gathered = X[row]
                products = self.values * X_gathered
                result = torch.zeros(N, dtype=out_dtype, device=self.device)
                result.scatter_add_(0, col, products)
                return result
            else:
                K = X.size(0)
                X_gathered = X[:, row]
                products = self.values.unsqueeze(0) * X_gathered
                result = torch.zeros(K, N, dtype=out_dtype, device=self.device)
                col_expanded = col.unsqueeze(0).expand(K, -1)
                result.scatter_add_(1, col_expanded, products)
                return result
    
    def _spsp_multiply(self, other: "SparseTensor") -> "SparseTensor":
        """
        Sparse-Sparse multiplication: A @ B where both are sparse.
        
        Uses custom autograd function to provide SPARSE gradients.
        Memory usage is O(nnz) not O(M*N).
        
        Parameters
        ----------
        other : SparseTensor
            Right-hand side sparse matrix.
        
        Returns
        -------
        SparseTensor
            Result C = A @ B.
        """
        M, K = self.sparse_shape
        K2, N = other.sparse_shape
        if K != K2:
            raise ValueError(f"Inner dimensions don't match: {K} vs {K2}")
        
        C_values, C_row, C_col, C_shape = _sparse_sparse_matmul_with_sparse_grad(
            self.values, self.row_indices, self.col_indices, (M, K),
            other.values, other.row_indices, other.col_indices, (K, N)
        )
        
        return SparseTensor(C_values, C_row, C_col, C_shape)
    
    def __matmul__(self, other: Union[torch.Tensor, "SparseTensor"]) -> Union[torch.Tensor, "SparseTensor"]:
        """
        Matrix multiplication: A @ other.
        
        Supports:
        - Sparse @ Dense vector: A @ x -> y
        - Sparse @ Dense matrix: A @ X -> Y
        - Sparse @ Sparse: A @ B -> C (with sparse gradients)
        
        Parameters
        ----------
        other : torch.Tensor or SparseTensor
            Right-hand side operand.
        
        Returns
        -------
        torch.Tensor or SparseTensor
            Result of multiplication.
        """
        if isinstance(other, SparseTensor):
            return self._spsp_multiply(other)
        return self._spmv_coo(other)
    
    def __rmatmul__(self, other: torch.Tensor) -> torch.Tensor:
        """
        Dense @ Sparse multiplication: other @ A.
        
        Parameters
        ----------
        other : torch.Tensor
            Left-hand side dense tensor.
        
        Returns
        -------
        torch.Tensor
            Result of multiplication.
        """
        return self._dense_sparse_mm(other)
    
    # =========================================================================
    # Linear Solve
    # =========================================================================
    
    def solve(
        self,
        b: torch.Tensor,
        backend: BackendType = "auto",
        method: MethodType = "auto",
        atol: float = 1e-10,
        maxiter: int = 10000,
        tol: float = 1e-12,
        verbose: bool = False,
    ) -> torch.Tensor:
        """
        Solve the sparse linear system Ax = b.
        
        Automatically handles batched tensors: if A is [...batch, M, N] and
        b is [...batch, M], returns x with shape [...batch, N].
        
        Parameters
        ----------
        b : torch.Tensor
            Right-hand side vector(s). Shape:
            - Non-batched: [M] or [M, K] for multiple RHS
            - Batched: [...batch, M] or [...batch, M, K]
        backend : {"auto", "scipy", "eigen", "cupy", "cudss"}, optional
            Solver backend. Default: "auto" (selects based on device).
            - "scipy": Uses SciPy's sparse solvers (CPU only)
            - "eigen": Uses Eigen C++ library (CPU only)
            - "cupy": Uses CuPy's sparse solvers (CUDA only)
            - "cudss": Uses NVIDIA cuDSS (CUDA only)
        method : str, optional
            Solver method. Default: "auto" (selects based on matrix properties).
            - Direct methods: "lu", "umfpack", "cholesky", "ldlt"
            - Iterative methods: "cg", "bicgstab", "gmres", "minres"
        atol : float, optional
            Absolute tolerance for iterative solvers. Default: 1e-10.
        maxiter : int, optional
            Maximum iterations for iterative solvers. Default: 10000.
        tol : float, optional
            Relative tolerance for direct solvers. Default: 1e-12.
        verbose : bool, optional
            If True, print a one-line summary of the auto-selected backend,
            method, and detected matrix properties (symmetric, SPD).
            Default: False.

        Returns
        -------
        torch.Tensor
            Solution x with same batch shape as b.
        
        Raises
        ------
        ValueError
            If matrix is not square.
        NotImplementedError
            If block sparse tensors are used (not yet supported).
        
        Examples
        --------
        >>> # Simple solve
        >>> A = SparseTensor(val, row, col, (3, 3))
        >>> b = torch.randn(3)
        >>> x = A.solve(b)
        
        >>> # Batched solve
        >>> A_batch = SparseTensor(val_batch, row, col, (4, 3, 3))
        >>> b_batch = torch.randn(4, 3)
        >>> x_batch = A_batch.solve(b_batch)
        
        >>> # Specify backend
        >>> x = A.solve(b, backend='scipy', method='cg')
        """
        if not self.is_square:
            raise ValueError("Matrix must be square for solve()")

        if self.is_block:
            raise NotImplementedError("solve() not yet supported for block sparse tensors")

        # Get matrix properties
        is_sym = self.is_symmetric().all().item() if self.is_batched else self.is_symmetric().item()
        is_pd = self.is_positive_definite().all().item() if self.is_batched else self.is_positive_definite().item()
        is_spd = is_sym and is_pd

        from .linear_solve import spsolve

        M, N = self.sparse_shape

        if self.is_batched:
            batch_shape = self.batch_shape
            vals_flat = self.values.reshape(-1, self.nnz)
            b_flat = b.reshape(-1, M)

            results = []
            for i in range(vals_flat.size(0)):
                # Only print on the first batch element to avoid spam
                x = spsolve(
                    vals_flat[i], self.row_indices, self.col_indices,
                    (M, N), b_flat[i],
                    backend=backend, method=method,
                    atol=atol, maxiter=maxiter, tol=tol,
                    is_symmetric=is_sym, is_spd=is_spd,
                    verbose=verbose and i == 0,
                )
                results.append(x)

            return torch.stack(results).reshape(*batch_shape, N)
        else:
            return spsolve(
                self.values, self.row_indices, self.col_indices,
                (M, N), b,
                backend=backend, method=method,
                atol=atol, maxiter=maxiter, tol=tol,
                is_symmetric=is_sym, is_spd=is_spd,
                verbose=verbose,
            )
    
    def solve_batch(
        self,
        values: torch.Tensor,
        b: torch.Tensor,
        backend: BackendType = "auto",
        method: MethodType = "auto",
        atol: float = 1e-10,
        maxiter: int = 10000,
        tol: float = 1e-12
    ) -> torch.Tensor:
        """
        Solve with different values but same sparsity structure.
        
        This is efficient when you have the same structure but different values
        (e.g., time-stepping, optimization, parameter sweeps).
        
        Parameters
        ----------
        values : torch.Tensor
            Matrix values. Shape [...batch, nnz] where ... are batch dimensions.
            All matrices share the same row_indices and col_indices.
        b : torch.Tensor
            Right-hand side. Shape [...batch, M].
        backend : {"auto", "scipy", "eigen", "cupy", "cudss"}, optional
            Solver backend. See solve() for details. Default: "auto".
        method : str, optional
            Solver method. See solve() for details. Default: "auto".
        atol : float, optional
            Absolute tolerance for iterative solvers. Default: 1e-10.
        maxiter : int, optional
            Maximum iterations for iterative solvers. Default: 10000.
        tol : float, optional
            Relative tolerance. Default: 1e-12.
        
        Returns
        -------
        torch.Tensor
            Solution x with shape [...batch, N].
        
        Examples
        --------
        >>> # Template matrix
        >>> A = SparseTensor(val, row, col, (10, 10))
        
        >>> # Batch of different values
        >>> val_batch = torch.stack([val * (1 + 0.1*i) for i in range(4)])  # [4, nnz]
        >>> b_batch = torch.randn(4, 10)
        
        >>> # Solve all at once
        >>> x_batch = A.solve_batch(val_batch, b_batch)  # [4, 10]
        """
        from .linear_solve import spsolve
        
        M, N = self.sparse_shape
        
        # Check properties using first batch element
        temp = SparseTensor(values[0] if values.dim() > 1 else values, 
                           self.row_indices, self.col_indices, (M, N))
        is_sym = temp.is_symmetric().item()
        is_pd = temp.is_positive_definite().item()
        is_spd = is_sym and is_pd
        
        if values.dim() > 1:
            batch_shape = values.shape[:-1]
            vals_flat = values.reshape(-1, self.nnz)
            b_flat = b.reshape(-1, M)
            
            results = []
            for i in range(vals_flat.size(0)):
                x = spsolve(
                    vals_flat[i], self.row_indices, self.col_indices, (M, N), b_flat[i],
                    backend=backend, method=method,
                    atol=atol, maxiter=maxiter, tol=tol,
                    is_symmetric=is_sym, is_spd=is_spd
                )
                results.append(x)
            
            return torch.stack(results).reshape(*batch_shape, N)
        else:
            return spsolve(
                values, self.row_indices, self.col_indices, (M, N), b,
                backend=backend, method=method,
                atol=atol, maxiter=maxiter, tol=tol,
                is_symmetric=is_sym, is_spd=is_spd
            )
    
    def nonlinear_solve(
        self,
        residual_fn,
        u0: torch.Tensor,
        *params,
        method: Literal['newton', 'picard', 'anderson'] = 'newton',
        tol: float = 1e-6,
        atol: float = 1e-10,
        max_iter: int = 50,
        line_search: bool = True,
        verbose: bool = False,
        linear_solver: BackendType = 'pytorch',
        linear_method: MethodType = 'cg',
    ) -> torch.Tensor:
        """
        Solve nonlinear equation F(u, A, θ) = 0 with adjoint-based gradients.
        
        The SparseTensor A is automatically passed as the first parameter to
        the residual function, enabling gradients to flow through A's values.
        
        Parameters
        ----------
        residual_fn : Callable
            Function F(u, A, *params) -> residual tensor.
            - u: Current solution estimate
            - A: This SparseTensor (passed automatically)
            - *params: Additional parameters with requires_grad=True
        u0 : torch.Tensor
            Initial guess for solution.
        *params : torch.Tensor
            Additional parameters (e.g., boundary conditions, coefficients).
            Tensors with requires_grad=True will receive gradients.
        method : {'newton', 'picard', 'anderson'}, optional
            Nonlinear solver method:
            - 'newton': Newton-Raphson with line search (default, fast)
            - 'picard': Fixed-point iteration (simple, slow)
            - 'anderson': Anderson acceleration (memory efficient)
        tol : float, optional
            Relative convergence tolerance. Default: 1e-6.
        atol : float, optional
            Absolute convergence tolerance. Default: 1e-10.
        max_iter : int, optional
            Maximum nonlinear iterations. Default: 50.
        line_search : bool, optional
            Use Armijo line search for Newton. Default: True.
        verbose : bool, optional
            Print convergence information. Default: False.
        linear_solver : str, optional
            Backend for linear solves. Default: 'pytorch'.
        linear_method : str, optional
            Method for linear solves. Default: 'cg'.
        
        Returns
        -------
        torch.Tensor
            Solution u* satisfying F(u*, A, θ) ≈ 0.
        
        Examples
        --------
        >>> # Nonlinear PDE: A @ u + u² = f
        >>> def residual(u, A, f):
        ...     return A @ u + u**2 - f
        ...
        >>> A = SparseTensor(val, row, col, (n, n))
        >>> f = torch.randn(n, requires_grad=True)
        >>> u0 = torch.zeros(n)
        >>> 
        >>> u = A.nonlinear_solve(residual, u0, f, method='newton')
        >>> 
        >>> # Gradients flow via adjoint method
        >>> loss = u.sum()
        >>> loss.backward()
        >>> print(f.grad)  # ∂u/∂f
        >>> print(A.values.grad)  # ∂u/∂A (if A.values.requires_grad)
        
        >>> # Nonlinear elasticity: K(u) @ u = F
        >>> def residual_elasticity(u, K, F, material):
        ...     # K depends on displacement through material nonlinearity
        ...     return K @ u - F + material * u**3
        ...
        >>> u = K.nonlinear_solve(residual_elasticity, u0, F, material)
        """
        from .nonlinear_solve import nonlinear_solve as _nonlinear_solve
        
        # Wrap residual_fn to pass SparseTensor as matvec
        M, N = self.sparse_shape
        
        def wrapped_residual(u, *all_params):
            # First param is the values tensor, rest are user params
            # Reconstruct sparse matvec capability
            return residual_fn(u, self, *all_params)
        
        # Include self.values in params if it requires grad
        all_params = params
        
        return _nonlinear_solve(
            wrapped_residual, u0, *all_params,
            method=method, tol=tol, atol=atol, max_iter=max_iter,
            line_search=line_search, verbose=verbose,
            linear_solver=linear_solver, linear_method=linear_method,
            )
    
    # =========================================================================
    # Norms
    # =========================================================================
    
    def norm(self, ord: Literal['fro', 1, 2] = 'fro') -> torch.Tensor:
        """
        Compute matrix norm.
        
        For batched tensors, returns norm for each batch element.
        
        Parameters
        ----------
        ord : {'fro', 1, 2}, optional
            Norm type:
            - 'fro': Frobenius norm (default)
            - 1: Maximum absolute column sum
            - 2: Spectral norm (largest singular value)
            
        Returns
        -------
        torch.Tensor
            Norm value(s). Shape [] for non-batched, [*batch_shape] for batched.
        
        Examples
        --------
        >>> A = SparseTensor(val, row, col, (3, 3))
        >>> A.norm('fro')  # tensor(5.0)
        
        >>> A_batch = SparseTensor(val_batch, row, col, (4, 3, 3))
        >>> A_batch.norm('fro')  # tensor([5.0, 5.0, 5.0, 5.0])
        """
        if self.is_batched:
            batch_shape = self.batch_shape
            vals_flat = self.values.reshape(-1, self.nnz)
            norms = []
            for i in range(vals_flat.size(0)):
                if ord == 'fro':
                    norms.append(vals_flat[i].norm())
                else:
                    idx = self._flat_to_batch_idx(i)
                    A_dense = self.to_dense(idx)
                    norms.append(torch.linalg.norm(A_dense, ord=ord))
            return torch.stack(norms).reshape(*batch_shape)
        else:
            if ord == 'fro':
                return self.values.norm()
            if self.is_cuda or not is_scipy_available():
                A = self.to_dense()
                return torch.linalg.norm(A, ord=ord)
            M, N = self.sparse_shape
            return scipy_norm(self.values, self.row_indices, self.col_indices, (M, N), ord=ord)
    
    def _flat_to_batch_idx(self, flat_idx: int) -> Tuple[int, ...]:
        """Convert flat batch index to tuple."""
        idx = []
        for s in reversed(self.batch_shape):
            idx.append(flat_idx % s)
            flat_idx //= s
        return tuple(reversed(idx))
    
    # =========================================================================
    # Visualization
    # =========================================================================
    
    def spy(
        self,
        batch_idx: Optional[Tuple[int, ...]] = None,
        ax=None,
        title: Optional[str] = None,
        cmap: str = 'viridis',
        show_grid: bool = True,
        grid_color: str = '#cccccc',
        grid_linewidth: float = 0.5,
        show_colorbar: bool = True,
        figsize: Tuple[float, float] = (8, 8),
        save_path: Optional[str] = None,
        dpi: int = 150,
    ):
        """
        Visualize the sparsity pattern with values shown as color intensity.
        
        Creates a spy plot where each matrix element is rendered as a pixel.
        Non-zero elements are colored with intensity proportional to the absolute
        value, while zero elements are shown as white. This provides a pixel-perfect
        visualization without overlapping markers.
        
        Parameters
        ----------
        batch_idx : Tuple[int, ...], optional
            For batched tensors, which batch element to visualize.
            Required if the tensor is batched.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates a new figure.
        title : str, optional
            Plot title. Defaults to showing matrix info.
        cmap : str, optional
            Colormap for values. Default: 'viridis'.
            Other options: 'plasma', 'hot', 'coolwarm', 'Greys', etc.
        show_grid : bool, optional
            Whether to show grid lines (only for matrices <= 30x30). Default: True.
        grid_color : str, optional
            Color of grid lines. Default: '#cccccc' (light gray).
        grid_linewidth : float, optional
            Width of grid lines. Default: 0.5.
        show_colorbar : bool, optional
            Whether to show colorbar for values. Default: True.
        figsize : Tuple[float, float], optional
            Figure size in inches. Default: (8, 8).
        save_path : str, optional
            If provided, save figure to this path.
        dpi : int, optional
            DPI for saved figure. Default: 150.
            
        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes object with the plot.
            
        Examples
        --------
        >>> A = SparseTensor(val, row, col, (100, 100))
        >>> A.spy()  # Basic spy plot
        >>> A.spy(cmap='hot', show_grid=False)  # Custom colormap, no grid
        >>> A.spy(save_path='matrix.png')  # Save to file
        
        >>> # For batched tensor
        >>> A_batch = SparseTensor(val_batch, row, col, (4, 100, 100))
        >>> A_batch.spy(batch_idx=(0,))  # Visualize first batch element
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.colors as mcolors
            from matplotlib.collections import PathCollection
        except ImportError:
            raise ImportError("matplotlib is required for spy(). Install with: pip install matplotlib")
        
        # Get indices and values
        if self.is_batched:
            if batch_idx is None:
                raise ValueError("batch_idx is required for batched tensors")
            # Flatten batch_idx to linear index
            flat_idx = 0
            for i, (idx, s) in enumerate(zip(batch_idx, self.batch_shape)):
                flat_idx = flat_idx * s + idx
            vals = self.values.reshape(-1, self.nnz)[flat_idx]
        else:
            vals = self.values
        
        row = self.row_indices.cpu().numpy()
        col = self.col_indices.cpu().numpy()
        vals_np = vals.abs().cpu().numpy()
        
        M, N = self.sparse_shape
        
        # Create figure if needed
        created_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            created_fig = True
        else:
            fig = ax.get_figure()
        
        # Normalize values for colormap
        if vals_np.max() > 0:
            vals_norm = vals_np / vals_np.max()
        else:
            vals_norm = vals_np
        
        # Build a dense image for visualization
        # Use NaN for empty cells (will be shown as white)
        import numpy as np
        image = np.full((M, N), np.nan, dtype=np.float32)
        image[row, col] = vals_norm
        
        # Create a colormap with white for NaN values
        cmap_obj = plt.cm.get_cmap(cmap).copy()
        cmap_obj.set_bad(color='white')
        
        # Use imshow for pixel-perfect rendering
        im = ax.imshow(
            image,
            cmap=cmap_obj,
            aspect='equal',
            interpolation='nearest',
            vmin=0, vmax=1,
            origin='upper'
        )
        
        # Add colorbar
        if show_colorbar:
            cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
            cbar.set_label('|value| (normalized)', fontsize=10)
        
        # Clean up axes - hide ticks for cleaner look
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add border
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('#333333')
            spine.set_linewidth(1)
        
        # Add grid only for small matrices
        if show_grid and max(M, N) <= 30:
            ax.set_xticks([i - 0.5 for i in range(N + 1)], minor=True)
            ax.set_yticks([i - 0.5 for i in range(M + 1)], minor=True)
            ax.grid(which='minor', color=grid_color, linewidth=grid_linewidth)
            ax.tick_params(which='minor', length=0)
        
        # Set title
        if title is None:
            nnz = len(vals_np)
            sparsity = 1 - nnz / (M * N)
            title = f'Sparse Matrix: {M}×{N}, nnz={nnz:,}, sparsity={sparsity:.1%}'
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        # Tight layout
        if created_fig:
            plt.tight_layout()
        
        # Save if requested
        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        
        return ax
    
    # =========================================================================
    # Eigenvalues and SVD
    # =========================================================================
    
    def eigs(
        self,
        k: int = 6,
        which: str = "LM",
        sigma: Optional[float] = None,
        return_eigenvectors: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute k eigenvalues and eigenvectors.
        
        For batched tensors, computes for each batch element.
        For CUDA tensors, uses LOBPCG algorithm.
        
        Parameters
        ----------
        k : int, optional
            Number of eigenvalues to compute. Default: 6.
        which : {"LM", "SM", "LR", "SR", "LA", "SA"}, optional
            Which eigenvalues to find:
            - "LM": Largest magnitude (default)
            - "SM": Smallest magnitude
            - "LR"/"SR": Largest/smallest real part
            - "LA"/"SA": Largest/smallest algebraic (for symmetric)
        sigma : float, optional
            Find eigenvalues near sigma (shift-invert mode).
        return_eigenvectors : bool, optional
            Whether to return eigenvectors. Default: True.
            
        Returns
        -------
        eigenvalues : torch.Tensor
            Shape [k] for non-batched, [*batch_shape, k] for batched.
        eigenvectors : torch.Tensor or None
            Shape [M, k] for non-batched, [*batch_shape, M, k] for batched.
            None if return_eigenvectors is False.
        
        Notes
        -----
        **Gradient Support:**
        
        - Both CPU and CUDA: Fully differentiable via adjoint method
        - Uses O(1) graph nodes regardless of iteration count
        - For symmetric matrices, prefer eigsh() for efficiency
        
        **Warning**: For non-symmetric matrices with complex eigenvalues,
        gradient computation is only supported for the real part.
        
        Examples
        --------
        >>> A = SparseTensor(val.requires_grad_(True), row, col, (n, n))
        >>> eigenvalues, eigenvectors = A.eigs(k=3)
        >>> loss = eigenvalues.real.sum()  # For complex eigenvalues
        >>> loss.backward()
        """
        M, N = self.sparse_shape
        
        if self.is_batched:
            batch_shape = self.batch_shape
            eigenvalues_list = []
            eigenvectors_list = []
            
            for idx in self._batch_indices():
                A_single = SparseTensor(
                    self.values[idx], self.row_indices, self.col_indices, (M, N)
                )
                evals, evecs = A_single.eigs(k, which, sigma, return_eigenvectors)
                eigenvalues_list.append(evals)
                if return_eigenvectors:
                    eigenvectors_list.append(evecs)
            
            eigenvalues = torch.stack(eigenvalues_list).reshape(*batch_shape, k)
            if return_eigenvectors:
                eigenvectors = torch.stack(eigenvectors_list).reshape(*batch_shape, M, k)
                return eigenvalues, eigenvectors
            return eigenvalues, None
        
        # For symmetric matrices or when using LA/SA, use eigsh (more efficient)
        if which in ("LA", "SA") or self.is_symmetric().item():
            return self.eigsh(k=k, which=which, sigma=sigma, return_eigenvectors=return_eigenvectors)
        
        # Use adjoint-based eigs for differentiability on all devices
        return EigshAdjoint.apply(
            self.values, self.row_indices, self.col_indices, (M, N),
            k, which, return_eigenvectors, self.device
        )
    
    def eigsh(
        self,
        k: int = 6,
        which: str = "LM",
        sigma: Optional[float] = None,
        return_eigenvectors: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute k eigenvalues for symmetric matrices.
        
        More efficient than eigs() for symmetric matrices.
        
        Parameters
        ----------
        k : int, optional
            Number of eigenvalues to compute. Default: 6.
        which : {"LM", "SM", "LA", "SA"}, optional
            Which eigenvalues to find:
            - "LM": Largest magnitude (default)
            - "SM": Smallest magnitude
            - "LA"/"SA": Largest/smallest algebraic
        sigma : float, optional
            Find eigenvalues near sigma.
        return_eigenvectors : bool, optional
            Whether to return eigenvectors. Default: True.
            
        Returns
        -------
        eigenvalues : torch.Tensor
            Shape [k] for non-batched, [*batch_shape, k] for batched.
        eigenvectors : torch.Tensor or None
            Shape [M, k] for non-batched, [*batch_shape, M, k] for batched.
        
        Notes
        -----
        **Gradient Support:**
        
        - Both CPU and CUDA: Fully differentiable via adjoint method
        - Uses O(1) graph nodes regardless of iteration count
        - Gradient computed as: ∂L/∂A = Σ_i (∂L/∂λ_i) * v_i @ v_i.T
        
        Examples
        --------
        >>> A = SparseTensor(val.requires_grad_(True), row, col, (n, n))
        >>> eigenvalues, eigenvectors = A.eigsh(k=3)
        >>> loss = eigenvalues.sum()
        >>> loss.backward()  # Computes ∂loss/∂val
        """
        M, N = self.sparse_shape
        
        if self.is_batched:
            batch_shape = self.batch_shape
            eigenvalues_list = []
            eigenvectors_list = []
            
            for idx in self._batch_indices():
                A_single = SparseTensor(
                    self.values[idx], self.row_indices, self.col_indices, (M, N)
                )
                evals, evecs = A_single.eigsh(k, which, sigma, return_eigenvectors)
                eigenvalues_list.append(evals)
                if return_eigenvectors:
                    eigenvectors_list.append(evecs)
            
            eigenvalues = torch.stack(eigenvalues_list).reshape(*batch_shape, k)
            if return_eigenvectors:
                eigenvectors = torch.stack(eigenvectors_list).reshape(*batch_shape, M, k)
                return eigenvalues, eigenvectors
            return eigenvalues, None
        
        # Use adjoint-based eigsh for differentiability on all devices
        return EigshAdjoint.apply(
            self.values, self.row_indices, self.col_indices, (M, N),
            k, which, return_eigenvectors, self.device
        )
    
    def svd(self, k: int = 6) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute truncated SVD.
        
        Parameters
        ----------
        k : int, optional
            Number of singular values to compute. Default: 6.
            
        Returns
        -------
        U : torch.Tensor
            Left singular vectors. Shape [M, k] or [*batch_shape, M, k].
        S : torch.Tensor
            Singular values. Shape [k] or [*batch_shape, k].
        Vt : torch.Tensor
            Right singular vectors. Shape [k, N] or [*batch_shape, k, N].
        
        Notes
        -----
        **Gradient Support:**
        
        - CUDA: Fully differentiable (uses power iteration with PyTorch operations)
        - CPU: NOT differentiable (uses SciPy which breaks gradient chain)
        
        For differentiable SVD on CPU, use `A.to_dense()` and `torch.linalg.svd()`.
        """
        M, N = self.sparse_shape
    
        if self.is_batched:
            batch_shape = self.batch_shape
            U_list, S_list, Vt_list = [], [], []
            
            for idx in self._batch_indices():
                A_single = SparseTensor(
                    self.values[idx], self.row_indices, self.col_indices, (M, N)
                )
                U, S, Vt = A_single.svd(k)
                U_list.append(U)
                S_list.append(S)
                Vt_list.append(Vt)
            
            U = torch.stack(U_list).reshape(*batch_shape, M, k)
            S = torch.stack(S_list).reshape(*batch_shape, k)
            Vt = torch.stack(Vt_list).reshape(*batch_shape, k, N)
            return U, S, Vt
        
        if self.is_cuda:
            matvec = lambda x: self._spmv_coo(x)
            matvec_T = lambda x: self.T()._spmv_coo(x)
            U, S, Vt = _power_iteration_svd(matvec, matvec_T, M, N, k, self.dtype, self.device)
            return U, S, Vt
        
        if not is_scipy_available():
            raise RuntimeError("SciPy is required for SVD on CPU")
        
        return scipy_svds(self.values, self.row_indices, self.col_indices, (M, N), k=k)
    
    def condition_number(self, ord: int = 2) -> torch.Tensor:
        """
        Estimate condition number.
        
        Parameters
        ----------
        ord : int, optional
            Norm order for condition number. Default: 2 (spectral).
            
        Returns
        -------
        torch.Tensor
            Condition number. Shape [] or [*batch_shape].
        """
        M, N = self.sparse_shape
        
        if self.is_batched:
            batch_shape = self.batch_shape
            cond_list = []
            
            for idx in self._batch_indices():
                A_single = SparseTensor(
                    self.values[idx], self.row_indices, self.col_indices, (M, N)
                )
                cond_list.append(A_single.condition_number(ord))
            
            return torch.stack(cond_list).reshape(*batch_shape)
        
        if ord == 2:
            k = min(6, min(M, N) - 2)
            if k < 2:
                A_dense = self.to_dense()
                S = torch.linalg.svdvals(A_dense)
                return S.max() / S.min()
            _, S, _ = self.svd(k=k)
            return S.max() / S.min()
        
        norm_A = self.norm(ord=ord)
        e = torch.randn(M, dtype=self.dtype, device=self.device)
        e = e / e.norm()
        x = self.solve(e)
        return norm_A * x.norm() / e.norm()
    
    def det(self) -> torch.Tensor:
        """
        Compute determinant of the sparse matrix with gradient support.
        
        Uses LU decomposition (CPU) or dense conversion (CUDA) to compute 
        the determinant efficiently. Supports automatic differentiation via
        the adjoint method.
        
        Returns
        -------
        torch.Tensor
            Determinant value. Shape [] for single matrix or [*batch_shape] for batched.
            
        Raises
        ------
        ValueError
            If matrix is not square
            
        Notes
        -----
        - Only square matrices have determinants
        - For large matrices, determinant values can overflow/underflow
        - Consider using log-determinant for numerical stability in such cases
        - Supports both CPU (via SciPy) and CUDA (via torch.linalg.det)
        - For batched tensors, computes determinant independently for each batch
        - Fully differentiable: gradients computed via adjoint method
        - Gradient formula: ∂det(A)/∂A = det(A) * (A^{-1})^T
        
        Performance Warning
        -------------------
        **CUDA performance is significantly slower than CPU for sparse matrices!**
        
        - CPU: Uses sparse LU decomposition (O(nnz^1.5)), ~0.3-0.8ms for n=10-1000
        - CUDA: Converts to dense (O(n²) memory + O(n³) compute), ~0.2-2.5ms
        
        The CUDA version requires converting the sparse matrix to dense format
        because cuDSS doesn't expose determinant computation for sparse
        matrices. This makes it inefficient for large sparse matrices.
        
        **Recommendation**: For sparse matrices, use `.cpu().det().cuda()` instead:
        
        >>> # Slow: CUDA with dense conversion
        >>> det_slow = A_cuda.det()  # ~2.5ms for n=1000
        >>> 
        >>> # Fast: CPU with sparse LU
        >>> det_fast = A_cuda.cpu().det()  # ~0.8ms for n=1000
        >>> det_fast = det_fast.cuda()  # Move result back if needed
        
        Examples
        --------
        >>> # Simple 2x2 matrix
        >>> val = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
        >>> row = torch.tensor([0, 0, 1, 1])
        >>> col = torch.tensor([0, 1, 0, 1])
        >>> A = SparseTensor(val, row, col, (2, 2))
        >>> det = A.det()
        >>> print(det)  # Should be -2.0
        >>> det.backward()
        >>> print(val.grad)  # Gradient w.r.t. matrix values
        >>>
        >>> # CUDA support
        >>> A_cuda = A.cuda()
        >>> det_cuda = A_cuda.det()
        >>>
        >>> # Batched matrices
        >>> val_batch = val.unsqueeze(0).expand(3, -1).clone()
        >>> A_batch = SparseTensor(val_batch, row, col, (3, 2, 2))
        >>> det_batch = A_batch.det()
        >>> print(det_batch.shape)  # torch.Size([3])
        """
        M, N = self.sparse_shape
        
        if M != N:
            raise ValueError(f"Matrix must be square for determinant, got shape ({M}, {N})")
        
        if self.is_batched:
            batch_shape = self.batch_shape
            det_list = []
            
            for idx in self._batch_indices():
                A_single = SparseTensor(
                    self.values[idx], self.row_indices, self.col_indices, (M, N)
                )
                det_list.append(A_single.det())
            
            return torch.stack(det_list).reshape(*batch_shape)
        
        # Use adjoint method for gradient support
        return DetAdjoint.apply(
            self.values, 
            self.row_indices, 
            self.col_indices, 
            (M, N),
            self.device,
            self.is_cuda
        )
    
    # =========================================================================
    # LU Factorization
    # =========================================================================
    
    def lu(self) -> "LUFactorization":
        """
        Compute LU decomposition for repeated solves.
        
        Returns
        -------
        LUFactorization
            Factorization object with solve() method.
        
        Examples
        --------
        >>> A = SparseTensor(val, row, col, (10, 10))
        >>> lu = A.lu()
        >>> x1 = lu.solve(b1)
        >>> x2 = lu.solve(b2)  # Reuses factorization
        """
        if self.is_batched:
            raise NotImplementedError("lu() not supported for batched tensors")
        
        if self.is_cuda:
            raise NotImplementedError("LU decomposition on CUDA not yet supported")
        
        if not is_scipy_available():
            raise RuntimeError("SciPy is required for LU decomposition")
        
        M, N = self.sparse_shape
        lu = scipy_lu(self.values, self.row_indices, self.col_indices, (M, N))
        return LUFactorization(lu, (M, N), self.dtype, self.device)
    
    # =========================================================================
    # String Representation
    # =========================================================================
    
    def __repr__(self) -> str:
        parts = [f"SparseTensor(shape={self._shape}"]
        if self.is_batched:
            parts.append(f"batch={self.batch_shape}")
        parts.append(f"sparse={self.sparse_shape}")
        if self.is_block:
            parts.append(f"block={self.block_shape}")
        parts.append(f"nnz={self.nnz}")
        parts.append(f"dtype={self.dtype}")
        parts.append(f"device={self.device}")
        return ", ".join(parts) + ")"
    
    # =========================================================================
    # Reduction Operations (sum, mean, prod)
    # =========================================================================
    
    def _normalize_axis(self, axis: Optional[Union[int, Tuple[int, ...]]]) -> Optional[Tuple[int, ...]]:
        """Normalize axis to tuple of positive indices."""
        if axis is None:
            return None
        if isinstance(axis, int):
            axis = (axis,)
        ndim = len(self._shape)
        return tuple(a if a >= 0 else ndim + a for a in axis)
    
    def _get_dim_type(self, dim: int) -> str:
        """Get the type of dimension: 'batch', 'sparse_m', 'sparse_n', or 'block'."""
        dim_m, dim_n = self._sparse_dim
        min_sparse = min(dim_m, dim_n)
        max_sparse = max(dim_m, dim_n)
        
        if dim < min_sparse:
            return 'batch'
        elif dim == dim_m:
            return 'sparse_m'
        elif dim == dim_n:
            return 'sparse_n'
        else:
            return 'block'
    
    def _values_axis_for_dim(self, dim: int) -> int:
        """
        Map tensor dimension to values tensor dimension.
        
        Values shape: [...batch, nnz, ...block]
        Tensor shape: [...batch, M, N, ...block]
        """
        dim_m, dim_n = self._sparse_dim
        min_sparse = min(dim_m, dim_n)
        max_sparse = max(dim_m, dim_n)
        
        if dim < min_sparse:
            # Batch dimension - same position
            return dim
        elif dim == dim_m or dim == dim_n:
            # Sparse dimension - maps to nnz axis
            return min_sparse  # nnz is at the position of first sparse dim
        else:
            # Block dimension - after nnz axis
            # Shift by -1 because we replaced (M, N) with (nnz,)
            return dim - 1
    
    def sum(
        self, 
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdim: bool = False
    ) -> Union[torch.Tensor, "SparseTensor"]:
        """
        Sum of sparse tensor elements over specified axis.
        
        Parameters
        ----------
        axis : int, tuple of ints, or None
            Axis or axes along which to sum. Axes correspond to:
            - Batch dimensions: [...batch] at the beginning
            - Sparse dimensions: (M, N) at sparse_dim positions
            - Block dimensions: [...block] at the end
            
            If None, sum over all elements (returns scalar tensor).
        keepdim : bool
            Whether to keep the reduced dimensions.
            
        Returns
        -------
        torch.Tensor or SparseTensor
            - If reducing over sparse dimensions: returns dense tensor
            - If reducing over batch/block dimensions only: returns SparseTensor
            - If axis=None: returns scalar tensor
        
        Examples
        --------
        >>> # Shape: [batch=2, M=10, N=10, block=3]
        >>> A = SparseTensor(val, row, col, (2, 10, 10, 3))
        >>> 
        >>> A.sum()           # Scalar: sum all elements
        >>> A.sum(axis=0)     # Sum over batch -> [10, 10, 3]
        >>> A.sum(axis=1)     # Sum over M (rows) -> [2, 10, 3] (dense)
        >>> A.sum(axis=2)     # Sum over N (cols) -> [2, 10, 3] (dense)
        >>> A.sum(axis=3)     # Sum over block -> SparseTensor [2, 10, 10]
        >>> A.sum(axis=(1,2)) # Sum over M and N -> [2, 3] (dense)
        """
        if axis is None:
            # Sum over all elements
            return self.values.sum()
        
        axes = self._normalize_axis(axis)
        dim_types = [self._get_dim_type(d) for d in axes]
        
        # Check if we're reducing over sparse dimensions
        has_sparse_reduction = any(dt in ('sparse_m', 'sparse_n') for dt in dim_types)
        
        if has_sparse_reduction:
            # Need to convert to dense for sparse reduction
            return self._sum_over_sparse(axes, keepdim)
        else:
            # Only batch/block reduction - can stay sparse
            return self._sum_over_batch_block(axes, keepdim)
    
    def _sum_over_sparse(
        self, 
        axes: Tuple[int, ...], 
        keepdim: bool
    ) -> torch.Tensor:
        """Sum that involves sparse dimensions - returns dense."""
        M, N = self.sparse_shape
        dim_m, dim_n = self._sparse_dim
        row, col = self.row_indices, self.col_indices
        
        # Separate sparse and non-sparse axes
        sparse_axes = [a for a in axes if self._get_dim_type(a) in ('sparse_m', 'sparse_n')]
        other_axes = [a for a in axes if self._get_dim_type(a) not in ('sparse_m', 'sparse_n')]
        
        reduce_m = dim_m in axes
        reduce_n = dim_n in axes
        
        if self.is_batched:
            B = self.batch_size
            batch_shape = self.batch_shape
            vals_flat = self.values.reshape(B, self.nnz, *self.block_shape) if self.is_block else self.values.reshape(B, self.nnz)
            
            if reduce_m and reduce_n:
                # Sum all sparse entries per batch
                result = vals_flat.sum(dim=1)  # [B, *block]
                result = result.reshape(*batch_shape, *self.block_shape) if self.is_block else result.reshape(*batch_shape)
            elif reduce_m:
                # Sum over rows -> result is [B, N, *block]
                result = torch.zeros(B, N, *self.block_shape, dtype=self.dtype, device=self.device)
                col_idx = col.unsqueeze(0).expand(B, -1)
                if self.is_block:
                    for i in range(B):
                        result[i].scatter_add_(0, col_idx[i].unsqueeze(-1).expand(-1, *self.block_shape), vals_flat[i])
                else:
                    result.scatter_add_(1, col_idx, vals_flat)
                result = result.reshape(*batch_shape, N, *self.block_shape) if self.is_block else result.reshape(*batch_shape, N)
            else:  # reduce_n
                # Sum over cols -> result is [B, M, *block]
                result = torch.zeros(B, M, *self.block_shape, dtype=self.dtype, device=self.device)
                row_idx = row.unsqueeze(0).expand(B, -1)
                if self.is_block:
                    for i in range(B):
                        result[i].scatter_add_(0, row_idx[i].unsqueeze(-1).expand(-1, *self.block_shape), vals_flat[i])
                else:
                    result.scatter_add_(1, row_idx, vals_flat)
                result = result.reshape(*batch_shape, M, *self.block_shape) if self.is_block else result.reshape(*batch_shape, M)
        else:
            vals = self.values
            
            if reduce_m and reduce_n:
                result = vals.sum(dim=0) if self.is_block else vals.sum()
            elif reduce_m:
                result = torch.zeros(N, *self.block_shape, dtype=self.dtype, device=self.device) if self.is_block else torch.zeros(N, dtype=self.dtype, device=self.device)
                if self.is_block:
                    result.scatter_add_(0, col.unsqueeze(-1).expand(-1, *self.block_shape), vals)
                else:
                    result.scatter_add_(0, col, vals)
            else:  # reduce_n
                result = torch.zeros(M, *self.block_shape, dtype=self.dtype, device=self.device) if self.is_block else torch.zeros(M, dtype=self.dtype, device=self.device)
                if self.is_block:
                    result.scatter_add_(0, row.unsqueeze(-1).expand(-1, *self.block_shape), vals)
                else:
                    result.scatter_add_(0, row, vals)
        
        # Handle other axes reduction
        if other_axes:
            result_axes = [self._values_axis_for_dim(a) for a in other_axes]
            result = result.sum(dim=tuple(result_axes), keepdim=keepdim)
        
        return result
    
    def _sum_over_batch_block(
        self, 
        axes: Tuple[int, ...], 
        keepdim: bool
    ) -> "SparseTensor":
        """Sum over batch/block dimensions only - stays sparse."""
        # Map tensor axes to values axes
        val_axes = tuple(self._values_axis_for_dim(a) for a in axes)
        new_values = self.values.sum(dim=val_axes, keepdim=keepdim)
        
        # Compute new shape
        new_shape = list(self._shape)
        if keepdim:
            for a in axes:
                new_shape[a] = 1
        else:
            for a in sorted(axes, reverse=True):
                del new_shape[a]
        
        # Adjust sparse_dim if needed
        new_sparse_dim = list(self._sparse_dim)
        if not keepdim:
            removed_before_m = sum(1 for a in axes if a < self._sparse_dim[0])
            removed_before_n = sum(1 for a in axes if a < self._sparse_dim[1])
            new_sparse_dim[0] -= removed_before_m
            new_sparse_dim[1] -= removed_before_n
        
        return SparseTensor(
            new_values, self.row_indices, self.col_indices, 
            tuple(new_shape), sparse_dim=tuple(new_sparse_dim)
        )
    
    def mean(
        self, 
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdim: bool = False
    ) -> Union[torch.Tensor, "SparseTensor"]:
        """
        Mean of sparse tensor elements over specified axis.
        
        Note: For sparse dimensions, this computes mean of non-zero values only,
        NOT the mean over all M*N elements. For full mean, use to_dense().mean().
        
        Parameters
        ----------
        axis : int, tuple of ints, or None
            Axis or axes along which to compute mean.
        keepdim : bool
            Whether to keep the reduced dimensions.
            
        Returns
        -------
        torch.Tensor or SparseTensor
            Mean values.
        
        Examples
        --------
        >>> A = SparseTensor(val, row, col, (10, 10))
        >>> A.mean()           # Mean of all non-zero values
        >>> A.mean(axis=0)     # Mean over batch dimension
        """
        if axis is None:
            return self.values.mean()
        
        axes = self._normalize_axis(axis)
        
        # For sparse dims, we compute sum/count of nnz (not M*N)
        sum_result = self.sum(axis=axis, keepdim=keepdim)
        
        # Compute divisor based on axes
        divisor = 1
        for a in axes:
            divisor *= self._shape[a]
        
        # But for sparse dimensions, divisor should be nnz not M*N
        dim_types = [self._get_dim_type(a) for a in axes]
        if 'sparse_m' in dim_types or 'sparse_n' in dim_types:
            # For sparse reduction, we're averaging over nnz values
            sparse_divisor = 1
            if 'sparse_m' in dim_types:
                sparse_divisor *= self.sparse_shape[0]
            if 'sparse_n' in dim_types:
                sparse_divisor *= self.sparse_shape[1]
            # Replace M*N with nnz
            divisor = divisor // sparse_divisor * self.nnz
        
        if isinstance(sum_result, SparseTensor):
            return SparseTensor(
                sum_result.values / divisor,
                sum_result.row_indices,
                sum_result.col_indices,
                sum_result.shape,
                sparse_dim=sum_result.sparse_dim
            )
        return sum_result / divisor
    
    def prod(
        self, 
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdim: bool = False
    ) -> Union[torch.Tensor, "SparseTensor"]:
        """
        Product of sparse tensor elements over specified axis.
        
        Warning: For sparse matrices, zero elements are not included in the product.
        This means prod() computes the product of non-zero values only.
        
        Parameters
        ----------
        axis : int, tuple of ints, or None
            Axis or axes along which to compute product.
        keepdim : bool
            Whether to keep the reduced dimensions.
            
        Returns
        -------
        torch.Tensor or SparseTensor
            Product values.
        
        Examples
        --------
        >>> A = SparseTensor(val, row, col, (10, 10))
        >>> A.prod()           # Product of all non-zero values
        >>> A.prod(axis=0)     # Product over batch dimension
        """
        if axis is None:
            return self.values.prod()
        
        axes = self._normalize_axis(axis)
        dim_types = [self._get_dim_type(d) for d in axes]
        
        # Check if we're reducing over sparse dimensions
        has_sparse_reduction = any(dt in ('sparse_m', 'sparse_n') for dt in dim_types)
        
        if has_sparse_reduction:
            # For sparse reduction, prod is complex - convert to dense
            warnings.warn(
                "prod() over sparse dimensions converts to dense. "
                "This may use significant memory for large matrices."
            )
            dense = self.to_dense()
            return dense.prod(dim=axes, keepdim=keepdim)
        else:
            # Only batch/block reduction
            val_axes = tuple(self._values_axis_for_dim(a) for a in axes)
            new_values = self.values.prod(dim=val_axes, keepdim=keepdim)
            
            new_shape = list(self._shape)
            if keepdim:
                for a in axes:
                    new_shape[a] = 1
            else:
                for a in sorted(axes, reverse=True):
                    del new_shape[a]
            
            new_sparse_dim = list(self._sparse_dim)
            if not keepdim:
                removed_before_m = sum(1 for a in axes if a < self._sparse_dim[0])
                removed_before_n = sum(1 for a in axes if a < self._sparse_dim[1])
                new_sparse_dim[0] -= removed_before_m
                new_sparse_dim[1] -= removed_before_n
            
            return SparseTensor(
                new_values, self.row_indices, self.col_indices,
                tuple(new_shape), sparse_dim=tuple(new_sparse_dim)
            )
    
    def max(
        self, 
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdim: bool = False
    ) -> Union[torch.Tensor, "SparseTensor"]:
        """Max of non-zero values over specified axis."""
        if axis is None:
            return self.values.max()
        
        axes = self._normalize_axis(axis)
        dim_types = [self._get_dim_type(d) for d in axes]
        has_sparse_reduction = any(dt in ('sparse_m', 'sparse_n') for dt in dim_types)
        
        if has_sparse_reduction:
            dense = self.to_dense()
            return dense.max(dim=axes[0], keepdim=keepdim).values if len(axes) == 1 else dense.amax(dim=axes, keepdim=keepdim)
        else:
            val_axes = tuple(self._values_axis_for_dim(a) for a in axes)
            new_values = self.values.amax(dim=val_axes, keepdim=keepdim)
            
            new_shape = list(self._shape)
            if keepdim:
                for a in axes:
                    new_shape[a] = 1
            else:
                for a in sorted(axes, reverse=True):
                    del new_shape[a]
            
            new_sparse_dim = list(self._sparse_dim)
            if not keepdim:
                removed_before_m = sum(1 for a in axes if a < self._sparse_dim[0])
                removed_before_n = sum(1 for a in axes if a < self._sparse_dim[1])
                new_sparse_dim[0] -= removed_before_m
                new_sparse_dim[1] -= removed_before_n
            
            return SparseTensor(
                new_values, self.row_indices, self.col_indices,
                tuple(new_shape), sparse_dim=tuple(new_sparse_dim)
            )
    
    def min(
        self, 
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdim: bool = False
    ) -> Union[torch.Tensor, "SparseTensor"]:
        """Min of non-zero values over specified axis."""
        if axis is None:
            return self.values.min()
        
        axes = self._normalize_axis(axis)
        dim_types = [self._get_dim_type(d) for d in axes]
        has_sparse_reduction = any(dt in ('sparse_m', 'sparse_n') for dt in dim_types)
        
        if has_sparse_reduction:
            dense = self.to_dense()
            return dense.min(dim=axes[0], keepdim=keepdim).values if len(axes) == 1 else dense.amin(dim=axes, keepdim=keepdim)
        else:
            val_axes = tuple(self._values_axis_for_dim(a) for a in axes)
            new_values = self.values.amin(dim=val_axes, keepdim=keepdim)
            
            new_shape = list(self._shape)
            if keepdim:
                for a in axes:
                    new_shape[a] = 1
            else:
                for a in sorted(axes, reverse=True):
                    del new_shape[a]
            
            new_sparse_dim = list(self._sparse_dim)
            if not keepdim:
                removed_before_m = sum(1 for a in axes if a < self._sparse_dim[0])
                removed_before_n = sum(1 for a in axes if a < self._sparse_dim[1])
                new_sparse_dim[0] -= removed_before_m
                new_sparse_dim[1] -= removed_before_n
            
            return SparseTensor(
                new_values, self.row_indices, self.col_indices,
                tuple(new_shape), sparse_dim=tuple(new_sparse_dim)
            )
    
    # =========================================================================
    # Element-wise Operations
    # =========================================================================
    
    def _apply_elementwise(self, func, *args, **kwargs) -> "SparseTensor":
        """Apply element-wise function to values.
        
        Returns the same type as self to support subclassing.
        Subclasses should ensure their __init__ accepts (values, row_indices, col_indices, shape)
        or override this method.
        """
        new_values = func(self.values, *args, **kwargs)
        # Use type(self) to preserve subclass type
        try:
            return type(self)(
                new_values, self.row_indices, self.col_indices, self._shape
            )
        except TypeError:
            # Fallback for subclasses with incompatible __init__
            return SparseTensor(
                new_values, self.row_indices, self.col_indices,
                self._shape, sparse_dim=self._sparse_dim
            )
    
    # Arithmetic operations
    def __add__(self, other: Union[torch.Tensor, "SparseTensor", float, int]) -> "SparseTensor":
        """Element-wise addition. For SparseTensor + SparseTensor, patterns must match."""
        if isinstance(other, SparseTensor):
            if not torch.equal(self.row_indices, other.row_indices) or \
               not torch.equal(self.col_indices, other.col_indices):
                raise ValueError("SparseTensor addition requires matching sparsity patterns")
            return self._apply_elementwise(lambda v: v + other.values)
        return self._apply_elementwise(lambda v: v + other)
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other: Union[torch.Tensor, "SparseTensor", float, int]) -> "SparseTensor":
        if isinstance(other, SparseTensor):
            if not torch.equal(self.row_indices, other.row_indices) or \
               not torch.equal(self.col_indices, other.col_indices):
                raise ValueError("SparseTensor subtraction requires matching sparsity patterns")
            return self._apply_elementwise(lambda v: v - other.values)
        return self._apply_elementwise(lambda v: v - other)
    
    def __rsub__(self, other):
        return self._apply_elementwise(lambda v: other - v)
    
    def __mul__(self, other: Union[torch.Tensor, "SparseTensor", float, int]) -> "SparseTensor":
        """Element-wise multiplication (Hadamard product for sparse tensors)."""
        if isinstance(other, SparseTensor):
            if not torch.equal(self.row_indices, other.row_indices) or \
               not torch.equal(self.col_indices, other.col_indices):
                raise ValueError("SparseTensor multiplication requires matching sparsity patterns")
            return self._apply_elementwise(lambda v: v * other.values)
        return self._apply_elementwise(lambda v: v * other)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other: Union[torch.Tensor, "SparseTensor", float, int]) -> "SparseTensor":
        if isinstance(other, SparseTensor):
            if not torch.equal(self.row_indices, other.row_indices) or \
               not torch.equal(self.col_indices, other.col_indices):
                raise ValueError("SparseTensor division requires matching sparsity patterns")
            return self._apply_elementwise(lambda v: v / other.values)
        return self._apply_elementwise(lambda v: v / other)
    
    def __rtruediv__(self, other):
        return self._apply_elementwise(lambda v: other / v)
    
    def __floordiv__(self, other):
        if isinstance(other, SparseTensor):
            return self._apply_elementwise(lambda v: v // other.values)
        return self._apply_elementwise(lambda v: v // other)
    
    def __pow__(self, exponent: Union[float, int, torch.Tensor]) -> "SparseTensor":
        return self._apply_elementwise(lambda v: v ** exponent)
    
    def __neg__(self) -> "SparseTensor":
        return self._apply_elementwise(lambda v: -v)
    
    def __pos__(self) -> "SparseTensor":
        return self
    
    def __abs__(self) -> "SparseTensor":
        return self._apply_elementwise(torch.abs)
    
    # Math functions - directly delegate to values
    def abs(self) -> "SparseTensor":
        """Element-wise absolute value."""
        return self._apply_elementwise(torch.abs)
    
    def sqrt(self) -> "SparseTensor":
        """Element-wise square root."""
        return self._apply_elementwise(torch.sqrt)
    
    def square(self) -> "SparseTensor":
        """Element-wise square."""
        return self._apply_elementwise(torch.square)
    
    def exp(self) -> "SparseTensor":
        """Element-wise exponential."""
        return self._apply_elementwise(torch.exp)
    
    def log(self) -> "SparseTensor":
        """Element-wise natural logarithm."""
        return self._apply_elementwise(torch.log)
    
    def log10(self) -> "SparseTensor":
        """Element-wise base-10 logarithm."""
        return self._apply_elementwise(torch.log10)
    
    def log2(self) -> "SparseTensor":
        """Element-wise base-2 logarithm."""
        return self._apply_elementwise(torch.log2)
    
    def sin(self) -> "SparseTensor":
        """Element-wise sine."""
        return self._apply_elementwise(torch.sin)
    
    def cos(self) -> "SparseTensor":
        """Element-wise cosine."""
        return self._apply_elementwise(torch.cos)
    
    def tan(self) -> "SparseTensor":
        """Element-wise tangent."""
        return self._apply_elementwise(torch.tan)
    
    def sinh(self) -> "SparseTensor":
        """Element-wise hyperbolic sine."""
        return self._apply_elementwise(torch.sinh)
    
    def cosh(self) -> "SparseTensor":
        """Element-wise hyperbolic cosine."""
        return self._apply_elementwise(torch.cosh)
    
    def tanh(self) -> "SparseTensor":
        """Element-wise hyperbolic tangent."""
        return self._apply_elementwise(torch.tanh)
    
    def sigmoid(self) -> "SparseTensor":
        """Element-wise sigmoid."""
        return self._apply_elementwise(torch.sigmoid)
    
    def relu(self) -> "SparseTensor":
        """Element-wise ReLU."""
        return self._apply_elementwise(torch.relu)
    
    def clamp(self, min: Optional[float] = None, max: Optional[float] = None) -> "SparseTensor":
        """Element-wise clamp."""
        return self._apply_elementwise(lambda v: torch.clamp(v, min=min, max=max))
    
    def sign(self) -> "SparseTensor":
        """Element-wise sign."""
        return self._apply_elementwise(torch.sign)
    
    def floor(self) -> "SparseTensor":
        """Element-wise floor."""
        return self._apply_elementwise(torch.floor)
    
    def ceil(self) -> "SparseTensor":
        """Element-wise ceil."""
        return self._apply_elementwise(torch.ceil)
    
    def round(self) -> "SparseTensor":
        """Element-wise round."""
        return self._apply_elementwise(torch.round)
    
    def reciprocal(self) -> "SparseTensor":
        """Element-wise reciprocal (1/x)."""
        return self._apply_elementwise(torch.reciprocal)
    
    def pow(self, exponent: Union[float, int, torch.Tensor]) -> "SparseTensor":
        """Element-wise power."""
        return self._apply_elementwise(lambda v: torch.pow(v, exponent))
    
    # Comparison operations (return SparseTensor with bool values)
    def __eq__(self, other) -> "SparseTensor":
        if isinstance(other, SparseTensor):
            return self._apply_elementwise(lambda v: v == other.values)
        return self._apply_elementwise(lambda v: v == other)
    
    def __ne__(self, other) -> "SparseTensor":
        if isinstance(other, SparseTensor):
            return self._apply_elementwise(lambda v: v != other.values)
        return self._apply_elementwise(lambda v: v != other)
    
    def __lt__(self, other) -> "SparseTensor":
        if isinstance(other, SparseTensor):
            return self._apply_elementwise(lambda v: v < other.values)
        return self._apply_elementwise(lambda v: v < other)
    
    def __le__(self, other) -> "SparseTensor":
        if isinstance(other, SparseTensor):
            return self._apply_elementwise(lambda v: v <= other.values)
        return self._apply_elementwise(lambda v: v <= other)
    
    def __gt__(self, other) -> "SparseTensor":
        if isinstance(other, SparseTensor):
            return self._apply_elementwise(lambda v: v > other.values)
        return self._apply_elementwise(lambda v: v > other)
    
    def __ge__(self, other) -> "SparseTensor":
        if isinstance(other, SparseTensor):
            return self._apply_elementwise(lambda v: v >= other.values)
        return self._apply_elementwise(lambda v: v >= other)
    
    # Boolean operations
    def logical_not(self) -> "SparseTensor":
        """Element-wise logical NOT."""
        return self._apply_elementwise(torch.logical_not)
    
    def logical_and(self, other: "SparseTensor") -> "SparseTensor":
        """Element-wise logical AND."""
        return self._apply_elementwise(lambda v: torch.logical_and(v, other.values))
    
    def logical_or(self, other: "SparseTensor") -> "SparseTensor":
        """Element-wise logical OR."""
        return self._apply_elementwise(lambda v: torch.logical_or(v, other.values))
    
    def logical_xor(self, other: "SparseTensor") -> "SparseTensor":
        """Element-wise logical XOR."""
        return self._apply_elementwise(lambda v: torch.logical_xor(v, other.values))
    
    # Type checking
    def isnan(self) -> "SparseTensor":
        """Element-wise isnan check."""
        return self._apply_elementwise(torch.isnan)
    
    def isinf(self) -> "SparseTensor":
        """Element-wise isinf check."""
        return self._apply_elementwise(torch.isinf)
    
    def isfinite(self) -> "SparseTensor":
        """Element-wise isfinite check."""
        return self._apply_elementwise(torch.isfinite)
    
    # Gradient-related
    def detach(self) -> "SparseTensor":
        """Detach from computation graph. Preserves subclass type."""
        try:
            return type(self)(
                self.values.detach(),
                self.row_indices,
                self.col_indices,
                self._shape
            )
        except TypeError:
            return SparseTensor(
                self.values.detach(),
                self.row_indices,
                self.col_indices,
                self._shape,
                sparse_dim=self._sparse_dim
            )
    
    def requires_grad_(self, requires_grad: bool = True) -> "SparseTensor":
        """Enable/disable gradient tracking."""
        self.values.requires_grad_(requires_grad)
        return self
    
    @property
    def requires_grad(self) -> bool:
        """Whether gradient tracking is enabled."""
        return self.values.requires_grad
    
    @property
    def grad(self) -> Optional[torch.Tensor]:
        """Gradient of values if available."""
        return self.values.grad
    
    def clone(self) -> "SparseTensor":
        """Create a copy of this SparseTensor. Preserves subclass type."""
        try:
            return type(self)(
                self.values.clone(),
                self.row_indices.clone(),
                self.col_indices.clone(),
                self._shape
            )
        except TypeError:
            return SparseTensor(
                self.values.clone(),
                self.row_indices.clone(),
                self.col_indices.clone(),
                self._shape,
                sparse_dim=self._sparse_dim
            )

    def contiguous(self) -> "SparseTensor":
        """Make values contiguous in memory. Preserves subclass type."""
        try:
            return type(self)(
                self.values.contiguous(),
                self.row_indices.contiguous(),
                self.col_indices.contiguous(),
                self._shape
            )
        except TypeError:
            return SparseTensor(
                self.values.contiguous(),
                self.row_indices.contiguous(),
                self.col_indices.contiguous(),
                self._shape,
                sparse_dim=self._sparse_dim
            )
    
    # =========================================================================
    # Persistence (I/O)
    # =========================================================================
    
    def save(
        self,
        path: Union[str, "os.PathLike"],
        metadata: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Save SparseTensor to safetensors format.
        
        Parameters
        ----------
        path : str or PathLike
            Output file path (should end with .safetensors).
        metadata : dict, optional
            Additional metadata to store.
        
        Example
        -------
        >>> A = SparseTensor(val, row, col, (100, 100))
        >>> A.save("matrix.safetensors")
        """
        from .io import save_sparse
        save_sparse(self, path, metadata)
    
    @classmethod
    def load(
        cls,
        path: Union[str, "os.PathLike"],
        device: Union[str, torch.device] = "cpu"
    ) -> "SparseTensor":
        """
        Load SparseTensor from safetensors format.
        
        Parameters
        ----------
        path : str or PathLike
            Input file path.
        device : str or torch.device
            Device to load tensors to.
        
        Returns
        -------
        SparseTensor
            The loaded sparse tensor.
        
        Example
        -------
        >>> A = SparseTensor.load("matrix.safetensors", device="cuda")
        """
        from .io import load_sparse
        return load_sparse(path, device)
    
    def save_distributed(
        self,
        directory: Union[str, "os.PathLike"],
        num_partitions: int,
        partition_method: str = "simple",
        coords: Optional[torch.Tensor] = None,
        verbose: bool = False
    ) -> None:
        """
        Save as partitioned files for distributed loading.
        
        Creates a directory with metadata and per-partition files.
        Each rank can then load only its own partition.
        
        Parameters
        ----------
        directory : str or PathLike
            Output directory path.
        num_partitions : int
            Number of partitions to create.
        partition_method : str
            'simple', 'metis', or 'geometric'.
        coords : torch.Tensor, optional
            Node coordinates for geometric partitioning.
        verbose : bool
            Print progress.
        
        Example
        -------
        >>> A.save_distributed("matrix_dist", num_partitions=4)
        # Each rank loads its partition:
        >>> partition = DSparseMatrix.load("matrix_dist", rank)
        """
        from .io import save_distributed
        save_distributed(self, directory, num_partitions, partition_method, coords, verbose)


# =============================================================================
# LUFactorization Class
# =============================================================================

class LUFactorization:
    """
    LU factorization wrapper for efficient repeated solves.
    
    Created by SparseTensor.lu().
    
    Parameters
    ----------
    lu_factor : scipy.sparse.linalg.SuperLU
        The SciPy LU factorization object.
    shape : Tuple[int, int]
        Matrix shape.
    dtype : torch.dtype
        Data type.
    device : torch.device
        Device.
    
    Examples
    --------
    >>> A = SparseTensor(val, row, col, (10, 10))
    >>> lu = A.lu()
    >>> x1 = lu.solve(b1)  # First solve
    >>> x2 = lu.solve(b2)  # Much faster - reuses factorization
    """
    
    def __init__(self, lu_factor, shape: Tuple[int, int], dtype: torch.dtype, device: torch.device):
        self._lu = lu_factor
        self._shape = shape
        self._dtype = dtype
        self._device = device
    
    def solve(self, b: torch.Tensor) -> torch.Tensor:
        """
        Solve Ax = b using the cached factorization.
        
        Parameters
        ----------
        b : torch.Tensor
            Right-hand side vector.
        
        Returns
        -------
        torch.Tensor
            Solution x.
        """
        import numpy as np
        b_np = b.detach().cpu().numpy()
        x_np = self._lu.solve(b_np)
        return torch.from_numpy(x_np).to(dtype=self._dtype, device=self._device)
    
    def __repr__(self) -> str:
        return f"LUFactorization(shape={self._shape})"


# =============================================================================
# SparseTensorList Class
# =============================================================================

class SparseTensorList:
    """
    A list of SparseTensors with different structures.
    
    Provides a unified interface for batch operations on matrices
    with different sparsity patterns. Unlike batched SparseTensor
    (which requires same structure), SparseTensorList allows
    each matrix to have different shape and sparsity pattern.
        
        Parameters
        ----------
    tensors : List[SparseTensor]
        List of SparseTensor objects.
    
    Attributes
    ----------
    shapes : List[Tuple[int, ...]]
        List of shapes for each tensor.
    device : torch.device
        Device (from first tensor).
    dtype : torch.dtype
        Data type (from first tensor).
    
    Examples
    --------
    >>> # Create matrices with different sizes
    >>> A1 = SparseTensor(val1, row1, col1, (10, 10))
    >>> A2 = SparseTensor(val2, row2, col2, (20, 20))
    >>> A3 = SparseTensor(val3, row3, col3, (30, 30))
    
    >>> # Create list
    >>> matrices = SparseTensorList([A1, A2, A3])
    >>> print(matrices.shapes)  # [(10, 10), (20, 20), (30, 30)]
    
    >>> # Batch solve
    >>> x_list = matrices.solve([b1, b2, b3])
    
    >>> # Check properties for all
    >>> is_sym = matrices.is_symmetric()  # [tensor(True), tensor(True), tensor(True)]
    """
    
    def __init__(self, tensors: List["SparseTensor"]):
        if not tensors:
            raise ValueError("SparseTensorList cannot be empty")
        self._tensors = list(tensors)
    
    @classmethod
    def from_coo_list(
        cls,
        matrices: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[int, ...]]],
    ) -> "SparseTensorList":
        """
        Create from list of COO data tuples.
        
        Parameters
        ----------
        matrices : List[Tuple]
            List of (values, row_indices, col_indices, shape) tuples.
        
        Returns
        -------
        SparseTensorList
            List of SparseTensors.
        
        Examples
        --------
        >>> data = [
        ...     (val1, row1, col1, (10, 10)),
        ...     (val2, row2, col2, (20, 20)),
        ... ]
        >>> matrices = SparseTensorList.from_coo_list(data)
        """
        tensors = [
            SparseTensor(val, row, col, shape)
            for val, row, col, shape in matrices
        ]
        return cls(tensors)
    
    @classmethod
    def from_torch_sparse_list(cls, A_list: List[torch.Tensor]) -> "SparseTensorList":
        """
        Create from list of PyTorch sparse tensors.
        
        Parameters
        ----------
        A_list : List[torch.Tensor]
            List of PyTorch sparse COO tensors.
        
        Returns
        -------
        SparseTensorList
            List of SparseTensors.
        """
        tensors = [SparseTensor.from_torch_sparse(A) for A in A_list]
        return cls(tensors)
    
    @property
    def shapes(self) -> List[Tuple[int, ...]]:
        """List of shapes for each tensor."""
        return [t.shape for t in self._tensors]
    
    @property
    def device(self) -> torch.device:
        """Device of the first tensor."""
        return self._tensors[0].device
    
    @property
    def dtype(self) -> torch.dtype:
        """Data type of the first tensor."""
        return self._tensors[0].dtype
    
    def __len__(self) -> int:
        """Number of tensors in the list."""
        return len(self._tensors)
    
    def __getitem__(self, idx: int) -> "SparseTensor":
        """
        Get tensor by index.
        
        Parameters
        ----------
        idx : int
            Index (supports negative indexing).
        
        Returns
        -------
        SparseTensor
            The tensor at that index.
        """
        if idx < 0:
            idx = len(self._tensors) + idx
        return self._tensors[idx]
    
    def __iter__(self):
        """Iterate over tensors."""
        return iter(self._tensors)
    
    def to(self, device: Union[str, torch.device]) -> "SparseTensorList":
        """
        Move all tensors to device.
        
        Parameters
        ----------
        device : str or torch.device
            Target device.
        
        Returns
        -------
        SparseTensorList
            New list with tensors on target device.
        """
        return SparseTensorList([t.to(device) for t in self._tensors])
    
    def cuda(self) -> "SparseTensorList":
        """Move all tensors to CUDA."""
        return self.to('cuda')
    
    def cpu(self) -> "SparseTensorList":
        """Move all tensors to CPU."""
        return self.to('cpu')
    
    # =========================================================================
    # Arithmetic Operations
    # =========================================================================
    
    def __matmul__(self, x_list: Union[List[torch.Tensor], torch.Tensor]) -> List[torch.Tensor]:
        """
        Batch matrix-vector/matrix multiplication.
        
        Parameters
        ----------
        x_list : List[torch.Tensor] or torch.Tensor
            If List: one vector/matrix per sparse tensor, each with compatible shape.
            If Tensor: broadcasted to all matrices (must have compatible shape for all).
        
        Returns
        -------
        List[torch.Tensor]
            List of results [A1 @ x1, A2 @ x2, ...] or [A1 @ x, A2 @ x, ...]
            
        Examples
        --------
        >>> matrices = SparseTensorList([A1, A2, A3])
        >>> # Per-matrix vectors
        >>> y_list = matrices @ [x1, x2, x3]
        >>> # Broadcast same vector
        >>> y_list = matrices @ x  # x applied to all
        """
        if isinstance(x_list, torch.Tensor):
            # Broadcast same tensor to all
            return [t @ x_list for t in self._tensors]
        
        if len(x_list) != len(self._tensors):
            raise ValueError(f"Expected {len(self._tensors)} vectors, got {len(x_list)}")
        return [t @ x for t, x in zip(self._tensors, x_list)]
    
    def __add__(self, other: Union["SparseTensorList", float, int]) -> "SparseTensorList":
        """
        Element-wise addition.
        
        Parameters
        ----------
        other : SparseTensorList or scalar
            If SparseTensorList: add corresponding matrices (must have same length).
            If scalar: add to all matrices.
            
        Returns
        -------
        SparseTensorList
            Result of addition.
        """
        if isinstance(other, SparseTensorList):
            if len(other) != len(self._tensors):
                raise ValueError(f"Length mismatch: {len(self._tensors)} vs {len(other)}")
            return SparseTensorList([a + b for a, b in zip(self._tensors, other._tensors)])
        # Scalar addition - add to values
        return SparseTensorList([
            SparseTensor(t.values + other, t.row_indices, t.col_indices, t.shape)
            for t in self._tensors
        ])
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other: Union["SparseTensorList", float, int]) -> "SparseTensorList":
        """Element-wise subtraction."""
        if isinstance(other, SparseTensorList):
            if len(other) != len(self._tensors):
                raise ValueError(f"Length mismatch: {len(self._tensors)} vs {len(other)}")
            return SparseTensorList([a - b for a, b in zip(self._tensors, other._tensors)])
        return SparseTensorList([
            SparseTensor(t.values - other, t.row_indices, t.col_indices, t.shape)
            for t in self._tensors
        ])
    
    def __rsub__(self, other):
        return SparseTensorList([
            SparseTensor(other - t.values, t.row_indices, t.col_indices, t.shape)
            for t in self._tensors
        ])
    
    def __mul__(self, other: Union["SparseTensorList", float, int, torch.Tensor]) -> "SparseTensorList":
        """
        Element-wise multiplication.
        
        Parameters
        ----------
        other : SparseTensorList, scalar, or Tensor
            If SparseTensorList: multiply corresponding matrices element-wise.
            If scalar/Tensor: multiply all values.
            
        Returns
        -------
        SparseTensorList
            Result of multiplication.
        """
        if isinstance(other, SparseTensorList):
            if len(other) != len(self._tensors):
                raise ValueError(f"Length mismatch: {len(self._tensors)} vs {len(other)}")
            return SparseTensorList([a * b for a, b in zip(self._tensors, other._tensors)])
        return SparseTensorList([t * other for t in self._tensors])
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other: Union[float, int, torch.Tensor]) -> "SparseTensorList":
        """Element-wise division by scalar."""
        return SparseTensorList([t / other for t in self._tensors])
    
    def __neg__(self) -> "SparseTensorList":
        """Negate all values."""
        return SparseTensorList([-t for t in self._tensors])
    
    def sum(self, axis: Optional[int] = None) -> Union[List[torch.Tensor], torch.Tensor]:
        """
        Sum values in each matrix.
        
        Parameters
        ----------
        axis : int, optional
            If None: sum all values in each matrix, return List[scalar].
            If 0: sum over rows for each matrix.
            If 1: sum over columns for each matrix.
            
        Returns
        -------
        List[torch.Tensor] or torch.Tensor
            If axis is None: List of scalar tensors (one per matrix).
            If axis is 0 or 1: List of 1D tensors.
            
        Examples
        --------
        >>> matrices = SparseTensorList([A1, A2, A3])
        >>> totals = matrices.sum()  # [sum(A1), sum(A2), sum(A3)]
        >>> row_sums = matrices.sum(axis=1)  # [A1.sum(1), A2.sum(1), ...]
        """
        return [t.sum(axis=axis) for t in self._tensors]
    
    def mean(self, axis: Optional[int] = None) -> List[torch.Tensor]:
        """
        Mean of values in each matrix.
        
        Parameters
        ----------
        axis : int, optional
            Same as sum().
            
        Returns
        -------
        List[torch.Tensor]
            List of mean values/vectors.
        """
        return [t.mean(axis=axis) for t in self._tensors]
    
    def max(self) -> List[torch.Tensor]:
        """Maximum value in each matrix."""
        return [t.max() for t in self._tensors]
    
    def min(self) -> List[torch.Tensor]:
        """Minimum value in each matrix."""
        return [t.min() for t in self._tensors]
    
    def abs(self) -> "SparseTensorList":
        """Absolute value of all elements."""
        return SparseTensorList([t.abs() for t in self._tensors])
    
    def clamp(self, min: Optional[float] = None, max: Optional[float] = None) -> "SparseTensorList":
        """Clamp values in all matrices."""
        return SparseTensorList([t.clamp(min=min, max=max) for t in self._tensors])
    
    def pow(self, exponent: float) -> "SparseTensorList":
        """Element-wise power."""
        return SparseTensorList([t.pow(exponent) for t in self._tensors])
    
    def sqrt(self) -> "SparseTensorList":
        """Element-wise square root."""
        return SparseTensorList([t.sqrt() for t in self._tensors])
    
    def exp(self) -> "SparseTensorList":
        """Element-wise exponential."""
        return SparseTensorList([t.exp() for t in self._tensors])
    
    def log(self) -> "SparseTensorList":
        """Element-wise natural logarithm."""
        return SparseTensorList([t.log() for t in self._tensors])
    
    # =========================================================================
    # Linear Algebra
    # =========================================================================
    
    def solve(self, b_list: List[torch.Tensor], **kwargs) -> List[torch.Tensor]:
        """
        Solve linear systems for all matrices.
        
        Parameters
        ----------
        b_list : List[torch.Tensor]
            List of right-hand side vectors, one per matrix.
        **kwargs
            Additional arguments passed to SparseTensor.solve().
        
        Returns
        -------
        List[torch.Tensor]
            List of solutions.
        
        Examples
        --------
        >>> matrices = SparseTensorList([A1, A2, A3])
        >>> x_list = matrices.solve([b1, b2, b3])
        """
        if len(b_list) != len(self._tensors):
            raise ValueError(f"Expected {len(self._tensors)} RHS vectors, got {len(b_list)}")
        return [t.solve(b, **kwargs) for t, b in zip(self._tensors, b_list)]
    
    def is_symmetric(self, **kwargs) -> List[torch.Tensor]:
        """
        Check symmetry for all matrices.
        
        Parameters
        ----------
        **kwargs
            Arguments passed to SparseTensor.is_symmetric().
        
        Returns
        -------
        List[torch.Tensor]
            List of boolean tensors.
        """
        return [t.is_symmetric(**kwargs) for t in self._tensors]
    
    def is_positive_definite(self, **kwargs) -> List[torch.Tensor]:
        """
        Check positive definiteness for all matrices.
        
        Parameters
        ----------
        **kwargs
            Arguments passed to SparseTensor.is_positive_definite().
        
        Returns
        -------
        List[torch.Tensor]
            List of boolean tensors.
        """
        return [t.is_positive_definite(**kwargs) for t in self._tensors]
    
    def norm(self, ord: Literal['fro', 1, 2] = 'fro') -> List[torch.Tensor]:
        """
        Compute norms for all matrices.
        
        Parameters
        ----------
        ord : {'fro', 1, 2}
            Norm type.
        
        Returns
        -------
        List[torch.Tensor]
            List of norm values.
        """
        return [t.norm(ord=ord) for t in self._tensors]
    
    def eigs(self, k: int = 6, **kwargs) -> List[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """
        Compute eigenvalues for all matrices.
        
        Parameters
        ----------
        k : int
            Number of eigenvalues.
        **kwargs
            Additional arguments.
        
        Returns
        -------
        List[Tuple[torch.Tensor, Optional[torch.Tensor]]]
            List of (eigenvalues, eigenvectors) tuples.
        """
        return [t.eigs(k=k, **kwargs) for t in self._tensors]
    
    def eigsh(self, k: int = 6, **kwargs) -> List[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """
        Compute eigenvalues for symmetric matrices.
        
        Parameters
        ----------
        k : int
            Number of eigenvalues.
        **kwargs
            Additional arguments.
        
        Returns
        -------
        List[Tuple[torch.Tensor, Optional[torch.Tensor]]]
            List of (eigenvalues, eigenvectors) tuples.
        """
        return [t.eigsh(k=k, **kwargs) for t in self._tensors]
    
    def svd(self, k: int = 6) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Compute SVD for all matrices.
        
        Parameters
        ----------
        k : int
            Number of singular values.
        
        Returns
        -------
        List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
            List of (U, S, Vt) tuples.
        """
        return [t.svd(k=k) for t in self._tensors]
    
    def condition_number(self, ord: int = 2) -> List[torch.Tensor]:
        """
        Compute condition numbers for all matrices.
        
        Parameters
        ----------
        ord : int
            Norm order.
        
        Returns
        -------
        List[torch.Tensor]
            List of condition numbers.
        """
        return [t.condition_number(ord=ord) for t in self._tensors]
    
    def det(self) -> List[torch.Tensor]:
        """
        Compute determinants for all matrices.
        
        Returns
        -------
        List[torch.Tensor]
            List of determinant values.
            
        Examples
        --------
        >>> matrices = SparseTensorList([A1, A2, A3])
        >>> dets = matrices.det()
        >>> print([d.item() for d in dets])
        """
        return [t.det() for t in self._tensors]
    
    def spy(
        self,
        indices: Optional[List[int]] = None,
        ncols: int = 3,
        figsize: Optional[Tuple[float, float]] = None,
        **kwargs
    ):
        """
        Visualize sparsity patterns for multiple matrices in a grid.
        
        Parameters
        ----------
        indices : List[int], optional
            Which matrices to visualize. Default: all.
        ncols : int, optional
            Number of columns in subplot grid. Default: 3.
        figsize : Tuple[float, float], optional
            Figure size. Auto-computed if None.
        **kwargs
            Additional arguments passed to SparseTensor.spy().
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
            
        Examples
        --------
        >>> matrices = SparseTensorList([A1, A2, A3, A4])
        >>> matrices.spy()  # Visualize all in grid
        >>> matrices.spy(indices=[0, 2])  # Visualize specific ones
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for spy(). Install with: pip install matplotlib")
        
        if indices is None:
            indices = list(range(len(self._tensors)))
        
        n = len(indices)
        nrows = (n + ncols - 1) // ncols
        
        if figsize is None:
            figsize = (4 * ncols, 4 * nrows)
        
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        
        for i, idx in enumerate(indices):
            row, col = i // ncols, i % ncols
            ax = axes[row, col]
            self._tensors[idx].spy(ax=ax, show_colorbar=False, **kwargs)
            M, N = self._tensors[idx].sparse_shape
            ax.set_title(f'[{idx}] {M}×{N}, nnz={self._tensors[idx].nnz:,}', fontsize=10)
        
        # Hide unused subplots
        for i in range(n, nrows * ncols):
            row, col = i // ncols, i % ncols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        return fig
    
    # =========================================================================
    # Conversion Methods
    # =========================================================================
    
    def to_block_diagonal(self) -> "SparseTensor":
        """
        Merge all matrices into a single block-diagonal SparseTensor.
        
        Creates a sparse matrix where each input matrix appears as a block
        on the diagonal: diag(A1, A2, ..., An).
        
        Returns
        -------
        SparseTensor
            Block-diagonal matrix with shape (sum(M_i), sum(N_i)).
            
        Notes
        -----
        The resulting matrix has the structure:
        
        ```
        [A1  0  0  ...]
        [ 0 A2  0  ...]
        [ 0  0 A3  ...]
        [... ... ... ]
        ```
        
        Examples
        --------
        >>> A1 = SparseTensor(val1, row1, col1, (10, 10))
        >>> A2 = SparseTensor(val2, row2, col2, (20, 20))
        >>> stl = SparseTensorList([A1, A2])
        >>> A_block = stl.to_block_diagonal()  # Shape (30, 30)
        """
        if len(self._tensors) == 0:
            raise ValueError("Cannot convert empty SparseTensorList to block diagonal")
        
        if len(self._tensors) == 1:
            return self._tensors[0]
        
        # Compute offsets
        row_offsets = [0]
        col_offsets = [0]
        
        for t in self._tensors:
            M, N = t.sparse_shape
            row_offsets.append(row_offsets[-1] + M)
            col_offsets.append(col_offsets[-1] + N)
        
        total_rows = row_offsets[-1]
        total_cols = col_offsets[-1]
        
        # Concatenate all COO data with offsets
        all_values = []
        all_rows = []
        all_cols = []
        
        for i, t in enumerate(self._tensors):
            all_values.append(t.values)
            all_rows.append(t.row_indices + row_offsets[i])
            all_cols.append(t.col_indices + col_offsets[i])
        
        values = torch.cat(all_values)
        rows = torch.cat(all_rows)
        cols = torch.cat(all_cols)
        
        return SparseTensor(values, rows, cols, (total_rows, total_cols))
    
    @classmethod
    def from_block_diagonal(
        cls,
        sparse: "SparseTensor",
        sizes: List[Tuple[int, int]]
    ) -> "SparseTensorList":
        """
        Split a block-diagonal SparseTensor into a list of matrices.
        
        Parameters
        ----------
        sparse : SparseTensor
            Block-diagonal matrix to split.
        sizes : List[Tuple[int, int]]
            List of (rows, cols) for each block. Must sum to sparse.shape.
            
        Returns
        -------
        SparseTensorList
            List of extracted blocks.
            
        Examples
        --------
        >>> A_block = SparseTensor(val, row, col, (30, 30))
        >>> stl = SparseTensorList.from_block_diagonal(A_block, [(10, 10), (20, 20)])
        >>> print(len(stl))  # 2
        """
        if sparse.is_batched:
            raise NotImplementedError("from_block_diagonal not supported for batched tensors")
        
        # Validate sizes
        total_rows = sum(s[0] for s in sizes)
        total_cols = sum(s[1] for s in sizes)
        
        if (total_rows, total_cols) != sparse.sparse_shape:
            raise ValueError(
                f"Sizes sum to ({total_rows}, {total_cols}) but sparse has shape {sparse.sparse_shape}"
            )
        
        # Compute offsets
        row_offsets = [0]
        col_offsets = [0]
        for m, n in sizes:
            row_offsets.append(row_offsets[-1] + m)
            col_offsets.append(col_offsets[-1] + n)
        
        tensors = []
        row = sparse.row_indices
        col = sparse.col_indices
        val = sparse.values
        
        for i, (m, n) in enumerate(sizes):
            r_start, r_end = row_offsets[i], row_offsets[i + 1]
            c_start, c_end = col_offsets[i], col_offsets[i + 1]
            
            # Find entries in this block
            mask = (row >= r_start) & (row < r_end) & (col >= c_start) & (col < c_end)
            
            block_row = row[mask] - r_start
            block_col = col[mask] - c_start
            block_val = val[mask]
            
            tensors.append(SparseTensor(block_val, block_row, block_col, (m, n)))
        
        return cls(tensors)
    
    @property
    def block_sizes(self) -> List[Tuple[int, int]]:
        """
        Get the (rows, cols) size of each matrix.
        
        Returns
        -------
        List[Tuple[int, int]]
            List of (M, N) tuples.
        """
        return [t.sparse_shape for t in self._tensors]
    
    @property
    def total_nnz(self) -> int:
        """Total number of non-zeros across all matrices."""
        return sum(t.nnz for t in self._tensors)
    
    @property
    def total_shape(self) -> Tuple[int, int]:
        """Shape of the block-diagonal representation."""
        total_rows = sum(t.sparse_shape[0] for t in self._tensors)
        total_cols = sum(t.sparse_shape[1] for t in self._tensors)
        return (total_rows, total_cols)
    
    def partition(
        self,
        num_partitions: int,
        threshold: int = 1000,
        partition_method: str = 'auto',
        device: Optional[Union[str, torch.device]] = None,
        verbose: bool = False
    ) -> "DSparseTensorList":
        """
        Create distributed version for parallel computing.
        
        Parameters
        ----------
        num_partitions : int
            Number of partitions (typically = world_size).
        threshold : int
            Graphs with nodes >= threshold are partitioned across ranks.
            Smaller graphs are assigned whole to individual ranks.
        partition_method : str
            Method for partitioning large graphs: 'metis', 'simple', 'auto'.
        device : torch.device, optional
            Target device.
        verbose : bool
            Print partition info.
            
        Returns
        -------
        DSparseTensorList
            Distributed sparse tensor list.
            
        Notes
        -----
        **Hybrid Strategy:**
        
        - Small graphs (< threshold nodes): Assigned whole to ranks round-robin.
          Zero edge cuts, no halo exchange needed.
        - Large graphs (>= threshold nodes): Partitioned across all ranks.
          Uses halo exchange for boundary nodes.
        
        This is optimal for molecular datasets where most molecules are small
        but some (proteins, polymers) can be very large.
        
        Examples
        --------
        >>> stl = SparseTensorList([A1, A2, A3, ...])
        >>> dstl = stl.partition(num_partitions=4, threshold=1000)
        >>> y_list = dstl @ x_list  # Distributed matmul
        """
        from .distributed import DSparseTensorList
        
        return DSparseTensorList.from_sparse_tensor_list(
            self,
            num_partitions=num_partitions,
            threshold=threshold,
            partition_method=partition_method,
            device=device,
            verbose=verbose
        )
    
    def __repr__(self) -> str:
        return f"SparseTensorList(n={len(self._tensors)}, device={self.device})"
