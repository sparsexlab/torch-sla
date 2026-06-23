"""
Batch Sparse Linear Solve for PyTorch

This module provides batch solving capabilities for sparse linear equations:
1. Same-layout batch solve: All matrices share the same sparsity pattern
2. Different-layout batch solve: Each matrix can have different sparsity pattern

For same-layout batches, we can leverage optimized batch operations.
For different-layout batches, we solve each system independently.

Device support
--------------
- ``cg`` / ``bicgstab`` / ``gmres`` / ``minres`` / ``lsqr`` / ``lsmr``: PyTorch-native
  and device-agnostic, so they run on **CPU, CUDA, and ROCm/HIP** out of the box.
- ``strumpack``: multifrontal sparse direct, also **CPU / CUDA / ROCm** (needs the
  optional ``torch-strumpack`` package).
- ``cudss_*``: NVIDIA cuDSS direct solver, **CUDA only**.
"""

import torch
from torch.autograd.function import Function
from typing import Tuple, List, Optional, Union, Literal
import warnings

from .backends import (
    get_cudss_module,
    is_cudss_available,
    BACKEND_METHODS,
)

# PyTorch-native methods are device-agnostic (CPU / CUDA / ROCm).
_PYTORCH_METHODS = set(BACKEND_METHODS.get(
    'pytorch', ['cg', 'bicgstab', 'gmres', 'minres', 'lsqr', 'lsmr']))


MethodType = Literal[
    # PyTorch-native, device-agnostic (CPU / CUDA / ROCm)
    'cg', 'bicgstab', 'gmres', 'minres', 'lsqr', 'lsmr',
    # multifrontal direct, device-agnostic (CPU / CUDA / ROCm)
    'strumpack',
    # NVIDIA cuDSS direct (CUDA only)
    'cudss', 'cudss_lu', 'cudss_cholesky', 'cudss_ldlt',
]


class BatchSparseLinearSolveSameLayout(Function):
    """
    Batch solve for matrices with the same sparsity pattern.
    
    All matrices share the same (row, col) indices, but have different values.
    This is common in optimization and neural network applications where
    the matrix structure is fixed but values change.
    """

    @staticmethod
    def forward(ctx,
                val_batch: torch.Tensor,  # [batch, nnz]
                row: torch.Tensor,         # [nnz]
                col: torch.Tensor,         # [nnz]
                shape: Tuple[int, int],
                b_batch: torch.Tensor,     # [batch, m]
                method: str,
                atol: float,
                maxiter: int):
        
        batch_size = val_batch.size(0)
        m, n = shape

        # Fast path: CG is *truly* batched (single scatter matvec over the shared
        # pattern, all systems iterate together) -- no Python loop. Device-agnostic.
        if method == 'cg':
            from .backends.pytorch_backend import batched_cg_same_pattern
            u_batch = batched_cg_same_pattern(
                val_batch, row, col, (m, n), b_batch, atol=atol, maxiter=maxiter)
            ctx.save_for_backward(val_batch, row, col, u_batch)
            ctx.A_shape = shape
            ctx.method = method
            ctx.atol = atol
            ctx.maxiter = maxiter
            return u_batch

        # Other methods: solve each system independently (no batched kernel yet).
        results = []
        for i in range(batch_size):
            val = val_batch[i]
            b = b_batch[i]

            if method in _PYTORCH_METHODS:
                # device-agnostic: CPU / CUDA / ROCm
                from .backends.pytorch_backend import pytorch_solve
                x = pytorch_solve(val, row, col, (m, n), b, method=method, atol=atol, maxiter=maxiter)
            elif method == 'strumpack':
                # multifrontal direct, device-agnostic (CPU / CUDA / ROCm)
                from .backends import strumpack_backend as _sp
                crow, ccol, cval = _sp._coo_to_csr(val, row, col, (m, n))
                x = _sp.solve(_sp.factor(crow, ccol, cval, n), b)
            elif method == 'cudss_lu':
                _cudss = get_cudss_module()
                x = _cudss.lu(torch.stack([row, col], 0), val, m, n, b)
            elif method == 'cudss_cholesky':
                _cudss = get_cudss_module()
                x = _cudss.cholesky(torch.stack([row, col], 0), val, m, n, b)
            elif method == 'cudss_ldlt':
                _cudss = get_cudss_module()
                x = _cudss.ldlt(torch.stack([row, col], 0), val, m, n, b)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            results.append(x)
        
        u_batch = torch.stack(results, dim=0)
        
        ctx.save_for_backward(val_batch, row, col, u_batch)
        ctx.A_shape = shape
        ctx.method = method
        ctx.atol = atol
        ctx.maxiter = maxiter
        
        return u_batch

    @staticmethod
    def backward(ctx, gradu_batch):
        val_batch, row, col, u_batch = ctx.saved_tensors
        m, n = ctx.A_shape
        method = ctx.method
        atol = ctx.atol
        maxiter = ctx.maxiter
        
        batch_size = val_batch.size(0)

        # Fast path: batched adjoint. Solve A_i^T gradb_i = gradu_i for all i at
        # once (transpose = swap row/col), then the COO-value gradient is a single
        # batched gather: dL/dval_i = -gradb_i[row] * u_i[col].
        if method == 'cg':
            from .backends.pytorch_backend import batched_cg_same_pattern
            gradb_batch = batched_cg_same_pattern(
                val_batch, col, row, (n, m), gradu_batch, atol=atol, maxiter=maxiter)
            gradval_batch = -gradb_batch[:, row] * u_batch[:, col]
            return gradval_batch, None, None, None, gradb_batch, None, None, None

        gradval_list = []
        gradb_list = []

        for i in range(batch_size):
            val = val_batch[i]
            u = u_batch[i]
            gradu = gradu_batch[i]
            
            # Solve A^T * gradb = gradu
            if method in _PYTORCH_METHODS:
                # transpose = swap (row, col); device-agnostic (CPU / CUDA / ROCm)
                from .backends.pytorch_backend import pytorch_solve
                gradb = pytorch_solve(val, col, row, (n, m), gradu, method=method, atol=atol, maxiter=maxiter)
            elif method == 'strumpack':
                from .backends import strumpack_backend as _sp
                crow, ccol, cval = _sp._coo_to_csr(val, row, col, (m, n))
                gradb = _sp.solve_transpose(_sp.factor(crow, ccol, cval, n), gradu)
            elif method in ['cudss_lu']:
                _cudss = get_cudss_module()
                gradb = _cudss.lu(torch.stack([col, row], 0), val, n, m, gradu)
            elif method == 'cudss_cholesky':
                _cudss = get_cudss_module()
                gradb = _cudss.cholesky(torch.stack([row, col], 0), val, m, n, gradu)
            elif method == 'cudss_ldlt':
                _cudss = get_cudss_module()
                gradb = _cudss.ldlt(torch.stack([row, col], 0), val, m, n, gradu)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            gradval = -gradb[row] * u[col]
            gradval_list.append(gradval)
            gradb_list.append(gradb)
        
        gradval_batch = torch.stack(gradval_list, dim=0)
        gradb_batch = torch.stack(gradb_list, dim=0)
        
        return gradval_batch, None, None, None, gradb_batch, None, None, None


def spsolve_batch_same_layout(
    val_batch: torch.Tensor,
    row: torch.Tensor,
    col: torch.Tensor,
    shape: Tuple[int, int],
    b_batch: torch.Tensor,
    method: MethodType = "bicgstab",
    atol: float = 1e-10,
    maxiter: int = 10000
) -> torch.Tensor:
    """
    Batch solve sparse linear systems with the SAME sparsity pattern.
    
    .. note::
        ``method='cg'`` runs a **truly batched** solver (one scatter matvec over
        the shared pattern, all systems iterate together) -- this is the fast
        path on GPU / ROCm. Other methods (bicgstab/gmres/minres/lsqr/lsmr,
        strumpack, cudss_*) currently solve each system in a Python loop.

    All matrices A_i share the same (row, col) structure but have different values.
    This is efficient when the sparsity pattern is fixed (e.g., FEM with fixed mesh).
    
    Solves: A_i @ x_i = b_i for i = 0, 1, ..., batch_size-1
    
    Parameters
    ----------
    val_batch : torch.Tensor
        [batch_size, nnz] Non-zero values for each matrix
    row : torch.Tensor
        [nnz] Row indices (shared across batch)
    col : torch.Tensor
        [nnz] Column indices (shared across batch)
    shape : Tuple[int, int]
        (m, n) Shape of each sparse matrix
    b_batch : torch.Tensor
        [batch_size, m] Right-hand side vectors
    method : str
        Solver method (same options as spsolve)
    atol : float
        Absolute tolerance for iterative solvers
    maxiter : int
        Maximum iterations for iterative solvers
        
    Returns
    -------
    torch.Tensor
        [batch_size, n] Solution vectors
        
    Example
    -------
    >>> import torch
    >>> from torch_sla import spsolve_batch_same_layout
    >>>
    >>> batch_size = 10
    >>> n = 100
    >>> nnz = 500
    >>> 
    >>> # Same sparsity pattern, different values
    >>> row = torch.randint(0, n, (nnz,))
    >>> col = torch.randint(0, n, (nnz,))
    >>> val_batch = torch.randn(batch_size, nnz, dtype=torch.float64)
    >>> b_batch = torch.randn(batch_size, n, dtype=torch.float64)
    >>>
    >>> x_batch = spsolve_batch_same_layout(val_batch, row, col, (n, n), b_batch)
    """
    
    # Validation
    assert val_batch.dim() == 2, f"val_batch must be 2D [batch, nnz], got {val_batch.dim()}D"
    assert b_batch.dim() == 2, f"b_batch must be 2D [batch, m], got {b_batch.dim()}D"
    assert val_batch.size(0) == b_batch.size(0), "Batch sizes must match"
    assert val_batch.size(1) == row.size(0), "val_batch[1] must equal nnz"
    assert val_batch.size(1) == col.size(0), "val_batch[1] must equal nnz"
    assert b_batch.size(1) == shape[0], "b_batch[1] must equal m"
    
    return BatchSparseLinearSolveSameLayout.apply(
        val_batch, row, col, shape, b_batch, method, atol, maxiter
    )


def spsolve_batch_different_layout(
    matrices: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[int, int]]],
    b_list: List[torch.Tensor],
    method: MethodType = "bicgstab",
    atol: float = 1e-10,
    maxiter: int = 10000
) -> List[torch.Tensor]:
    """
    Batch solve sparse linear systems with DIFFERENT sparsity patterns.
    
    .. deprecated::
        Use SparseTensorList.solve() instead for a more Pythonic interface:
        
        >>> matrices = SparseTensorList([A1, A2, A3])
        >>> x_list = matrices.solve([b1, b2, b3])
    
    Each matrix can have a different structure. This is useful when dealing
    with heterogeneous problems or adaptive mesh refinement.
    
    Parameters
    ----------
    matrices : List[Tuple[val, row, col, shape]]
        List of sparse matrices, each as (values, row_indices, col_indices, shape)
    b_list : List[torch.Tensor]
        List of right-hand side vectors
    method : str
        Solver method (same options as spsolve)
    atol : float
        Absolute tolerance for iterative solvers
    maxiter : int
        Maximum iterations for iterative solvers
        
    Returns
    -------
    List[torch.Tensor]
        List of solution vectors
        
    Example
    -------
    >>> import torch
    >>> from torch_sla import spsolve_batch_different_layout
    >>>
    >>> # Different matrices with different sizes/patterns
    >>> matrices = []
    >>> b_list = []
    >>> for n in [50, 100, 150]:
    ...     nnz = n * 5
    ...     val = torch.randn(nnz, dtype=torch.float64)
    ...     row = torch.randint(0, n, (nnz,))
    ...     col = torch.randint(0, n, (nnz,))
    ...     matrices.append((val, row, col, (n, n)))
    ...     b_list.append(torch.randn(n, dtype=torch.float64))
    >>>
    >>> x_list = spsolve_batch_different_layout(matrices, b_list)
    """
    from .linear_solve import spsolve
    
    assert len(matrices) == len(b_list), "Number of matrices must equal number of RHS vectors"
    
    results = []
    for (val, row, col, shape), b in zip(matrices, b_list):
        x = spsolve(val, row, col, shape, b, method=method, atol=atol, maxiter=maxiter)
        results.append(x)
    
    return results


def spsolve_batch_coo_same_layout(
    A_template: torch.Tensor,
    val_batch: torch.Tensor,
    b_batch: torch.Tensor,
    method: MethodType = "bicgstab",
    **kwargs
) -> torch.Tensor:
    """
    Batch solve using a template sparse COO tensor for the structure.
    
    Parameters
    ----------
    A_template : torch.Tensor
        Sparse COO tensor defining the sparsity pattern
    val_batch : torch.Tensor
        [batch_size, nnz] Values for each matrix
    b_batch : torch.Tensor
        [batch_size, m] Right-hand side vectors
    method : str
        Solver method
    **kwargs
        Additional arguments passed to spsolve_batch_same_layout
        
    Returns
    -------
    torch.Tensor
        [batch_size, n] Solution vectors
    """
    assert A_template.is_sparse, "A_template must be sparse"
    
    indices = A_template._indices()
    row = indices[0]
    col = indices[1]
    shape = tuple(A_template.shape)
    
    return spsolve_batch_same_layout(val_batch, row, col, shape, b_batch, method, **kwargs)


def spsolve_batch_coo_different_layout(
    A_list: List[torch.Tensor],
    b_list: List[torch.Tensor],
    method: MethodType = "bicgstab",
    **kwargs
) -> List[torch.Tensor]:
    """
    Batch solve using sparse COO tensors with different structures.
    
    Parameters
    ----------
    A_list : List[torch.Tensor]
        List of sparse COO tensors
    b_list : List[torch.Tensor]
        List of right-hand side vectors
    method : str
        Solver method
    **kwargs
        Additional arguments passed to spsolve_batch_different_layout
        
    Returns
    -------
    List[torch.Tensor]
        List of solution vectors
    """
    matrices = []
    for A in A_list:
        assert A.is_sparse, "All matrices must be sparse"
        indices = A._indices()
        val = A._values()
        row = indices[0]
        col = indices[1]
        shape = tuple(A.shape)
        matrices.append((val, row, col, shape))
    
    return spsolve_batch_different_layout(matrices, b_list, method, **kwargs)


# Parallel batch solver for better GPU utilization
class ParallelBatchSolver:
    """
    High-performance parallel batch solver.
    
    This class pre-analyzes the sparsity pattern and caches factorization
    information for repeated solves with the same structure.
    
    Example
    -------
    >>> solver = ParallelBatchSolver(row, col, shape, method='cudss_lu')
    >>> 
    >>> # Solve multiple batches efficiently
    >>> for val_batch, b_batch in data_loader:
    ...     x_batch = solver.solve(val_batch, b_batch)
    """
    
    def __init__(
        self,
        row: torch.Tensor,
        col: torch.Tensor,
        shape: Tuple[int, int],
        method: MethodType = "bicgstab",
        device: Optional[str] = None
    ):
        """
        Initialize the parallel batch solver.
        
        Parameters
        ----------
        row : torch.Tensor
            [nnz] Row indices
        col : torch.Tensor
            [nnz] Column indices
        shape : Tuple[int, int]
            (m, n) Matrix shape
        method : str
            Solver method
        device : str, optional
            Device for computation
        """
        self.row = row
        self.col = col
        self.shape = shape
        self.method = method
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move indices to device
        self.row = self.row.to(self.device)
        self.col = self.col.to(self.device)
    
    def solve(
        self,
        val_batch: torch.Tensor,
        b_batch: torch.Tensor,
        atol: float = 1e-10,
        maxiter: int = 10000
    ) -> torch.Tensor:
        """
        Solve batch of linear systems.
        
        Parameters
        ----------
        val_batch : torch.Tensor
            [batch_size, nnz] Matrix values
        b_batch : torch.Tensor
            [batch_size, m] Right-hand side vectors
        atol : float
            Tolerance for iterative solvers
        maxiter : int
            Maximum iterations
            
        Returns
        -------
        torch.Tensor
            [batch_size, n] Solution vectors
        """
        val_batch = val_batch.to(self.device)
        b_batch = b_batch.to(self.device)
        
        return spsolve_batch_same_layout(
            val_batch, self.row, self.col, self.shape, b_batch,
            method=self.method, atol=atol, maxiter=maxiter
        )
    
    def __call__(self, val_batch: torch.Tensor, b_batch: torch.Tensor, **kwargs) -> torch.Tensor:
        """Callable interface for the solver."""
        return self.solve(val_batch, b_batch, **kwargs)

