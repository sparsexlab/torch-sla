"""
Sparse Linear Solve module for PyTorch

This module provides differentiable sparse linear equation solvers with multiple backends.

Backends:
---------
- 'scipy': SciPy backend (CPU only) - Direct solvers via LU/UMFPACK
- 'eigen': Eigen backend (CPU only) - Iterative solvers (CG, BiCGStab)
- 'pytorch': PyTorch-native (CPU & CUDA) - Iterative solvers for large-scale problems
- 'cupy': CuPy backend (CUDA only) - Direct and iterative solvers via cupyx.scipy
- 'cudss': NVIDIA cuDSS (CUDA only) - Direct solvers (LU, Cholesky, LDLT)

Methods:
--------
Direct solvers:
- 'lu': LU factorization (scipy, cupy, cudss)
- 'umfpack': UMFPACK direct solver (scipy only)
- 'lu': LU decomposition
- 'cholesky': Cholesky decomposition (SPD matrices)
- 'ldlt': LDLT decomposition (symmetric matrices, cudss)

Iterative solvers:
- 'cg': Conjugate Gradient (SPD matrices)
- 'bicgstab': BiCGStab (general matrices)
- 'gmres': GMRES (general matrices)
- 'minres': MINRES (symmetric matrices, scipy)

Usage:
------
    # Auto-select backend and method based on device and problem size
    x = spsolve(val, row, col, shape, b)

    # Specify backend and method
    x = spsolve(val, row, col, shape, b, backend='scipy', method='lu')
    x = spsolve(val, row, col, shape, b, backend='cudss', method='lu')
    x = spsolve(val, row, col, shape, b, backend='pytorch', method='cg')  # GPU iterative
    x = spsolve(val, row, col, shape, b, backend='cupy', method='lu')     # CuPy GPU direct
"""

import warnings
import torch
from torch.autograd.function import Function
from typing import Tuple, Optional, Union, Literal

from .backends import (
    get_eigen_module,
    get_cudss_module,
    is_scipy_available,
    is_eigen_available,
    is_pytorch_available,
    is_cupy_available,
    is_cudss_available,
    select_backend,
    select_method,
    BACKEND_METHODS,
    CUDA_ITERATIVE_THRESHOLD,
    BackendType,
    MethodType,
)
from .backends.scipy_backend import scipy_solve
from .backends.pytorch_backend import pytorch_solve


# ============================================================================
# Autograd Functions for gradient support
# ============================================================================


class SparseLinearSolveScipySuperLU(Function):
    """SciPy SuperLU solver with gradient support"""

    @staticmethod
    def forward(ctx, val, row, col, shape, b, method, atol, maxiter):
        u = scipy_solve(val, row, col, shape, b, method=method, atol=atol, maxiter=maxiter)
        ctx.save_for_backward(val, row, col, u)
        ctx.shape = shape
        ctx.method = method
        ctx.atol = atol
        ctx.maxiter = maxiter
        return u

    @staticmethod
    def backward(ctx, gradu):
        val, row, col, u = ctx.saved_tensors
        shape = ctx.shape
        method = ctx.method
        atol = ctx.atol
        maxiter = ctx.maxiter

        # Solve A^T * gradb = gradu
        gradb = scipy_solve(val, col, row, (shape[1], shape[0]), gradu,
                           method=method, atol=atol, maxiter=maxiter)
        gradval = -gradb[row] * u[col]
        if gradval.dim() == 2:
            gradval = gradval.sum(-1)
        return gradval, None, None, None, gradb, None, None, None


class SparseLinearSolveEigenCG(Function):
    """Eigen CG solver with gradient support"""

    @staticmethod
    def forward(ctx, val, row, col, shape, b, atol, maxiter):
        _eigen = get_eigen_module()
        indices = torch.stack([row, col], 0)
        if b.dim() == 2:
            cols = [_eigen.cg(indices, val, shape[0], shape[1], b[:, k], atol, maxiter)
                    for k in range(b.shape[1])]
            u = torch.stack(cols, dim=1)
        else:
            u = _eigen.cg(indices, val, shape[0], shape[1], b, atol, maxiter)
        ctx.save_for_backward(val, row, col, u)
        ctx.A_shape = shape
        ctx.atol = atol
        ctx.maxiter = maxiter
        return u

    @staticmethod
    def backward(ctx, gradu):
        _eigen = get_eigen_module()
        val, row, col, u = ctx.saved_tensors
        m, n = ctx.A_shape
        atol = ctx.atol
        maxiter = ctx.maxiter
        indices_T = torch.stack([col, row], 0)
        if gradu.dim() == 2:
            cols = [_eigen.cg(indices_T, val, n, m, gradu[:, k], atol, maxiter)
                    for k in range(gradu.shape[1])]
            gradb = torch.stack(cols, dim=1)
        else:
            gradb = _eigen.cg(indices_T, val, n, m, gradu, atol, maxiter)
        gradval = -gradb[row] * u[col]
        if gradval.dim() == 2:
            gradval = gradval.sum(-1)
        return gradval, None, None, None, gradb, None, None


class SparseLinearSolveEigenBiCGStab(Function):
    """Eigen BiCGStab solver with gradient support"""

    @staticmethod
    def forward(ctx, val, row, col, shape, b, atol, maxiter):
        _eigen = get_eigen_module()
        indices = torch.stack([row, col], 0)
        if b.dim() == 2:
            cols = [_eigen.bicgstab(indices, val, shape[0], shape[1], b[:, k], atol, maxiter)
                    for k in range(b.shape[1])]
            u = torch.stack(cols, dim=1)
        else:
            u = _eigen.bicgstab(indices, val, shape[0], shape[1], b, atol, maxiter)
        ctx.save_for_backward(val, row, col, u)
        ctx.A_shape = shape
        ctx.atol = atol
        ctx.maxiter = maxiter
        return u

    @staticmethod
    def backward(ctx, gradu):
        _eigen = get_eigen_module()
        val, row, col, u = ctx.saved_tensors
        m, n = ctx.A_shape
        atol = ctx.atol
        maxiter = ctx.maxiter
        indices_T = torch.stack([col, row], 0)
        if gradu.dim() == 2:
            cols = [_eigen.bicgstab(indices_T, val, n, m, gradu[:, k], atol, maxiter)
                    for k in range(gradu.shape[1])]
            gradb = torch.stack(cols, dim=1)
        else:
            gradb = _eigen.bicgstab(indices_T, val, n, m, gradu, atol, maxiter)
        gradval = -gradb[row] * u[col]
        if gradval.dim() == 2:
            gradval = gradval.sum(-1)
        return gradval, None, None, None, gradb, None, None


class SparseLinearSolveCuPy(Function):
    """CuPy solver with gradient support (direct and iterative methods)"""

    @staticmethod
    def forward(ctx, val, row, col, shape, b, method, atol, maxiter, tol):
        from .backends.cupy_backend import cupy_solve
        u = cupy_solve(val, row, col, shape, b, method=method, atol=atol, maxiter=maxiter, tol=tol)
        ctx.save_for_backward(val, row, col, u)
        ctx.A_shape = shape
        ctx.method = method
        ctx.atol = atol
        ctx.maxiter = maxiter
        ctx.tol = tol
        return u

    @staticmethod
    def backward(ctx, gradu):
        from .backends.cupy_backend import cupy_solve
        val, row, col, u = ctx.saved_tensors
        m, n = ctx.A_shape
        method = ctx.method
        atol = ctx.atol
        maxiter = ctx.maxiter
        tol = ctx.tol

        # Solve A^T * gradb = gradu (adjoint method)
        gradb = cupy_solve(val, col, row, (n, m), gradu,
                          method=method, atol=atol, maxiter=maxiter, tol=tol)
        gradval = -gradb[row] * u[col]
        if gradval.dim() == 2:
            gradval = gradval.sum(-1)
        return gradval, None, None, None, gradb, None, None, None, None


class SparseLinearSolveCuDSS(Function):
    """cuDSS general solver with gradient support"""

    @staticmethod
    def forward(ctx, val, row, col, shape, b, matrix_type):
        cudss = get_cudss_module()
        indices = torch.stack([row, col], 0)
        u = cudss.solve(indices, val, shape[0], shape[1], b, matrix_type, "default")
        ctx.save_for_backward(val, row, col, u)
        ctx.A_shape = shape
        ctx.matrix_type = matrix_type
        return u

    @staticmethod
    def backward(ctx, gradu):
        cudss = get_cudss_module()
        val, row, col, u = ctx.saved_tensors
        m, n = ctx.A_shape
        matrix_type = ctx.matrix_type

        if matrix_type in ['symmetric', 'spd', 'hpd']:
            indices = torch.stack([row, col], 0)
            gradb = cudss.solve(indices, val, m, n, gradu, matrix_type, "default")
        else:
            indices_T = torch.stack([col, row], 0)
            gradb = cudss.solve(indices_T, val, n, m, gradu, "general", "default")

        gradval = -gradb[row] * u[col]
        if gradval.dim() == 2:
            gradval = gradval.sum(-1)
        return gradval, None, None, None, gradb, None


class SparseLinearSolveCuDSSLU(Function):
    """cuDSS LU solver with gradient support"""

    @staticmethod
    def forward(ctx, val, row, col, shape, b):
        cudss = get_cudss_module()
        indices = torch.stack([row, col], 0)
        u = cudss.lu(indices, val, shape[0], shape[1], b)
        ctx.save_for_backward(val, row, col, u)
        ctx.A_shape = shape
        return u

    @staticmethod
    def backward(ctx, gradu):
        cudss = get_cudss_module()
        val, row, col, u = ctx.saved_tensors
        m, n = ctx.A_shape
        indices_T = torch.stack([col, row], 0)
        gradb = cudss.lu(indices_T, val, n, m, gradu)
        gradval = -gradb[row] * u[col]
        if gradval.dim() == 2:
            gradval = gradval.sum(-1)
        return gradval, None, None, None, gradb


class SparseLinearSolveCuDSSCholesky(Function):
    """cuDSS Cholesky solver with gradient support"""

    @staticmethod
    def forward(ctx, val, row, col, shape, b):
        cudss = get_cudss_module()
        indices = torch.stack([row, col], 0)
        u = cudss.cholesky(indices, val, shape[0], shape[1], b)
        ctx.save_for_backward(val, row, col, u)
        ctx.A_shape = shape
        return u

    @staticmethod
    def backward(ctx, gradu):
        cudss = get_cudss_module()
        val, row, col, u = ctx.saved_tensors
        m, n = ctx.A_shape
        indices = torch.stack([row, col], 0)
        gradb = cudss.cholesky(indices, val, m, n, gradu)
        gradval = -gradb[row] * u[col]
        if gradval.dim() == 2:
            gradval = gradval.sum(-1)
        return gradval, None, None, None, gradb


class SparseLinearSolveCuDSSLDLT(Function):
    """cuDSS LDLT solver with gradient support"""

    @staticmethod
    def forward(ctx, val, row, col, shape, b):
        cudss = get_cudss_module()
        indices = torch.stack([row, col], 0)
        u = cudss.ldlt(indices, val, shape[0], shape[1], b)
        ctx.save_for_backward(val, row, col, u)
        ctx.A_shape = shape
        return u

    @staticmethod
    def backward(ctx, gradu):
        cudss = get_cudss_module()
        val, row, col, u = ctx.saved_tensors
        m, n = ctx.A_shape
        indices = torch.stack([row, col], 0)
        gradb = cudss.ldlt(indices, val, m, n, gradu)
        gradval = -gradb[row] * u[col]
        if gradval.dim() == 2:
            gradval = gradval.sum(-1)
        return gradval, None, None, None, gradb


class SparseLinearSolvePyTorch(Function):
    """PyTorch-native iterative solver with gradient support (works on CPU & CUDA)"""

    @staticmethod
    def forward(ctx, val, row, col, shape, b, method, atol, rtol, maxiter, preconditioner, mixed_precision):
        u = pytorch_solve(val, row, col, shape, b, method=method, atol=atol, rtol=rtol,
                          maxiter=maxiter, preconditioner=preconditioner, mixed_precision=mixed_precision)
        # Convert back to input dtype if mixed precision was used
        if mixed_precision and u.dtype != val.dtype:
            u = u.to(val.dtype)
        ctx.save_for_backward(val, row, col, u)
        ctx.shape = shape
        ctx.method = method
        ctx.atol = atol
        ctx.rtol = rtol
        ctx.maxiter = maxiter
        ctx.preconditioner = preconditioner
        ctx.mixed_precision = mixed_precision
        return u

    @staticmethod
    def backward(ctx, gradu):
        val, row, col, u = ctx.saved_tensors
        shape = ctx.shape
        method = ctx.method
        atol = ctx.atol
        rtol = ctx.rtol
        maxiter = ctx.maxiter
        preconditioner = ctx.preconditioner
        mixed_precision = ctx.mixed_precision

        # Solve A^T * gradb = gradu
        gradb = pytorch_solve(val, col, row, (shape[1], shape[0]), gradu,
                              method=method, atol=atol, rtol=rtol, maxiter=maxiter,
                              preconditioner=preconditioner, mixed_precision=mixed_precision)
        if gradb.dtype != val.dtype:
            gradb = gradb.to(val.dtype)
        gradval = -gradb[row] * u[col]
        if gradval.dim() == 2:
            gradval = gradval.sum(-1)
        return gradval, None, None, None, gradb, None, None, None, None, None, None


# ============================================================================
# Main solve function
# ============================================================================

def spsolve(
    val: torch.Tensor,
    row: torch.Tensor,
    col: torch.Tensor,
    shape: Tuple[int, int],
    b: torch.Tensor,
    backend: BackendType = "auto",
    method: MethodType = "auto",
    atol: float = 1e-10,
    maxiter: int = 10000,
    tol: float = 1e-12,
    matrix_type: str = "general",
    is_symmetric: bool = False,
    is_spd: bool = False,
    preconditioner: str = "jacobi",
    mixed_precision: bool = False,
    verbose: bool = False,
) -> torch.Tensor:
    """
    Solve the Sparse Linear Equation Ax = b with gradient support.

    Supports multiple backends for CPU and CUDA tensors.

    Parameters
    ----------
    val : torch.Tensor
        [nnz] Non-zero values of sparse matrix A in COO format
    row : torch.Tensor
        [nnz] Row indices
    col : torch.Tensor
        [nnz] Column indices
    shape : Tuple[int, int]
        (m, n) Shape of sparse matrix A
    b : torch.Tensor
        [m] or [m, K] Right-hand side vector (or matrix for multiple RHS)
    backend : str, optional
        Backend to use:
        - 'auto': Auto-select based on device and problem size (default)
        - 'scipy': SciPy (CPU only, uses LU/UMFPACK)
        - 'eigen': Eigen C++ (CPU only, iterative)
        - 'pytorch': PyTorch-native (CPU & CUDA, iterative) - best for large problems
        - 'cupy': CuPy (CUDA only, direct and iterative via cupyx.scipy)
        - 'cudss': NVIDIA cuDSS (CUDA only, direct)
    method : str, optional
        Solver method. Available methods depend on backend:
        - 'auto': Auto-select based on matrix properties
        - 'lu': LU factorization (scipy, cupy, cudss)
        - 'umfpack': UMFPACK direct solver (scipy only)
        - 'cholesky', 'ldlt': Direct solvers (cudss)
        - 'cg', 'cgs', 'bicgstab', 'gmres': Iterative solvers
    atol : float, optional
        Absolute tolerance for iterative solvers, by default 1e-10
    maxiter : int, optional
        Maximum iterations for iterative solvers, by default 10000
    tol : float, optional
        Tolerance for direct solvers, by default 1e-12
    matrix_type : str, optional
        Matrix type for cuDSS: 'general', 'symmetric', 'spd', by default "general"
    is_symmetric : bool, optional
        Hint that matrix is symmetric (for auto method selection)
    is_spd : bool, optional
        Hint that matrix is symmetric positive definite

    Returns
    -------
    torch.Tensor
        [n] or [n, K] Solution vector (or matrix for multiple RHS)

    Examples
    --------
    >>> import torch
    >>> from torch_sla import spsolve
    >>>
    >>> # Create a simple SPD matrix
    >>> val = torch.tensor([4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0], dtype=torch.float64)
    >>> row = torch.tensor([0, 0, 1, 1, 1, 2, 2], dtype=torch.int64)
    >>> col = torch.tensor([0, 1, 0, 1, 2, 1, 2], dtype=torch.int64)
    >>> shape = (3, 3)
    >>> b = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    >>>
    >>> # Auto-select backend and method
    >>> x = spsolve(val, row, col, shape, b)
    >>>
    >>> # Specify backend and method
    >>> x = spsolve(val, row, col, shape, b, backend='scipy', method='lu')
    >>>
    >>> # On CUDA
    >>> val_cuda = val.cuda()
    >>> row_cuda = row.cuda()
    >>> col_cuda = col.cuda()
    >>> b_cuda = b.cuda()
    >>> x_cuda = spsolve(val_cuda, row_cuda, col_cuda, shape, b_cuda, backend='cudss', method='lu')
    """
    # Input validation
    assert val.dim() == 1, f"val must be 1D tensor, got {val.dim()}"
    assert row.dim() == 1, f"row must be 1D tensor, got {row.dim()}"
    assert col.dim() == 1, f"col must be 1D tensor, got {col.dim()}"
    assert b.dim() in (1, 2), f"b must be 1D or 2D tensor, got {b.dim()}"
    assert shape[0] > 0, f"shape[0] must be positive, got {shape[0]}"
    assert shape[1] > 0, f"shape[1] must be positive, got {shape[1]}"
    assert val.size(0) == row.size(0), "val and row must have same size"
    assert val.size(0) == col.size(0), "val and col must have same size"
    assert b.size(0) == shape[0], "b and shape[0] must have same size"
    assert val.dtype == b.dtype, "val and b must have same dtype"

    device = val.device
    n = shape[0]  # Problem size (DOF)

    # Parse combined backend_method strings (e.g., "cudss_lu" → backend="cudss", method="lu")
    if backend == "auto" and method != "auto":
        for bk in BACKEND_METHODS:
            prefix = bk + "_"
            if method.startswith(prefix):
                backend = bk
                method = method[len(prefix):]
                break
            if method == bk:
                backend = bk
                method = "auto"
                break

    # Auto-select backend based on device and problem size
    if backend == "auto":
        backend = select_backend(device, n=n)

    # Auto-select method
    if method == "auto":
        method = select_method(backend, is_symmetric=is_symmetric, is_spd=is_spd)

    # Validate backend-method combination
    valid_methods = BACKEND_METHODS.get(backend, [])
    if method not in valid_methods and method != "auto":
        raise ValueError(f"Method '{method}' not supported by backend '{backend}'. "
                        f"Available methods: {valid_methods}")

    if verbose:
        dtype_str = str(val.dtype).replace("torch.", "")
        print(
            f"[torch-sla] solve: n={n}, nnz={val.numel()}, dtype={dtype_str}, "
            f"device={device.type}, symmetric={bool(is_symmetric)}, spd={bool(is_spd)}, "
            f"backend={backend}, method={method}"
        )

    # ========================================================================
    # SciPy backend (CPU)
    # ========================================================================
    if backend == "scipy":
        if val.is_cuda:
            warnings.warn("SciPy backend requires CPU, moving tensors to CPU")
            val = val.cpu()
            row = row.cpu()
            col = col.cpu()
            b = b.cpu()

        if not is_scipy_available():
            raise RuntimeError("SciPy is not available. Install with: pip install scipy")

        return SparseLinearSolveScipySuperLU.apply(
            val, row, col, shape, b, method, atol, maxiter
        )

    # ========================================================================
    # Eigen backend (CPU)
    # ========================================================================
    elif backend == "eigen":
        if val.is_cuda:
            warnings.warn("Eigen backend requires CPU, moving tensors to CPU")
            val = val.cpu()
            row = row.cpu()
            col = col.cpu()
            b = b.cpu()

        if not is_eigen_available():
            raise RuntimeError("Eigen backend is not available. Ensure C++ extension is compiled.")

        if val.dtype != torch.float64:
            warnings.warn("Using float64 is recommended for good precision with iterative solvers")

        if method == "cg":
            return SparseLinearSolveEigenCG.apply(val, row, col, shape, b, atol, maxiter)
        else:  # bicgstab
            return SparseLinearSolveEigenBiCGStab.apply(val, row, col, shape, b, atol, maxiter)

    # ========================================================================
    # CuPy backend (CUDA)
    # ========================================================================
    elif backend == "cupy":
        if not val.is_cuda:
            raise ValueError("CuPy backend requires CUDA tensors")
        if not is_cupy_available():
            raise RuntimeError("CuPy backend is not available. Install with: pip install cupy-cuda12x")

        return SparseLinearSolveCuPy.apply(val, row, col, shape, b, method, atol, maxiter, tol)

    # ========================================================================
    # cuDSS backend (CUDA)
    # ========================================================================
    elif backend == "cudss":
        if not val.is_cuda:
            raise ValueError("cuDSS backend requires CUDA tensors")
        if not is_cudss_available():
            raise RuntimeError("cuDSS backend is not available. Install with: pip install nvmath-python[cu12]")

        if method == "lu":
            return SparseLinearSolveCuDSSLU.apply(val, row, col, shape, b)
        elif method == "cholesky":
            return SparseLinearSolveCuDSSCholesky.apply(val, row, col, shape, b)
        elif method == "ldlt":
            return SparseLinearSolveCuDSSLDLT.apply(val, row, col, shape, b)
        else:
            # Use general solver with matrix_type
            return SparseLinearSolveCuDSS.apply(val, row, col, shape, b, matrix_type)

    # ========================================================================
    # PyTorch backend (CPU & CUDA - iterative)
    # ========================================================================
    elif backend == "pytorch":
        # PyTorch-native iterative solvers work on both CPU and CUDA
        if val.dtype != torch.float64:
            warnings.warn("Using float64 is recommended for good precision with iterative solvers")

        rtol = 1e-10  # Relative tolerance (stricter for better accuracy)
        return SparseLinearSolvePyTorch.apply(val, row, col, shape, b, method, atol, rtol, maxiter, preconditioner, mixed_precision)

    else:
        raise ValueError(f"Unknown backend: {backend}. Available: scipy, eigen, pytorch, cupy, cudss")


def spsolve_coo(A: torch.Tensor, b: torch.Tensor, **kwargs) -> torch.Tensor:
    """Solve Ax = b where A is a sparse COO tensor

    Parameters
    ----------
    A : torch.Tensor
        Sparse COO tensor representing the matrix
    b : torch.Tensor
        Right-hand side vector
    **kwargs
        Additional arguments passed to spsolve()

    Returns
    -------
    torch.Tensor
        Solution vector x
    """
    assert A.is_sparse, "A must be a sparse tensor"
    assert A.layout == torch.sparse_coo, "A must be in COO format"

    indices = A._indices()
    values = A._values()
    shape = tuple(A.shape)

    row = indices[0]
    col = indices[1]

    return spsolve(values, row, col, shape, b, **kwargs)


def spsolve_csr(A: torch.Tensor, b: torch.Tensor, **kwargs) -> torch.Tensor:
    """Solve Ax = b where A is a sparse CSR tensor

    Parameters
    ----------
    A : torch.Tensor
        Sparse CSR tensor representing the matrix
    b : torch.Tensor
        Right-hand side vector
    **kwargs
        Additional arguments passed to spsolve()

    Returns
    -------
    torch.Tensor
        Solution vector x
    """
    # Convert CSR to COO
    A_coo = A.to_sparse_coo()
    return spsolve_coo(A_coo, b, **kwargs)
