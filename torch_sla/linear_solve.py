"""
Sparse Linear Solve module for PyTorch

This module provides differentiable sparse linear equation solvers with multiple backends.

Backends:
---------
- 'scipy': SciPy backend (CPU only) - Direct solvers via LU/UMFPACK
- 'pytorch': PyTorch-native (CPU & CUDA) - Iterative solvers for large-scale problems
- 'cudss': NVIDIA cuDSS (CUDA only) - Direct solvers (LU, Cholesky, LDLT)

Methods:
--------
Direct solvers:
- 'lu': LU factorization (scipy, cudss)
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
"""

import warnings
import torch
from torch.autograd.function import Function
from typing import Tuple, Optional, Union, Literal

from .backends import (
    get_cudss_module,
    is_scipy_available,
    is_pytorch_available,
    is_cudss_available,
    is_pyamg_available,
    is_amgx_available,
    is_strumpack_available,
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

        # Solve the adjoint system A^H * gradb = gradu (conjugate transpose,
        # not just A^T). For real matrices .conj() is a no-op; for complex it
        # makes the Wirtinger gradient correct.
        gradb = scipy_solve(torch.conj_physical(val), col, row, (shape[1], shape[0]), gradu,
                           method=method, atol=atol, maxiter=maxiter)
        gradval = -gradb[row] * torch.conj_physical(u[col])
        if gradval.dim() == 2:
            gradval = gradval.sum(-1)
        return gradval, None, None, None, gradb, None, None, None


class SparseLinearSolveAmgX(Function):
    """NVIDIA AmgX (via torch-amgx) GPU solver with gradient support.

    Forward: build AmgX resources/config/matrix/solver (consulting
    ``SOLVER_CACHE``), run ``solver.solve`` on the right-hand side.
    Backward: build the conjugate-transpose system, solve via the same
    AmgX path, assemble the adjoint gradient.
    """

    @staticmethod
    def forward(ctx, val, row, col, shape, b, tol, maxiter, method,
                preconditioner):
        from .backends.amgx_backend import amgx_solve
        u = amgx_solve(val, row, col, shape, b,
                       tol=tol, maxiter=maxiter, method=method,
                       preconditioner=preconditioner)
        ctx.save_for_backward(val, row, col, u)
        ctx.shape = shape
        ctx.tol = tol
        ctx.maxiter = maxiter
        ctx.method = method
        ctx.preconditioner = preconditioner
        return u

    @staticmethod
    def backward(ctx, gradu):
        from .backends.amgx_backend import amgx_solve
        val, row, col, u = ctx.saved_tensors
        shape = ctx.shape
        gradb = amgx_solve(torch.conj_physical(val), col, row,
                           (shape[1], shape[0]), gradu,
                           tol=ctx.tol, maxiter=ctx.maxiter,
                           method=ctx.method,
                           preconditioner=ctx.preconditioner)
        gradval = -gradb[row] * torch.conj_physical(u[col])
        if gradval.dim() == 2:
            gradval = gradval.sum(-1)
        return gradval, None, None, None, gradb, None, None, None, None


class SparseLinearSolvePyAMG(Function):
    """PyAMG-hybrid standalone solver with gradient support.

    Forward: build a multigrid hierarchy via PyAMG on CPU, run V-cycle
    iterations on whatever device the inputs live on.  Backward: same
    AMG hierarchy on the transpose (real symmetric -> reuse; complex
    or non-symmetric -> rebuild on the conjugate transpose).
    """

    @staticmethod
    def forward(ctx, val, row, col, shape, b, tol, maxiter, method):
        from .backends.pyamg_backend import pyamg_solve
        u = pyamg_solve(val, row, col, shape, b,
                        tol=tol, maxiter=maxiter, method=method)
        ctx.save_for_backward(val, row, col, u)
        ctx.shape = shape
        ctx.tol = tol
        ctx.maxiter = maxiter
        ctx.method = method
        return u

    @staticmethod
    def backward(ctx, gradu):
        from .backends.pyamg_backend import pyamg_solve
        val, row, col, u = ctx.saved_tensors
        shape = ctx.shape
        gradb = pyamg_solve(torch.conj_physical(val), col, row,
                            (shape[1], shape[0]), gradu,
                            tol=ctx.tol, maxiter=ctx.maxiter, method=ctx.method)
        gradval = -gradb[row] * torch.conj_physical(u[col])
        if gradval.dim() == 2:
            gradval = gradval.sum(-1)
        return gradval, None, None, None, gradb, None, None, None


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

        # Solve the adjoint system A^H * gradb = gradu (conjugate transpose);
        # .conj is a no-op for real, correct Wirtinger gradient for complex.
        gradb = pytorch_solve(torch.conj_physical(val), col, row, (shape[1], shape[0]), gradu,
                              method=method, atol=atol, rtol=rtol, maxiter=maxiter,
                              preconditioner=preconditioner, mixed_precision=mixed_precision)
        if gradb.dtype != val.dtype:
            gradb = gradb.to(val.dtype)
        gradval = -gradb[row] * torch.conj_physical(u[col])
        if gradval.dim() == 2:
            gradval = gradval.sum(-1)
        return gradval, None, None, None, gradb, None, None, None, None, None, None


class SparseLinearSolveStrumpack(Function):
    """STRUMPACK sparse direct solver (CPU / CUDA / ROCm) with gradient support.

    Forward factors the (coalesced) matrix once via torch-strumpack's
    multifrontal LU and solves. Backward reuses the cached factorization for
    the transpose solve ``A^T grad_b = grad_u`` and assembles the adjoint
    ``grad_val[k] = -grad_b[row[k]] * u[col[k]]`` on the original COO pattern.

    Real float64 and complex128 (STRUMPACK builds both ``double`` and
    ``complex<double>``); this is torch-sla's portable direct path on AMD ROCm
    where cuDSS (NVIDIA-only) is unavailable.
    """

    @staticmethod
    def forward(ctx, val, row, col, shape, b):
        from .backends import strumpack_backend as _sp
        crow, ccol, cvals = _sp._coo_to_csr(val, row, col, shape)
        fac = _sp.factor(crow, ccol, cvals, shape[0])
        u = _sp.solve(fac, b)
        ctx.save_for_backward(val, row, col, u)
        ctx.shape = shape
        ctx.b_dim = b.dim()
        return u

    @staticmethod
    def backward(ctx, grad_u):
        from .backends import strumpack_backend as _sp
        val, row, col, u = ctx.saved_tensors
        n = ctx.shape[0]
        # Adjoint: solve A^H grad_b = grad_u, then grad_val = -grad_b[i] conj(u[j]).
        # A^H = conj(A)^T -> COO (col, row, conj(val)). ``.conj()`` is a no-op on
        # real (recovers the A^T / -grad_b[row]*u[col] real adjoint), correct for
        # complex. STRUMPACK re-factors A^H (a different matrix when complex).
        crow, ccol, cvals = _sp._coo_to_csr(torch.conj_physical(val), col, row, (n, n))
        fac_h = _sp.factor(crow, ccol, cvals, n)
        grad_b = _sp.solve(fac_h, grad_u)
        if ctx.b_dim == 1:
            grad_val = -(grad_b[row] * torch.conj_physical(u[col]))
        else:  # multiple RHS: sum over the rhs columns
            grad_val = -(grad_b[row, :] * torch.conj_physical(u[col, :])).sum(dim=1)
        return grad_val, None, None, None, grad_b


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
        - 'pytorch': PyTorch-native (CPU & CUDA, iterative) - best for large problems
        - 'cudss': NVIDIA cuDSS (CUDA only, direct)
    method : str, optional
        Solver method. Available methods depend on backend:
        - 'auto': Auto-select based on matrix properties
        - 'lu': LU factorization (scipy, cudss)
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

    # ========================================================================
    # AmgX backend (Linux + Windows + NVIDIA CUDA only)
    # ========================================================================
    elif backend == "amgx":
        if not val.is_cuda:
            raise ValueError("AmgX backend requires CUDA tensors")
        if not is_amgx_available():
            raise RuntimeError(
                "AmgX backend is not available. Install with: "
                "pip install torch-sla[amgx]    "
                "# pulls torch-amgx wheel (Linux/Windows + NVIDIA CUDA)"
            )
        amgx_method = "pbicgstab" if method == "auto" else method
        # ``preconditioner`` propagated from the public solve() API:
        # "none" (Krylov w/o preconditioner), "jacobi"/"jacobi_l1",
        # "block_jacobi", "multicolor_dilu" (etc.), default "amg".
        amgx_pc = preconditioner or "amg"
        if amgx_pc == "jacobi":   # torch-sla's generic name -> AmgX
            amgx_pc = "jacobi_l1"
        return SparseLinearSolveAmgX.apply(val, row, col, shape, b,
                                           atol, maxiter, amgx_method,
                                           amgx_pc)

    # ========================================================================
    # PyAMG-hybrid backend (CPU setup + torch.sparse V-cycle; cross-platform)
    # ========================================================================
    elif backend == "pyamg":
        if not is_pyamg_available():
            raise RuntimeError(
                "PyAMG backend is not available. Install with: pip install pyamg"
            )
        # PyAMG drives both the CPU coarsening setup and the V-cycle as a
        # standalone iterative solver. ``method`` selects the AMG variant
        # (Ruge-Stuben classical = default; smoothed_aggregation as a
        # plug-in for unstructured / vector PDEs).
        amg_method = "ruge_stuben" if method in ("auto", "amg",
                                                 "ruge_stuben") else method
        if amg_method == "sa":
            amg_method = "smoothed_aggregation"
        return SparseLinearSolvePyAMG.apply(val, row, col, shape, b,
                                            atol, maxiter, amg_method)

    # ========================================================================
    # STRUMPACK backend (sparse direct; CPU / CUDA / ROCm via torch-strumpack)
    # ========================================================================
    elif backend == "strumpack":
        if not is_strumpack_available():
            raise RuntimeError(
                "STRUMPACK backend is not available. Install with: "
                "pip install torch-strumpack    "
                "# cpu / cuda / rocm wheel (portable direct solver, incl. AMD)"
            )
        return SparseLinearSolveStrumpack.apply(val, row, col, shape, b)

    else:
        raise ValueError(
            f"Unknown backend: {backend}. "
            f"Available: scipy, pytorch, cudss, pyamg, amgx, strumpack"
        )


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
