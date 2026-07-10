"""
SciPy backend for CPU sparse linear algebra operations.

SciPy provides highly optimized sparse solvers that are generally faster
and more stable than custom implementations.

Methods:
- 'lu': LU factorization via SuperLU (default)
- 'umfpack': UMFPACK direct solver (requires scikit-umfpack)
- 'cg': Conjugate Gradient (for SPD matrices)
- 'bicgstab': BiCGStab (for general matrices)
- 'gmres': GMRES (for general matrices)
- 'lgmres': LGMRES (for general matrices)
- 'minres': MINRES (for symmetric matrices)
- 'qmr': QMR (for general matrices)
"""

import torch
import numpy as np
from typing import Tuple, Optional, Literal, Union
import warnings

try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Check for UMFPACK
try:
    from scikits.umfpack import spsolve as umfpack_solve
    UMFPACK_AVAILABLE = True
except ImportError:
    UMFPACK_AVAILABLE = False


def is_scipy_available() -> bool:
    """Check if SciPy is available"""
    return SCIPY_AVAILABLE


def is_umfpack_available() -> bool:
    """Check if UMFPACK is available"""
    return UMFPACK_AVAILABLE


def torch_coo_to_scipy_csr(
    val: torch.Tensor,
    row: torch.Tensor,
    col: torch.Tensor,
    shape: Tuple[int, int]
) -> "sp.csr_matrix":
    """Convert PyTorch COO tensors to SciPy CSR matrix"""
    if not SCIPY_AVAILABLE:
        raise ImportError("SciPy is required for CPU sparse operations")
    
    val_np = val.detach().cpu().numpy()
    row_np = row.detach().cpu().numpy()
    col_np = col.detach().cpu().numpy()
    
    coo = sp.coo_matrix((val_np, (row_np, col_np)), shape=shape)
    return coo.tocsr()


def scipy_solve(
    val: torch.Tensor,
    row: torch.Tensor,
    col: torch.Tensor,
    shape: Tuple[int, int],
    b: torch.Tensor,
    method: str = "lu",
    atol: float = 1e-10,
    maxiter: int = 10000,
    M: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Solve sparse linear system using SciPy.
    
    Parameters
    ----------
    val : torch.Tensor
        Non-zero values
    row, col : torch.Tensor
        Row and column indices
    shape : Tuple[int, int]
        Matrix shape
    b : torch.Tensor
        Right-hand side
    method : str
        Solver method:
        Direct solvers:
        - 'lu': LU factorization via SuperLU (default)
        - 'umfpack': UMFPACK LU factorization
        Iterative solvers:
        - 'cg': Conjugate Gradient (SPD)
        - 'bicgstab': BiCGStab
        - 'gmres': GMRES
        - 'lgmres': LGMRES
        - 'minres': MINRES (symmetric)
        - 'qmr': QMR
    atol : float
        Tolerance for iterative solvers
    maxiter : int
        Maximum iterations
    M : torch.Tensor, optional
        Preconditioner
        
    Returns
    -------
    torch.Tensor
        Solution vector
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("SciPy is required for CPU sparse operations")
    
    # Convert to SciPy
    A = torch_coo_to_scipy_csr(val, row, col, shape)
    b_np = b.detach().cpu().numpy()
    
    # Preconditioner
    M_scipy = None
    if M is not None:
        M_np = M.detach().cpu().numpy()
        M_scipy = spla.LinearOperator(shape, matvec=lambda x: M_np @ x)
    
    # Helper for iterative solvers: they only accept 1D b, so loop over columns
    def _iterative_solve(solver_fn, A, b_np, **kwargs):
        if b_np.ndim == 2:
            cols = []
            for k in range(b_np.shape[1]):
                x_k, info = solver_fn(A, b_np[:, k], **kwargs)
                if info != 0:
                    warnings.warn(f"{solver_fn.__name__} did not converge for column {k} (info={info})")
                cols.append(x_k)
            return np.column_stack(cols)
        else:
            x_np, info = solver_fn(A, b_np, **kwargs)
            if info != 0:
                warnings.warn(f"{solver_fn.__name__} did not converge (info={info})")
            return x_np

    # Direct solvers (spla.spsolve natively supports 2D b)
    if method in ("lu", "superlu"):
        try:
            x_np = np.asarray(spla.spsolve(A, b_np, use_umfpack=False))
        except Exception as e:
            warnings.warn(f"SuperLU solver failed: {e}, falling back to iterative")
            x_np = _iterative_solve(spla.bicgstab, A, b_np, atol=atol, maxiter=maxiter, M=M_scipy)

    elif method == "umfpack":
        if not UMFPACK_AVAILABLE:
            warnings.warn("UMFPACK not available, falling back to SuperLU")
            x_np = np.asarray(spla.spsolve(A, b_np, use_umfpack=False))
        else:
            try:
                x_np = np.asarray(spla.spsolve(A, b_np, use_umfpack=True))
            except Exception as e:
                warnings.warn(f"UMFPACK solver failed: {e}, falling back to SuperLU")
                x_np = np.asarray(spla.spsolve(A, b_np, use_umfpack=False))

    # Iterative solvers (single-RHS API, column-loop for 2D b)
    elif method == "cg":
        x_np = _iterative_solve(spla.cg, A, b_np, atol=atol, maxiter=maxiter, M=M_scipy)

    elif method == "bicgstab":
        x_np = _iterative_solve(spla.bicgstab, A, b_np, atol=atol, maxiter=maxiter, M=M_scipy)

    elif method == "gmres":
        x_np = _iterative_solve(spla.gmres, A, b_np, atol=atol, maxiter=maxiter, M=M_scipy)

    elif method == "lgmres":
        x_np = _iterative_solve(spla.lgmres, A, b_np, atol=atol, maxiter=maxiter, M=M_scipy)

    elif method == "minres":
        # scipy>=1.14 removed the ``tol`` kwarg on minres; it takes ``rtol``.
        x_np = _iterative_solve(spla.minres, A, b_np, rtol=atol, maxiter=maxiter, M=M_scipy)

    elif method == "qmr":
        x_np = _iterative_solve(spla.qmr, A, b_np, atol=atol, maxiter=maxiter, M1=M_scipy)

    # Legacy method names for backward compatibility
    elif method == "spsolve" or method == "auto":
        x_np = np.asarray(spla.spsolve(A, b_np))

    else:
        raise ValueError(f"Unknown method: {method}. Available: lu, umfpack, cg, bicgstab, gmres, lgmres, minres, qmr")

    return torch.from_numpy(np.ascontiguousarray(x_np)).to(dtype=val.dtype, device=val.device)


def scipy_eigs(
    val: torch.Tensor,
    row: torch.Tensor,
    col: torch.Tensor,
    shape: Tuple[int, int],
    k: int = 6,
    which: str = "LM",
    sigma: Optional[float] = None,
    return_eigenvectors: bool = True
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Compute eigenvalues and eigenvectors using SciPy.
    
    Parameters
    ----------
    val, row, col : torch.Tensor
        COO format sparse matrix
    shape : Tuple[int, int]
        Matrix shape
    k : int
        Number of eigenvalues
    which : str
        Which eigenvalues: 'LM', 'SM', 'LA', 'SA', 'BE'
    sigma : float, optional
        Shift for shift-invert mode
    return_eigenvectors : bool
        Whether to return eigenvectors
        
    Returns
    -------
    eigenvalues : torch.Tensor
    eigenvectors : torch.Tensor or None
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("SciPy is required")
    
    A = torch_coo_to_scipy_csr(val, row, col, shape)
    
    if return_eigenvectors:
        eigenvalues, eigenvectors = spla.eigs(A, k=k, which=which, sigma=sigma)
        return (
            torch.from_numpy(eigenvalues.real).to(dtype=val.dtype),
            torch.from_numpy(eigenvectors.real).to(dtype=val.dtype)
        )
    else:
        eigenvalues = spla.eigs(A, k=k, which=which, sigma=sigma, return_eigenvectors=False)
        return torch.from_numpy(eigenvalues.real).to(dtype=val.dtype), None


def scipy_eigsh(
    val: torch.Tensor,
    row: torch.Tensor,
    col: torch.Tensor,
    shape: Tuple[int, int],
    k: int = 6,
    which: str = "LM",
    sigma: Optional[float] = None,
    return_eigenvectors: bool = True
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Compute eigenvalues for symmetric matrices using SciPy.
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("SciPy is required")
    
    A = torch_coo_to_scipy_csr(val, row, col, shape)
    
    if return_eigenvectors:
        eigenvalues, eigenvectors = spla.eigsh(A, k=k, which=which, sigma=sigma)
        return (
            torch.from_numpy(eigenvalues).to(dtype=val.dtype),
            torch.from_numpy(eigenvectors).to(dtype=val.dtype)
        )
    else:
        eigenvalues = spla.eigsh(A, k=k, which=which, sigma=sigma, return_eigenvectors=False)
        return torch.from_numpy(eigenvalues).to(dtype=val.dtype), None


def scipy_svds(
    val: torch.Tensor,
    row: torch.Tensor,
    col: torch.Tensor,
    shape: Tuple[int, int],
    k: int = 6,
    which: str = "LM"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute singular values and vectors using SciPy.
    
    Returns
    -------
    U, S, Vt : torch.Tensor
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("SciPy is required")
    
    A = torch_coo_to_scipy_csr(val, row, col, shape)
    U, S, Vt = spla.svds(A, k=k, which=which)
    
    # SciPy may return arrays with negative strides, need to copy
    return (
        torch.from_numpy(U.copy()).to(dtype=val.dtype),
        torch.from_numpy(S.copy()).to(dtype=val.dtype),
        torch.from_numpy(Vt.copy()).to(dtype=val.dtype)
    )


def scipy_norm(
    val: torch.Tensor,
    row: torch.Tensor,
    col: torch.Tensor,
    shape: Tuple[int, int],
    ord: Literal['fro', 1, -1, 2, -2, float('inf'), float('-inf')] = 'fro'
) -> torch.Tensor:
    """
    Compute matrix norm using SciPy.
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("SciPy is required")
    
    A = torch_coo_to_scipy_csr(val, row, col, shape)
    
    norm_val = spla.norm(A, ord=ord)
    return torch.tensor(norm_val, dtype=val.dtype)


def scipy_lu(
    val: torch.Tensor,
    row: torch.Tensor,
    col: torch.Tensor,
    shape: Tuple[int, int]
):
    """
    Compute LU decomposition using SciPy (SuperLU).
    
    Returns
    -------
    lu : scipy.sparse.linalg.SuperLU
        LU factorization object that can be used for solving
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("SciPy is required")
    
    A = torch_coo_to_scipy_csr(val, row, col, shape)
    return spla.splu(A.tocsc())


def scipy_ilu(
    val: torch.Tensor,
    row: torch.Tensor,
    col: torch.Tensor,
    shape: Tuple[int, int],
    drop_tol: float = 1e-4,
    fill_factor: float = 10.0
):
    """
    Compute incomplete LU decomposition (preconditioner).
    
    Returns
    -------
    ilu : scipy.sparse.linalg.SuperLU
        ILU factorization that can be used as preconditioner
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("SciPy is required")
    
    A = torch_coo_to_scipy_csr(val, row, col, shape)
    return spla.spilu(A.tocsc(), drop_tol=drop_tol, fill_factor=fill_factor)


def scipy_det(
    val: torch.Tensor,
    row: torch.Tensor,
    col: torch.Tensor,
    shape: Tuple[int, int]
) -> torch.Tensor:
    """
    Compute determinant of sparse matrix using LU decomposition.
    
    For a square matrix A, det(A) is computed from the LU factorization:
    det(A) = det(P) * det(L) * det(U)
    where P is the permutation matrix, L is lower triangular (det=1),
    and U is upper triangular (det = product of diagonal elements).
    
    Parameters
    ----------
    val : torch.Tensor
        Non-zero values
    row, col : torch.Tensor
        Row and column indices
    shape : Tuple[int, int]
        Matrix shape (must be square)
        
    Returns
    -------
    torch.Tensor
        Determinant value (scalar tensor)
        
    Raises
    ------
    ValueError
        If matrix is not square
    ImportError
        If SciPy is not available
        
    Notes
    -----
    - For large matrices, the determinant can overflow/underflow
    - Consider using log-determinant for numerical stability
    - This uses LU decomposition via SuperLU
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("SciPy is required for determinant computation")
    
    if shape[0] != shape[1]:
        raise ValueError(f"Matrix must be square for determinant, got shape {shape}")
    
    # Convert to SciPy CSC format (required for splu)
    A = torch_coo_to_scipy_csr(val, row, col, shape)
    A_csc = A.tocsc()
    
    # Compute LU decomposition
    lu = spla.splu(A_csc)
    
    # Determinant from LU factorization
    # det(A) = det(P) * det(L) * det(U)
    # det(L) = 1 (unit diagonal)
    # det(U) = product of diagonal elements
    # det(P) = (-1)^(number of row exchanges)
    
    # Get diagonal of U
    diag_U = lu.U.diagonal()
    det_U = np.prod(diag_U)
    
    # Get sign from permutation
    # The permutation is represented by perm_r (row permutation)
    # Count inversions to get sign
    perm = lu.perm_r
    n = len(perm)
    # Simple way: count how many swaps needed
    sign = 1
    visited = [False] * n
    for i in range(n):
        if not visited[i]:
            j = i
            cycle_length = 0
            while not visited[j]:
                visited[j] = True
                j = perm[j]
                cycle_length += 1
            if cycle_length > 1:
                sign *= (-1) ** (cycle_length - 1)
    
    det_val = sign * det_U
    
    return torch.tensor(det_val, dtype=val.dtype, device=val.device)
