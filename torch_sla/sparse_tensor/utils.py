"""Top-level helpers for SparseTensor (memory estimation, method auto-select).

Lifted out of core.py during the file split.
"""
from __future__ import annotations

from typing import Tuple, Optional, Union, Literal

import torch


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

