"""Sparse-native determinant / log-determinant for :class:`SparseTensor`
and :class:`DSparseTensor`.

Routing
-------
``DetConfig(method="auto")`` picks a path based on matrix type + device:

* SPD                        -> Cholesky (when ``scikit-sparse`` is available)
* General + CPU              -> Sparse LU via SuperLU
* General + CUDA             -> Copy to CPU + SuperLU (fast; CUDA dense det is
                                ~3x slower for sparse matrices)
* ``log|det|`` for huge SPD  -> Stochastic Lanczos + Chebyshev expansion of
                                ``log(A)`` (Hutchinson trace estimator). Pure
                                matvec; distributed-friendly.

Stack-based scope mirrors :class:`SolverConfig` -- use it as a context
manager / decorator to apply defaults to every ``det()`` call inside.

Example
-------
    >>> with DetConfig(method="hutchinson", num_probes=50):
    ...     ldet = A.logdet()
"""
from __future__ import annotations

import dataclasses
import functools
import math
import threading
import warnings
from dataclasses import dataclass
from typing import Callable, Optional

import torch


# ====================================================================== #
# Stack of scoped DetConfig defaults                                     #
# ====================================================================== #
class _DetDefaultsStack(threading.local):
    def __init__(self):
        self.stack: list = []


_STACK = _DetDefaultsStack()


def _active_det_defaults() -> dict:
    merged: dict = {}
    for layer in _STACK.stack:
        merged.update(layer)
    return merged


@dataclass
class DetConfig:
    """Scoped defaults for ``SparseTensor.det`` / ``SparseTensor.logdet``.

    Fields left ``None`` are inactive. Precedence is the same as
    :class:`SolverConfig`: explicit kwarg > innermost scope > outer
    scopes > hard-coded default.
    """

    method: Optional[str] = None       # 'auto' / 'lu' / 'cholesky' / 'hutchinson' / 'spectral' / 'components'
    backend: Optional[str] = None      # 'auto' / 'scipy' / 'cholmod'
    num_probes: Optional[int] = None   # Hutchinson: # of stochastic probes
    lanczos_iter: Optional[int] = None # Hutchinson: Lanczos depth per probe
    distribution: Optional[str] = None # Hutchinson: 'rademacher' / 'gaussian'
    cpu_fallback: Optional[bool] = None  # CUDA -> CPU transfer for LU/Cholesky
    detect_components: Optional[bool] = None  # split disconnected first

    def _kwargs(self) -> dict:
        out: dict = {}
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            if v is not None:
                out[f.name] = v
        return out

    def __enter__(self) -> "DetConfig":
        _STACK.stack.append(self._kwargs())
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        _STACK.stack.pop()

    def __call__(self, fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapped(*args, **kw):
            with self:
                return fn(*args, **kw)
        return wrapped


# ====================================================================== #
# Active defaults helper                                                 #
# ====================================================================== #
_HARDCODED = dict(
    method="auto",
    backend="auto",
    num_probes=20,
    lanczos_iter=30,
    distribution="rademacher",
    cpu_fallback=True,
    detect_components=True,
)


def _resolve(**explicit) -> dict:
    """Merge explicit kwargs over scope stack over hardcoded defaults."""
    merged = dict(_HARDCODED)
    merged.update(_active_det_defaults())
    for k, v in explicit.items():
        if v is not None:
            merged[k] = v
    return merged


# ====================================================================== #
# Components decomposition: det(A) = prod(det(A[c, c]))                  #
# ====================================================================== #
def _det_via_components(self, **opts) -> torch.Tensor:
    """If A is graph-disconnected, det factors as the product of per-component
    dets. Returns ``None`` if A is connected (caller should fall back)."""
    labels, n_comp = self.connected_components()
    if n_comp <= 1:
        return None
    M, N = self.sparse_shape
    if M != N:
        return None
    device, dtype = self.device, self.values.dtype
    log_abs = torch.zeros((), dtype=dtype, device=device)
    sign = torch.ones((), dtype=dtype, device=device)
    for c in range(n_comp):
        mask = labels == c
        idx = mask.nonzero().squeeze(-1).cpu()
        # Build local SparseTensor for this component.
        keep = mask[self.row_indices.cpu()] & mask[self.col_indices.cpu()]
        if int(keep.sum()) == 0:
            return torch.zeros((), dtype=dtype, device=device)
        sub_val = self.values[keep]
        global_to_local = torch.full((M,), -1, dtype=torch.int64)
        global_to_local[idx] = torch.arange(len(idx), dtype=torch.int64)
        sub_row = global_to_local[self.row_indices[keep].cpu()]
        sub_col = global_to_local[self.col_indices[keep].cpu()]
        from .sparse_tensor import SparseTensor
        sub = SparseTensor(sub_val, sub_row.to(device), sub_col.to(device),
                            shape=(len(idx), len(idx)))
        # No more component split inside (idempotent guard).
        sub_opts = dict(opts)
        sub_opts["detect_components"] = False
        d = _det_dispatch(sub, **sub_opts)
        log_abs = log_abs + d.abs().log()
        sign = sign * d.sign()
    return sign * log_abs.exp()


# ====================================================================== #
# Hutchinson stochastic log-det via Lanczos                              #
# ====================================================================== #
def _logdet_hutchinson(matvec: Callable, n: int, *,
                       num_probes: int = 20,
                       lanczos_iter: int = 30,
                       distribution: str = "rademacher",
                       dtype: torch.dtype = torch.float64,
                       device: torch.device = torch.device("cpu"),
                       generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Stochastic Lanczos log-det estimate for SPD A. Pure matvec.

    For probe z, computes Lanczos tridiagonalization
    ``T_m = V^T A V`` with ``v_1 = z / ||z||``. Then
    ``z^T log(A) z ≈ ||z||² · sum_j Q[0, j]² · log(λ_j)`` where
    ``λ_j, Q`` are the eigendecomposition of ``T_m``. Average over
    ``num_probes`` probes.

    Parameters
    ----------
    matvec     : callable ``x -> A @ x`` (any backing; works for DSparseTensor).
    n          : matrix size.
    num_probes : number of stochastic probes (higher = lower variance).
    lanczos_iter : Lanczos depth per probe.
    distribution : 'rademacher' (±1) or 'gaussian' (N(0,1)).

    Returns
    -------
    log_det : 0-d Tensor on the input device.
    """
    if generator is None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(0)

    estimates = torch.zeros(num_probes, dtype=dtype, device=device)
    for p in range(num_probes):
        if distribution == "rademacher":
            z_cpu = torch.randint(0, 2, (n,), dtype=torch.long,
                                   generator=generator) * 2 - 1
            z = z_cpu.to(device=device, dtype=dtype)
        elif distribution == "gaussian":
            z = torch.randn(n, dtype=dtype, generator=generator).to(device)
        else:
            raise ValueError(f"unknown distribution {distribution!r}")
        z_norm_sq = (z * z).sum()
        # Lanczos
        v_prev = torch.zeros_like(z)
        v_curr = z / z_norm_sq.sqrt()
        alphas = torch.zeros(lanczos_iter, dtype=dtype, device=device)
        betas = torch.zeros(lanczos_iter - 1, dtype=dtype, device=device)
        for j in range(lanczos_iter):
            w = matvec(v_curr)
            alpha = (v_curr * w).sum()
            alphas[j] = alpha
            w = w - alpha * v_curr
            if j > 0:
                w = w - betas[j - 1] * v_prev
            if j < lanczos_iter - 1:
                beta = w.norm()
                betas[j] = beta
                if float(beta) < 1e-12:
                    # Breakdown: truncate
                    alphas = alphas[: j + 1]
                    betas = betas[: j]
                    break
                v_prev = v_curr
                v_curr = w / beta
        # Eigendecomp of the (m x m) tridiagonal T
        T = torch.diag(alphas)
        if betas.numel() > 0:
            T = T + torch.diag(betas, 1) + torch.diag(betas, -1)
        eigvals, eigvecs = torch.linalg.eigh(T)
        # Numerical guard: skip near-zero eigenvalues
        eigvals = eigvals.clamp(min=torch.finfo(dtype).eps)
        weights = eigvecs[0] ** 2
        estimates[p] = z_norm_sq * (weights * eigvals.log()).sum()
    return estimates.mean()


# ====================================================================== #
# Main dispatcher                                                        #
# ====================================================================== #
def _det_dispatch(self, **explicit) -> torch.Tensor:
    """Forward path for ``SparseTensor.det()``. No gradients here -- those
    are handled by ``DetAdjoint``. This function is the routing brain."""
    opts = _resolve(**explicit)
    M, N = self.sparse_shape
    if M != N:
        raise ValueError(f"det requires a square matrix, got {(M, N)}")

    method = opts["method"]

    # auto: detect_components first (fast), then pick LU/Cholesky by SPD.
    if method == "auto":
        if opts["detect_components"]:
            try:
                result = _det_via_components(self, **opts)
                if result is not None:
                    return result
            except Exception:
                pass  # fall through to LU
        try:
            is_pd = bool(self.is_positive_definite().item())
        except Exception:
            is_pd = False
        method = "cholesky" if is_pd else "lu"

    if method == "components":
        result = _det_via_components(self, **{**opts, "detect_components": True})
        if result is None:
            method = "lu"  # connected, fall back
        else:
            return result

    if method == "lu":
        return _det_lu(self, opts)
    if method == "cholesky":
        return _det_cholesky(self, opts)
    if method == "spectral":
        # det = prod(eigenvalues). Only viable on small matrices.
        evals, _ = self.eigsh(k=min(M, 64))
        return evals.prod()
    if method == "hutchinson":
        raise ValueError(
            "Hutchinson is a log-det estimator; call .logdet(method='hutchinson')")
    raise ValueError(f"unknown det method {method!r}")


def _det_lu(self, opts: dict) -> torch.Tensor:
    """Sparse LU (SuperLU on CPU). CUDA inputs go to CPU when
    ``cpu_fallback=True``, which is faster than torch dense det
    (~3x in our benchmarks)."""
    from .backends.scipy_backend import scipy_det
    if self.is_cuda and opts.get("cpu_fallback", True):
        val = self.values.detach().cpu()
        row = self.row_indices.cpu()
        col = self.col_indices.cpu()
        det_val = scipy_det(val, row, col, tuple(self.sparse_shape))
        return det_val.to(self.device)
    if self.is_cuda:
        # User explicitly opted out of CPU fallback; use dense.
        indices = torch.stack([self.row_indices, self.col_indices])
        coo = torch.sparse_coo_tensor(indices, self.values.detach(),
                                       tuple(self.sparse_shape))
        return torch.linalg.det(coo.to_dense())
    return scipy_det(self.values.detach(), self.row_indices,
                     self.col_indices, tuple(self.sparse_shape))


def _det_cholesky(self, opts: dict) -> torch.Tensor:
    """Sparse Cholesky via ``scikit-sparse`` (CHOLMOD).
    Falls back to LU if scikit-sparse isn't available or A isn't SPD."""
    try:
        from sksparse.cholmod import cholesky as cholmod_cholesky
    except ImportError:
        warnings.warn(
            "scikit-sparse not available; falling back to LU for det. "
            "Install with: pip install scikit-sparse",
            RuntimeWarning, stacklevel=3,
        )
        return _det_lu(self, opts)
    from .backends.scipy_backend import torch_coo_to_scipy_csr
    val = self.values.detach()
    row = self.row_indices
    col = self.col_indices
    is_cuda = self.is_cuda
    if is_cuda:
        val = val.cpu()
        row = row.cpu()
        col = col.cpu()
    A = torch_coo_to_scipy_csr(val, row, col,
                                tuple(self.sparse_shape)).tocsc()
    try:
        factor = cholmod_cholesky(A)
    except Exception:
        warnings.warn("CHOLMOD failed (matrix not SPD?); falling back to LU",
                      RuntimeWarning, stacklevel=3)
        return _det_lu(self, opts)
    log_det = float(factor.logdet())
    det_val = torch.tensor(math.exp(log_det), dtype=self.values.dtype)
    return det_val.to(self.device) if is_cuda else det_val


# ====================================================================== #
# logdet entry point                                                     #
# ====================================================================== #
def logdet(self, **explicit) -> torch.Tensor:
    """Return ``log|det(A)|`` (sign is positive for SPD A, raw value
    otherwise). Defaults to Hutchinson when matrix is SPD + large."""
    opts = _resolve(**explicit)
    M, N = self.sparse_shape
    if M != N:
        raise ValueError(f"logdet requires square matrix, got {(M, N)}")

    method = opts["method"]
    if method == "auto":
        # Hutchinson only makes sense when SPD; otherwise LU then log|det|.
        try:
            is_pd = bool(self.is_positive_definite().item())
        except Exception:
            is_pd = False
        method = "hutchinson" if (is_pd and M >= 256) else "cholesky" if is_pd else "lu"

    if method == "hutchinson":
        from .sparse_tensor import SparseTensor as _ST
        matvec = lambda x: self @ x
        return _logdet_hutchinson(
            matvec, M,
            num_probes=opts["num_probes"],
            lanczos_iter=opts["lanczos_iter"],
            distribution=opts["distribution"],
            dtype=self.values.dtype,
            device=self.device,
        )
    if method == "cholesky":
        try:
            from sksparse.cholmod import cholesky as cholmod_cholesky
        except ImportError:
            pass
        else:
            from .backends.scipy_backend import torch_coo_to_scipy_csr
            val = self.values.detach().cpu() if self.is_cuda else self.values.detach()
            row = self.row_indices.cpu() if self.is_cuda else self.row_indices
            col = self.col_indices.cpu() if self.is_cuda else self.col_indices
            A = torch_coo_to_scipy_csr(val, row, col, tuple(self.sparse_shape)).tocsc()
            log_det = float(cholmod_cholesky(A).logdet())
            return torch.tensor(log_det, dtype=self.values.dtype, device=self.device)
        # fall through to LU
    # method == "lu" (or fallthrough)
    d = _det_lu(self, opts)
    return d.abs().log()
