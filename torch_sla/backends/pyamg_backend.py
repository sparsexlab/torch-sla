"""PyAMG-hybrid backend: CPU-side AMG setup + torch.sparse V-cycle.

PyAMG (https://github.com/pyamg/pyamg) is a pure-Python algebraic
multigrid library with mature classical Ruge-Stuben and smoothed
aggregation coarsening. It runs only on CPU, but its hierarchies (the
``A`` / ``R`` / ``P`` operators at every level) are simple sparse
matrices that we can lift into :class:`torch.sparse_coo_tensor` and move
to any device. The V-cycle itself is then six SpMV ops per level plus
some smoother iterations, which is happy to run on GPU.

This gives us **cross-platform AMG**:

* macOS: hierarchy + V-cycle both on CPU (PyAMG-only path; no CUDA).
* Linux/Windows CUDA: hierarchy on CPU (PyAMG), V-cycle on GPU via
  ``torch.sparse``. Setup overhead amortises across repeated solves
  whenever the sparsity pattern is unchanged.
* Apple Silicon GPU (MPS): once ``torch.sparse`` lands MPS kernels, the
  V-cycle path follows automatically without any change here.

Two entry points are exposed:

* :func:`pyamg_solve` -- standalone iterative AMG solver. One V-cycle
  per outer iteration until convergence.
* :func:`pyamg_preconditioner` -- factory returning a callable usable
  inside any iterative solver (CG, BiCGStab, ...) as ``M^{-1} r``.

A :class:`PyAMGHierarchy` is also exported for power users who want to
hold the hierarchy explicitly (e.g. for solver caching).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

# Lazy: pyamg import deferred so the module import doesn't break on
# environments without pyamg installed; users get a clear error only on
# first use.
_pyamg = None


def _load_pyamg():
    """Lazy import of pyamg with a clear error when it's missing."""
    global _pyamg
    if _pyamg is None:
        try:
            import pyamg  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "torch_sla.backends.pyamg_backend requires the optional "
                "dependency pyamg. Install it with `pip install pyamg`."
            ) from exc
        _pyamg = pyamg
    return _pyamg


def is_pyamg_available() -> bool:
    """Return ``True`` iff ``pyamg`` can be imported."""
    try:
        _load_pyamg()
        return True
    except ImportError:
        return False


# ====================================================================== #
# Hierarchy representation
# ====================================================================== #
@dataclass
class _Level:
    """One level of the AMG hierarchy as torch.sparse operators."""
    A: Tensor                  # n x n CSR
    D_inv: Tensor              # n,  -- 1/diag(A) cached for Jacobi smoother
    R: Optional[Tensor] = None # n_coarse x n  (None on the coarsest level)
    P: Optional[Tensor] = None # n x n_coarse  (None on the coarsest level)
    # Dense LU on the coarsest level so we can solve A_c y = r_c exactly.
    A_dense_inv: Optional[Tensor] = None


def _scipy_csr_to_torch_sparse(A_csr, *, dtype: torch.dtype,
                               device: torch.device) -> Tensor:
    """Convert a scipy CSR matrix to a :class:`torch.sparse_csr_tensor` on
    the requested device. Used per hierarchy level."""
    A_coo = A_csr.tocoo()
    indices = torch.from_numpy(np.stack([A_coo.row, A_coo.col]).astype(np.int64))
    values = torch.from_numpy(np.ascontiguousarray(A_coo.data)).to(dtype)
    sp = torch.sparse_coo_tensor(indices, values, A_coo.shape,
                                 device=device).coalesce()
    return sp.to_sparse_csr()


def _scipy_diag(A_csr) -> np.ndarray:
    """Pull the diagonal of a scipy sparse matrix (1-D numpy array)."""
    return np.asarray(A_csr.diagonal())


@dataclass
class PyAMGHierarchy:
    """Captured PyAMG multigrid hierarchy in torch.sparse form.

    The hierarchy is built once on CPU by PyAMG and then materialised as
    a list of :class:`_Level` objects whose ``A`` / ``R`` / ``P``
    operators are :class:`torch.sparse_csr_tensor` instances on
    :attr:`device`. The coarsest level additionally caches a dense
    inverse so the coarse solve is a single matvec.

    Build via :meth:`from_scipy_csr` or :meth:`from_coo`.
    """
    levels: List[_Level] = field(default_factory=list)
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    dtype: torch.dtype = torch.float64
    method: str = "ruge_stuben"
    num_pre_smooth: int = 1
    num_post_smooth: int = 1
    cycle: Literal["V"] = "V"

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_scipy_csr(cls, A_csr, *,
                       method: Literal["ruge_stuben",
                                       "smoothed_aggregation"] = "ruge_stuben",
                       device: Optional[torch.device] = None,
                       dtype: Optional[torch.dtype] = None,
                       num_pre_smooth: int = 1,
                       num_post_smooth: int = 1,
                       max_levels: int = 10,
                       max_coarse: int = 64,
                       strength: float = 0.25,
                       **kwargs) -> "PyAMGHierarchy":
        """Build a hierarchy from a scipy CSR matrix.

        Parameters
        ----------
        A_csr : scipy.sparse.csr_matrix
            The (real, square) matrix to precondition.
        method : ``"ruge_stuben"`` | ``"smoothed_aggregation"``
            Coarsening algorithm.
        device, dtype : optional
            Device and dtype of the resulting :class:`torch.sparse`
            operators. Default: CPU + float64.
        num_pre_smooth, num_post_smooth : int
            Jacobi smoothing iterations per level.
        max_levels, max_coarse, strength :
            Forwarded to PyAMG (level cap, coarse-grid size cap,
            strength-of-connection threshold).
        """
        pyamg = _load_pyamg()
        if method == "ruge_stuben":
            ml = pyamg.ruge_stuben_solver(
                A_csr, max_levels=max_levels, max_coarse=max_coarse,
                strength=("classical", {"theta": strength}), **kwargs,
            )
        elif method == "smoothed_aggregation":
            ml = pyamg.smoothed_aggregation_solver(
                A_csr, max_levels=max_levels, max_coarse=max_coarse,
                strength=("symmetric", {"theta": strength}), **kwargs,
            )
        else:
            raise ValueError(f"Unknown method {method!r}")

        device = torch.device(device) if device is not None else torch.device("cpu")
        dtype = dtype if dtype is not None else torch.float64

        levels: List[_Level] = []
        for i, lvl in enumerate(ml.levels):
            A_t = _scipy_csr_to_torch_sparse(lvl.A, dtype=dtype, device=device)
            diag = _scipy_diag(lvl.A).astype(np.float64)
            # Guard against zero diagonal entries (shouldn't happen for AMG
            # but defensive: replace zeros with 1 so D_inv stays finite;
            # those rows act as identity in the smoother, which is fine
            # because the matrix at that row is structurally null anyway.)
            diag = np.where(np.abs(diag) > 1e-30, diag, 1.0)
            D_inv = torch.from_numpy(1.0 / diag).to(dtype=dtype, device=device)

            R = (_scipy_csr_to_torch_sparse(lvl.R, dtype=dtype, device=device)
                 if (hasattr(lvl, "R") and lvl.R is not None) else None)
            P = (_scipy_csr_to_torch_sparse(lvl.P, dtype=dtype, device=device)
                 if (hasattr(lvl, "P") and lvl.P is not None) else None)

            # Coarsest level: cache the dense matrix for a direct solve on the
            # fly. We do NOT pre-invert -- ``torch.linalg.inv`` fails on
            # rank-deficient coarse operators (which smoothed-aggregation
            # can produce when the near-null-space is non-trivial), whereas
            # ``torch.linalg.lstsq`` in ``_v_cycle`` handles both cases.
            A_dense_inv = None
            if R is None and P is None:
                A_dense_inv = torch.from_numpy(lvl.A.toarray()).to(
                    dtype=dtype, device=device
                )

            levels.append(_Level(A=A_t, D_inv=D_inv, R=R, P=P,
                                 A_dense_inv=A_dense_inv))

        return cls(levels=levels, device=device, dtype=dtype, method=method,
                   num_pre_smooth=num_pre_smooth,
                   num_post_smooth=num_post_smooth)

    @classmethod
    def from_coo(cls, val: Tensor, row: Tensor, col: Tensor,
                 shape: Tuple[int, int], **kwargs) -> "PyAMGHierarchy":
        """Build a hierarchy from a torch COO triple. Convenience wrapper
        that bounces through scipy CSR for PyAMG's setup phase."""
        import scipy.sparse as sp
        kwargs.setdefault("device", val.device)
        kwargs.setdefault("dtype", val.dtype)
        A_csr = sp.coo_matrix(
            (val.detach().cpu().numpy(),
             (row.detach().cpu().numpy(), col.detach().cpu().numpy())),
            shape=shape,
        ).tocsr()
        return cls.from_scipy_csr(A_csr, **kwargs)

    # ------------------------------------------------------------------ #
    # V-cycle
    # ------------------------------------------------------------------ #
    def _smoother(self, A: Tensor, D_inv: Tensor, x: Tensor, b: Tensor,
                  num_iters: int) -> Tensor:
        """Weighted Jacobi smoother. ``num_iters`` sweeps of
        ``x <- x + omega * D_inv * (b - A x)`` with the canonical
        ``omega = 2/3`` for classical AMG."""
        omega = 2.0 / 3.0
        for _ in range(num_iters):
            r = b - torch.sparse.mm(A, x.unsqueeze(1)).squeeze(1)
            x = x + omega * D_inv * r
        return x

    def _v_cycle(self, b: Tensor, level: int = 0) -> Tensor:
        L = self.levels[level]
        if L.A_dense_inv is not None:
            # Coarsest level: direct dense solve. ``lstsq`` over ``solve``
            # so we degrade gracefully on rank-deficient coarse operators
            # (which smoothed_aggregation may produce on small problems).
            return torch.linalg.lstsq(L.A_dense_inv, b.unsqueeze(1)).solution.squeeze(1)

        x = torch.zeros_like(b)
        x = self._smoother(L.A, L.D_inv, x, b, self.num_pre_smooth)

        # Restrict the residual to the next coarser level.
        r = b - torch.sparse.mm(L.A, x.unsqueeze(1)).squeeze(1)
        rc = torch.sparse.mm(L.R, r.unsqueeze(1)).squeeze(1)

        ec = self._v_cycle(rc, level + 1)

        # Prolong correction and update.
        x = x + torch.sparse.mm(L.P, ec.unsqueeze(1)).squeeze(1)
        x = self._smoother(L.A, L.D_inv, x, b, self.num_post_smooth)
        return x

    def v_cycle(self, b: Tensor) -> Tensor:
        """Execute one V-cycle on the residual vector ``b``."""
        if b.dim() != 1:
            raise ValueError(
                f"PyAMGHierarchy.v_cycle expects a 1-D vector, got "
                f"shape {tuple(b.shape)}"
            )
        return self._v_cycle(b.to(device=self.device, dtype=self.dtype))

    # Make instances usable as a preconditioner callable.
    def __call__(self, r: Tensor) -> Tensor:
        return self.v_cycle(r)


# ====================================================================== #
# Standalone AMG solver
# ====================================================================== #
def pyamg_solve(val: Tensor, row: Tensor, col: Tensor,
                shape: Tuple[int, int], b: Tensor,
                *,
                tol: float = 1e-8,
                maxiter: int = 100,
                method: Literal["ruge_stuben",
                                "smoothed_aggregation"] = "ruge_stuben",
                hierarchy: Optional[PyAMGHierarchy] = None,
                return_info: bool = False,
                **kwargs):
    """Solve ``A x = b`` using AMG as a standalone iterative solver.

    One V-cycle per outer iteration; converges in <10 iterations for
    most well-conditioned PDE problems.

    Parameters
    ----------
    val, row, col, shape : torch.Tensor + tuple
        COO representation of the matrix.
    b : torch.Tensor
        Right-hand side, 1-D.
    tol : float
        Stop when ``||r||_2 / ||b||_2 < tol``.
    maxiter : int
        Cap on outer V-cycle iterations.
    method : ``"ruge_stuben"`` | ``"smoothed_aggregation"``
        PyAMG coarsening method.
    hierarchy : :class:`PyAMGHierarchy`, optional
        Reuse a pre-built hierarchy (skips PyAMG setup); the calling
        code is responsible for ensuring the sparsity pattern matches.
    return_info : bool
        If ``True``, return ``(x, info_dict)`` instead of just ``x``.
    """
    if hierarchy is None:
        hierarchy = PyAMGHierarchy.from_coo(
            val, row, col, shape, method=method, **kwargs
        )

    # Build a torch sparse handle for the finest-level A used in residual
    # evaluations + matvec. (Same as ``hierarchy.levels[0].A``.)
    A0 = hierarchy.levels[0].A
    x = torch.zeros_like(b)
    b_norm = float(b.norm().item()) or 1.0

    converged = False
    final_residual = float("nan")
    iter_count = 0
    for it in range(maxiter):
        r = b - torch.sparse.mm(A0, x.unsqueeze(1)).squeeze(1)
        rn = float(r.norm().item())
        if rn / b_norm < tol:
            converged = True
            final_residual = rn
            iter_count = it
            break
        e = hierarchy.v_cycle(r)
        x = x + e
        final_residual = rn
        iter_count = it + 1
    else:
        # Last residual evaluation -- did we converge on the final step?
        r = b - torch.sparse.mm(A0, x.unsqueeze(1)).squeeze(1)
        final_residual = float(r.norm().item())
        converged = (final_residual / b_norm < tol)

    if return_info:
        info = {
            "iter_count": iter_count,
            "residual": final_residual,
            "converged": converged,
            "method": f"amg-{method}",
            "backend": "pyamg",
        }
        return x, info
    return x


def pyamg_preconditioner(val: Tensor, row: Tensor, col: Tensor,
                         shape: Tuple[int, int],
                         **kwargs) -> PyAMGHierarchy:
    """Build an AMG hierarchy and return it as a callable suitable for
    use as ``M^{-1}`` inside any iterative solver (CG, BiCGStab, ...).

    Forwarding keyword arguments to :meth:`PyAMGHierarchy.from_coo`.
    Returns the hierarchy object itself, which is callable.
    """
    return PyAMGHierarchy.from_coo(val, row, col, shape, **kwargs)
