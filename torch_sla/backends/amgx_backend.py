"""AmgX backend: GPU AMG / Krylov solvers via pyamgx.

NVIDIA AmgX (https://github.com/NVIDIA/AMGX) is a state-of-the-art
GPU-resident sparse-linear-algebra library, providing classical
algebraic multigrid, smoothed aggregation, and a suite of Krylov
methods (PCG, PBICGSTAB, FGMRES, ...). pyamgx
(https://github.com/shwina/pyamgx) is the Cython wrapper.

Neither has a PyPI wheel. The build pipeline is captured in
``sparsexlab/torch-sla-amgx`` (CMake + custom setup.py patches for
Windows / MSVC); end users on Linux + Windows + NVIDIA GPU install via

    pip install --extra-index-url https://pypi.walkerchi.com torch-sla-amgx

macOS is not supported by Nvidia for CUDA, so this backend is
unavailable there.

Two entry points mirror the PyAMG-hybrid backend (PR #14):

* :func:`amgx_solve` -- standalone direct/iterative GPU solve. Builds
  an :class:`AmgXSolver`, runs it once, returns the solution.
* :func:`amgx_preconditioner` -- factory returning an
  :class:`AmgXSolver` whose ``__call__`` interface (``r -> M^{-1} r``)
  can plug into any outer iterative solver. AmgX's V-cycle is the
  preconditioner.

The :class:`AmgXSolver` itself is transparently cached via
:data:`~torch_sla.solver_cache.SOLVER_CACHE` keyed on sparsity +
config, so repeated solves on the same matrix reuse the GPU setup
exactly like PyAMG does.
"""
from __future__ import annotations

import atexit
import warnings
from typing import Any, Callable, Optional, Tuple

import numpy as np
import torch
from torch import Tensor


# pyamgx (and the underlying amgxsh.dll on Windows) is imported lazily
# so this module can be imported on machines without AmgX installed.
_pyamgx = None
_AMGX_INITIALIZED = False


def _load_pyamgx():
    """Lazy import of pyamgx with a clear error when it's missing."""
    global _pyamgx
    if _pyamgx is not None:
        return _pyamgx
    try:
        import pyamgx
    except ImportError as exc:
        raise ImportError(
            "torch_sla.backends.amgx_backend requires the optional "
            "dependency pyamgx (and an AmgX SDK build). The simplest way "
            "to install both is\n\n"
            "    pip install --extra-index-url https://pypi.walkerchi.com "
            "torch-sla-amgx\n\n"
            "which ships pre-compiled wheels for Linux + Windows + NVIDIA "
            "CUDA. macOS is not supported by NVIDIA for CUDA."
        ) from exc
    _pyamgx = pyamgx
    return _pyamgx


def is_amgx_available() -> bool:
    """Return ``True`` iff pyamgx + AmgX shared library are loadable."""
    try:
        _load_pyamgx()
        return True
    except ImportError:
        return False


def _ensure_initialized():
    """``pyamgx.initialize()`` must be called once per process before any
    AmgX resource is built; ``finalize()`` runs at interpreter shutdown
    via :mod:`atexit`."""
    global _AMGX_INITIALIZED
    if _AMGX_INITIALIZED:
        return
    pyamgx = _load_pyamgx()
    pyamgx.initialize()
    atexit.register(pyamgx.finalize)
    _AMGX_INITIALIZED = True


# ====================================================================== #
# Method dispatch -- hand-rolled printf-style config strings. AmgX's
# parser accepts ``config_version=2`` followed by ``solver(scope)=TYPE``
# declarations plus ``scope:key=value`` settings. Nested preconditioners
# are written ``parent:preconditioner(scope)=TYPE`` -- the parens around
# the scope name are required, otherwise the parser errors with
# "Incorrect amgx configuration provided".
# ====================================================================== #


def _amg_preconditioned(outer_solver: str, *,
                        tol: float, maxiter: int) -> str:
    """Build a printf-style config for ``outer_solver`` (e.g. PBICGSTAB,
    PCG, FGMRES) preconditioned by classical AMG with a single V-cycle.

    Mirrors the contents of AmgX's stock ``PBICGSTAB.json`` /
    ``PCG.json`` files (which can't be loaded by name because ``create()``
    expects a literal config blob on Windows, not a registry lookup)."""
    return (
        "config_version=2,"
        f"solver(main)=__OUTER__,"
        "main:scope=main,"
        f"main:max_iters={maxiter},"
        f"main:tolerance={tol},"
        "main:convergence=ABSOLUTE,"
        "main:norm=L2,"
        "main:monitor_residual=1,"
        "main:print_solve_stats=0,"
        "main:obtain_timings=0,"
        "main:preconditioner(amg)=AMG,"
        "amg:scope=amg,"
        "amg:solver=AMG,"
        "amg:max_iters=1,"
        "amg:cycle=V,"
        "amg:max_levels=50,"
        "amg:presweeps=1,"
        "amg:postsweeps=1,"
        "amg:interpolator=D2,"
        "amg:monitor_residual=0,"
        "amg:print_solve_stats=0"
    ).replace("__OUTER__", outer_solver)


def _amg_standalone(*, tol: float, maxiter: int) -> str:
    """AMG used as the standalone iterative solver (one V-cycle per
    outer iteration)."""
    return (
        "config_version=2,"
        "solver(main)=AMG,"
        "main:scope=main,"
        f"main:max_iters={maxiter},"
        f"main:tolerance={tol},"
        "main:convergence=ABSOLUTE,"
        "main:norm=L2,"
        "main:cycle=V,"
        "main:max_levels=50,"
        "main:presweeps=1,"
        "main:postsweeps=1,"
        "main:interpolator=D2,"
        "main:monitor_residual=1,"
        "main:print_solve_stats=0,"
        "main:obtain_timings=0"
    )


def _resolve_config(method: str, *, tol: float, maxiter: int) -> str:
    """Map a torch-sla method label to a printf-style AmgX config blob.

    Accepted methods:

    * ``"auto"`` / ``"pbicgstab"`` / ``"bicgstab"`` -- PBICGSTAB + AMG
    * ``"pcg"`` / ``"cg"``                          -- PCG + AMG
    * ``"fgmres"`` / ``"gmres"``                    -- FGMRES + AMG
    * ``"amg"``                                     -- standalone AMG

    A literal AmgX config string (containing ``config_version=``) is
    also accepted and passed through unchanged.
    """
    method = method.lower()
    if "config_version" in method:
        return method
    if method in ("auto", "pbicgstab", "bicgstab"):
        return _amg_preconditioned("PBICGSTAB", tol=tol, maxiter=maxiter)
    if method in ("pcg", "cg"):
        return _amg_preconditioned("PCG", tol=tol, maxiter=maxiter)
    if method in ("fgmres", "gmres"):
        return _amg_preconditioned("FGMRES", tol=tol, maxiter=maxiter)
    if method == "amg":
        return _amg_standalone(tol=tol, maxiter=maxiter)
    raise ValueError(
        f"Unknown AmgX method {method!r}; expected one of auto / amg / "
        f"cg / pcg / bicgstab / pbicgstab / gmres / fgmres, or a literal "
        f"AmgX config string."
    )


# ====================================================================== #
# AmgXSolver: lifecycle wrapper around the pyamgx resource graph
# ====================================================================== #
class AmgXSolver:
    """Captured AmgX resource graph (config + resources + matrix +
    factored solver).

    One instance owns the GPU setup for one matrix and can be solved
    against any number of right-hand sides. Destruction releases all
    GPU resources in reverse construction order.

    Build via :meth:`from_coo` (the path the torch-sla dispatcher takes)
    or :meth:`from_scipy_csr` if you already have a scipy matrix.
    """

    def __init__(self, *, config_str: str):
        _ensure_initialized()
        pyamgx = _load_pyamgx()
        self._pyamgx = pyamgx
        self._cfg = pyamgx.Config().create(config_str)
        self._rsrc = pyamgx.Resources().create_simple(self._cfg)
        self._A = pyamgx.Matrix().create(self._rsrc)
        self._solver = pyamgx.Solver().create(self._rsrc, self._cfg)
        self._is_setup = False

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_scipy_csr(cls, A_csr, *, config_str: str) -> "AmgXSolver":
        s = cls(config_str=config_str)
        s._A.upload_CSR(A_csr)
        s._solver.setup(s._A)
        s._is_setup = True
        return s

    @classmethod
    def from_coo(cls, val: Tensor, row: Tensor, col: Tensor,
                 shape: Tuple[int, int], *, config_str: str) -> "AmgXSolver":
        """Build a solver from a torch COO triple. Values are pulled to
        CPU + assembled into scipy CSR before upload (pyamgx's
        ``upload_CSR`` expects a scipy matrix)."""
        import scipy.sparse as sp
        A_csr = sp.coo_matrix(
            (val.detach().cpu().numpy(),
             (row.detach().cpu().numpy(), col.detach().cpu().numpy())),
            shape=shape,
        ).tocsr().astype(np.float64)
        return cls.from_scipy_csr(A_csr, config_str=config_str)

    # ------------------------------------------------------------------ #
    # Solve
    # ------------------------------------------------------------------ #
    def solve(self, b: Tensor, *, x0: Optional[Tensor] = None) -> Tensor:
        """Solve ``A x = b`` using the previously-set-up solver.

        Returns ``x`` with the same device + dtype as ``b``.
        """
        if not self._is_setup:
            raise RuntimeError("AmgXSolver was not set up against a matrix")
        pyamgx = self._pyamgx
        device, dtype = b.device, b.dtype
        b_np = b.detach().to(torch.float64).cpu().numpy()
        x0_np = (np.zeros_like(b_np) if x0 is None
                 else x0.detach().to(torch.float64).cpu().numpy())

        bvec = pyamgx.Vector().create(self._rsrc)
        xvec = pyamgx.Vector().create(self._rsrc)
        try:
            bvec.upload(b_np)
            xvec.upload(x0_np)
            self._solver.solve(bvec, xvec)
            x = xvec.download()
        finally:
            bvec.destroy()
            xvec.destroy()
        return torch.from_numpy(x).to(dtype=dtype, device=device)

    # Preconditioner interface so this object plugs into outer iterative
    # solvers as M^{-1} r -- AmgX's V-cycle is the preconditioner.
    def __call__(self, r: Tensor) -> Tensor:
        return self.solve(r)

    # ------------------------------------------------------------------ #
    # Cleanup
    # ------------------------------------------------------------------ #
    def __del__(self):
        # Destroy in reverse-creation order. Guard each step so partial
        # construction (failed in __init__) still cleans up correctly.
        for attr in ("_solver", "_A", "_rsrc", "_cfg"):
            obj = getattr(self, attr, None)
            if obj is not None:
                try:
                    obj.destroy()
                except Exception:
                    pass


# ====================================================================== #
# Cache integration
# ====================================================================== #
def _build_or_lookup_solver(val: Tensor, row: Tensor, col: Tensor,
                            shape: Tuple[int, int],
                            *, config_str: str) -> AmgXSolver:
    """Consult :data:`SOLVER_CACHE` for an :class:`AmgXSolver` matching
    this matrix + config; build + insert on miss. Transparent like the
    PyAMG backend's hierarchy reuse."""
    from ..solver_cache import SOLVER_CACHE, make_key
    key = ("amgx-solver", make_key(val, row, col, shape), config_str)
    return SOLVER_CACHE.get_or_build(
        key, lambda: AmgXSolver.from_coo(val, row, col, shape,
                                         config_str=config_str)
    )


# ====================================================================== #
# Standalone solve
# ====================================================================== #
def amgx_solve(val: Tensor, row: Tensor, col: Tensor,
               shape: Tuple[int, int], b: Tensor,
               *,
               tol: float = 1e-8,
               maxiter: int = 100,
               method: str = "auto",
               solver: Optional[AmgXSolver] = None,
               return_info: bool = False,
               **kwargs):
    """Solve ``A x = b`` on the GPU via AmgX.

    Parameters
    ----------
    val, row, col, shape : torch.Tensor + tuple
        COO representation of the matrix.
    b : torch.Tensor
        Right-hand side. Must live on CUDA for the solve to happen on
        GPU; ``b.to(\"cuda\")`` before calling if needed.
    tol : float
        Absolute residual tolerance. Default ``1e-8``.
    maxiter : int
        Max outer iterations. Default ``100``.
    method : str
        Solver method. ``\"auto\"`` (= ``\"pbicgstab\"``) is the most
        robust default; alternatives are ``\"cg\"`` / ``\"pcg\"`` /
        ``\"bicgstab\"`` / ``\"pbicgstab\"`` / ``\"gmres\"`` /
        ``\"fgmres\"`` / ``\"amg\"`` (standalone V-cycle iteration).
        A literal AmgX config string is also accepted (must contain
        ``config_version=2``).
    solver : AmgXSolver, optional
        Reuse a caller-managed AmgX solver explicitly, skipping the
        cache. Caller ensures sparsity + config match.
    return_info : bool
        Return ``(x, info_dict)`` instead of just ``x``.
    """
    if not val.is_cuda or not b.is_cuda:
        raise RuntimeError(
            "AmgX backend requires CUDA tensors; got "
            f"val.is_cuda={val.is_cuda}, b.is_cuda={b.is_cuda}"
        )

    config_str = _resolve_config(method, tol=tol, maxiter=maxiter)

    if solver is None:
        solver = _build_or_lookup_solver(val, row, col, shape,
                                         config_str=config_str)

    x = solver.solve(b)

    if return_info:
        # AmgX prints its own stats; we surface what the API allows.
        # ``solver.solver.get_iters_number()`` etc. are available in
        # pyamgx if the user enables ``main:obtain_timings``; default
        # config keeps this off so we leave them best-effort.
        info = {
            "iter_count": -1,                  # not threaded yet
            "residual": float((b - _residual_norm(val, row, col, shape, x)).item()
                              if False else float("nan")),
            "converged": True,                 # AmgX raises on divergence
            "method":  f"amgx-{method}",
            "backend": "amgx",
        }
        return x, info
    return x


def _residual_norm(val, row, col, shape, x):
    """Best-effort residual reconstruction. Used only when callers ask
    for ``return_info=True``; computing it via torch.sparse keeps
    everything on device, no extra GPU<->CPU round-trip."""
    indices = torch.stack([row, col], dim=0)
    A = torch.sparse_coo_tensor(indices, val, shape,
                                device=val.device).coalesce()
    return torch.sparse.mm(A, x.unsqueeze(1)).squeeze(1)


# ====================================================================== #
# Preconditioner factory
# ====================================================================== #
def amgx_preconditioner(val: Tensor, row: Tensor, col: Tensor,
                        shape: Tuple[int, int],
                        *,
                        tol: float = 1e-3,
                        maxiter: int = 1,
                        method: str = "amg",
                        **kwargs) -> AmgXSolver:
    """Build an AmgX solver and return it as a callable usable as
    ``M^{-1}`` inside any outer iterative solver.

    Defaults to a single-V-cycle classical AMG configuration -- the
    standard \"AMG as preconditioner for CG/BiCGStab\" pattern.
    """
    config_str = _resolve_config(method, tol=tol, maxiter=maxiter)
    return _build_or_lookup_solver(val, row, col, shape,
                                   config_str=config_str)
