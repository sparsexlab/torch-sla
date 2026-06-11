"""AmgX backend: GPU AMG / Krylov solvers via torch-amgx.

NVIDIA AmgX (https://github.com/NVIDIA/AMGX) is a state-of-the-art
GPU-resident sparse-linear-algebra library, providing classical
algebraic multigrid, smoothed aggregation, and a suite of Krylov
methods (PCG, PBICGSTAB, FGMRES, ...). torch-amgx
(https://github.com/sparsexlab/torch-amgx) is our maintained
PyTorch-native binding, replacing the older pyamgx wrapper.

torch-amgx ships pre-compiled wheels for Linux + Windows + NVIDIA CUDA:

    pip install torch-amgx        # one wheel per (OS, py, CUDA major)

macOS is not supported by NVIDIA for CUDA, so this backend is
unavailable there. ``torch-amgx`` is an *optional* dependency of
``torch-sla`` -- the import is lazy and ``is_amgx_available()`` returns
``False`` cleanly if the package or CUDA is missing.

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

Compared to the previous pyamgx-based implementation:

* CSR assembly happens **on-device** via ``torch_amgx.Solver.setup_coo``
  -- no GPU<->CPU round-trip for the matrix at construction time.
* The solve path stays on GPU: ``b`` is consumed and ``x`` is returned
  as a CUDA tensor without any numpy detour.
* ``Config`` is a frozen dataclass from torch-amgx, not a printf-style
  string we hand-roll. The literal-config-string escape hatch is
  still supported (pass it as the ``method`` kwarg).
"""
from __future__ import annotations

import warnings
from typing import Any, Callable, Optional, Tuple

import torch
from torch import Tensor


# torch-amgx is imported lazily so this module can be imported on
# machines without it installed (macOS, CPU-only Linux/Windows).
_torch_amgx = None


def _load_torch_amgx():
    """Lazy import of torch-amgx with a clear error when it's missing."""
    global _torch_amgx
    if _torch_amgx is not None:
        return _torch_amgx
    try:
        import torch_amgx
    except ImportError as exc:
        raise ImportError(
            "torch_sla.backends.amgx_backend requires the optional "
            "dependency torch-amgx. Install via\n\n"
            "    pip install torch-amgx\n\n"
            "which ships pre-compiled wheels for Linux + Windows + "
            "NVIDIA CUDA. macOS is not supported by NVIDIA for CUDA."
        ) from exc
    _torch_amgx = torch_amgx
    return _torch_amgx


def is_amgx_available() -> bool:
    """Return ``True`` iff torch-amgx is importable and CUDA is usable."""
    try:
        tam = _load_torch_amgx()
    except ImportError:
        return False
    return tam.is_available()


# ====================================================================== #
# Config builder -- delegates printf-style config-string construction
# to torch_amgx.Config. The literal-config-string escape hatch is kept
# for parity with the previous backend API.
# ====================================================================== #
def _build_config(method: str, *, tol: float, maxiter: int,
                  preconditioner: str = "amg"):
    tam = _load_torch_amgx()
    if "config_version" in method.lower():
        return tam.Config(amgx_config_str=method)
    return tam.Config(method=method.lower(), tol=tol, maxiter=maxiter,
                      preconditioner=preconditioner)


# ====================================================================== #
# AmgXSolver: thin wrapper around torch_amgx.Solver
# ====================================================================== #
class AmgXSolver:
    """Captured AmgX resource graph (config + resources + matrix +
    factored solver).

    One instance owns the GPU setup for one matrix and can be solved
    against any number of right-hand sides. Garbage collecting the
    instance tears down the underlying AmgX handles.

    Build via :meth:`from_coo` (the path the torch-sla dispatcher takes).
    """

    def __init__(self, *, config_str: str):
        tam = _load_torch_amgx()
        # Keep ``config_str`` around so the cache key is stable across
        # equivalent Config dataclass instances.
        self._config_str = config_str
        self._solver = tam.Solver(tam.Config(amgx_config_str=config_str))
        self._is_setup = False

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_coo(cls, val: Tensor, row: Tensor, col: Tensor,
                 shape: Tuple[int, int], *, config_str: str) -> "AmgXSolver":
        """Build a solver from a torch COO triple. CSR assembly happens
        on-device -- no GPU<->CPU round-trip for the matrix."""
        s = cls(config_str=config_str)
        s._solver.setup_coo(val, row, col, shape)
        s._is_setup = True
        return s

    # ------------------------------------------------------------------ #
    # Solve
    # ------------------------------------------------------------------ #
    def solve(self, b: Tensor, *, x0: Optional[Tensor] = None) -> Tensor:
        """Solve ``A x = b`` using the previously-set-up solver.

        ``x0`` provides a warm-start initial guess when supplied.
        """
        if not self._is_setup:
            raise RuntimeError("AmgXSolver was not set up against a matrix")
        if x0 is None:
            return self._solver.solve(b)
        x = x0.clone()
        self._solver.solve_into(b, x)
        return x

    # Preconditioner interface so this object plugs into outer iterative
    # solvers as M^{-1} r -- AmgX's V-cycle is the preconditioner.
    def __call__(self, r: Tensor) -> Tensor:
        return self.solve(r)


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
               preconditioner: str = "amg",
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
        GPU; ``b.to("cuda")`` before calling if needed.
    tol : float
        Absolute residual tolerance. Default ``1e-8``.
    maxiter : int
        Max outer iterations. Default ``100``.
    method : str
        Solver method. ``"auto"`` (= ``"pbicgstab"``) is the most
        robust default; alternatives are ``"cg"`` / ``"pcg"`` /
        ``"bicgstab"`` / ``"pbicgstab"`` / ``"gmres"`` /
        ``"fgmres"`` / ``"amg"`` (standalone V-cycle iteration).
        A literal AmgX config string is also accepted (must contain
        ``config_version=2``).
    preconditioner : str
        Inner preconditioner. Default ``"amg"`` (single V-cycle classical
        AMG, the historical default). Alternatives mirror AmgX's
        built-ins: ``"jacobi_l1"`` / ``"block_jacobi"`` /
        ``"multicolor_gs"`` / ``"multicolor_dilu"`` /
        ``"multicolor_ilu"`` / ``"chebyshev"`` / ``"polynomial"`` /
        ``"kaczmarz"`` / ``"none"`` (unpreconditioned Krylov).
        Ignored when ``method="amg"`` (AMG is itself the solver).
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

    cfg = _build_config(method, tol=tol, maxiter=maxiter,
                        preconditioner=preconditioner)
    config_str = cfg.build_config_str()

    if solver is None:
        solver = _build_or_lookup_solver(val, row, col, shape,
                                         config_str=config_str)

    if return_info:
        x, info = solver._solver.solve(b, return_info=True)
        return x, {
            "iter_count": info.iter_count,
            "residual": info.residual,
            "converged": info.converged,
            "method":  f"amgx-{method}",
            "backend": "amgx",
        }
    return solver.solve(b)


# ====================================================================== #
# Preconditioner factory
# ====================================================================== #
def amgx_preconditioner(val: Tensor, row: Tensor, col: Tensor,
                        shape: Tuple[int, int],
                        *,
                        tol: float = 1e-3,
                        maxiter: int = 1,
                        method: str = "amg",
                        preconditioner: str = "amg",
                        **kwargs) -> AmgXSolver:
    """Build an AmgX solver and return it as a callable usable as
    ``M^{-1}`` inside any outer iterative solver.

    Defaults to a single-V-cycle classical AMG configuration -- the
    standard "AMG as preconditioner for CG/BiCGStab" pattern. Pass
    ``preconditioner=`` to compose AmgX's other preconditioners
    (jacobi_l1, block_jacobi, multicolor_dilu, chebyshev, ...) inside
    the requested outer ``method``.
    """
    cfg = _build_config(method, tol=tol, maxiter=maxiter,
                        preconditioner=preconditioner)
    config_str = cfg.build_config_str()
    return _build_or_lookup_solver(val, row, col, shape,
                                   config_str=config_str)
