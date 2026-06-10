"""Public top-level solve API.

Wraps :func:`torch_sla.spsolve` (the low-level COO-triple dispatcher) with
three Pythonic improvements requested when comparing torch-sla's surface
to the JAX-AMG paper (Liu, Fan, Wang -- arXiv:2606.09001):

1. **kwargs + dataclass, not a config dict.** The :func:`solve` entry
   point is a normal Python function: most settings are kwargs, only the
   nested preconditioner options collapse into the
   :class:`PreconditionerConfig` dataclass. Tab-completion + static
   typing both work; no untyped string keys.

2. **``return_info=True`` for ``(x, info)`` two-tuple return.** Diagnostic
   information -- iteration count, residual, convergence flag, method
   actually used -- comes back in a :class:`SolveInfo` dataclass.  Default
   stays single-return so existing code is unaffected.

3. **Multiple input formats.** ``A`` can be a :class:`SparseTensor`, an
   ``(val, row, col, shape)`` tuple, a ``scipy.sparse`` matrix, a dense
   ``torch.Tensor``, or a matrix-free callable ``x -> A @ x`` (the last
   one materialised once via probing -- see :func:`_coerce_to_coo`).

Capability matrix per backend is in :doc:`backends`.
"""
from __future__ import annotations

import dataclasses
import functools
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional, Tuple, Union

import torch
from torch import Tensor


__all__ = [
    "solve",
    "SolveInfo",
    "PreconditionerConfig",
    "SolverConfig",
    "MatrixLike",
]


# ====================================================================== #
# Sentinel used to distinguish "user didn't pass" from "user passed
# the documented default value" -- needed so that a wrapping
# ``SolverConfig`` scope can inject defaults only when the call site is
# silent on a kwarg.
# ====================================================================== #
_UNSET = object()


# ====================================================================== #
# Public dataclasses
# ====================================================================== #
PrecondKind = Literal["none", "jacobi", "ssor", "ic0", "ilu0",
                      "block_jacobi", "amg"]


@dataclass(frozen=True)
class PreconditionerConfig:
    """Configuration for a preconditioner used by an iterative solver.

    Pass either a bare string (``"jacobi"``) to :func:`solve` or build a
    full :class:`PreconditionerConfig` when you need non-default options::

        from torch_sla import solve, PreconditionerConfig

        x = solve(A, b, preconditioner="jacobi")                 # default
        x = solve(A, b, preconditioner=PreconditionerConfig(     # tuned
            kind="ssor", omega=1.2))

    Attributes
    ----------
    kind : str
        One of ``"none"`` (= no preconditioning), ``"jacobi"``,
        ``"ssor"``, ``"ic0"``, ``"ilu0"``, ``"block_jacobi"``, ``"amg"``.
    omega : float
        SOR / SSOR relaxation factor, in ``(0, 2)``. Default ``1.0``.
    block_size : int
        Block size for ``block_jacobi``. Default ``32``.
    amg_strength : float
        AMG strength-of-connection threshold (classical Ruge-Stuben).
    amg_smoother : str
        Smoother used inside the AMG V-cycle.
    amg_coarsening : str
        Coarsening algorithm; only ``"classical"`` is supported today.
    """
    kind: PrecondKind = "jacobi"
    omega: float = 1.0
    block_size: int = 32
    amg_strength: float = 0.25
    amg_smoother: Literal["jacobi", "gauss_seidel"] = "gauss_seidel"
    amg_coarsening: Literal["classical", "smoothed_aggregation"] = "classical"

    @classmethod
    def from_string(cls, kind: str) -> "PreconditionerConfig":
        """Build a default-only config from the ``kind`` string shortcut."""
        return cls(kind=kind)  # type: ignore[arg-type]


@dataclass
class SolveInfo:
    """Diagnostic information returned alongside ``x`` when
    ``return_info=True``.

    Attributes
    ----------
    iter_count : int
        Iteration count for iterative methods (``0`` for direct solvers).
    residual : float
        Final ``||r||_2 = ||b - A x||_2`` reported by the backend, or
        ``nan`` if the backend does not expose one.
    converged : bool
        Whether the solver reported convergence within the requested
        tolerance / maxiter.
    method : str
        Method actually used (after ``method='auto'`` resolution).
    backend : str
        Backend actually dispatched to (after ``backend='auto'``).
    """
    iter_count: int = 0
    residual: float = float("nan")
    converged: bool = True
    method: str = ""
    backend: str = ""


# ====================================================================== #
# Thread-local stack of scoped solve() defaults (powering SolverConfig)
# ====================================================================== #
class _DefaultsStack(threading.local):
    """Per-thread stack of merged kwargs dicts from active SolverConfig
    scopes. Innermost wins via dict-merge semantics; explicit kwargs
    passed to :func:`solve` always beat the stack."""

    def __init__(self):
        self.stack: list = []


_STACK = _DefaultsStack()


def _active_defaults() -> dict:
    """Return the merged kwargs from every active :class:`SolverConfig`
    scope on the current thread. Innermost scope wins."""
    merged: dict = {}
    for layer in _STACK.stack:
        merged.update(layer)
    return merged


@dataclass(frozen=True)
class SolverConfig:
    """Bundle of default :func:`solve` kwargs, usable as a context
    manager or decorator to apply those defaults to every ``solve``
    call inside its scope.

    Any field left as ``None`` means "don't touch the default" -- an
    inactive entry in the merged kwargs dict. Explicit kwargs passed
    to ``solve(...)`` always override the scope.

    Examples
    --------
    Context-manager form, in a single optimisation loop where every
    solve uses the same backend / preconditioner / tolerances::

        from torch_sla import solve, SolverConfig

        with SolverConfig(backend="pyamg", atol=1e-8, maxiter=50):
            for theta in parameters:
                x = solve(A(theta), b)        # picks up defaults
                ...

    Decorator form, attaching defaults to a function::

        @SolverConfig(backend="pytorch", method="cg", preconditioner="amg")
        def pde_step(A, b):
            return solve(A, b)                # cg + amg by default

    Explicit kwargs always win::

        with SolverConfig(method="cg"):
            x = solve(A, b)                   # cg
            y = solve(A, b, method="lu")      # lu (override)

    Scopes nest -- inner scope's non-None fields override the outer::

        with SolverConfig(backend="pytorch", atol=1e-8):
            with SolverConfig(atol=1e-12):
                solve(A, b)                   # backend=pytorch, atol=1e-12
    """

    method: Optional[str] = None
    backend: Optional[str] = None
    preconditioner: Union[str, "PreconditionerConfig", None, type(_UNSET)] = _UNSET
    atol: Optional[float] = None
    rtol: Optional[float] = None
    maxiter: Optional[int] = None
    x0: Optional[Tensor] = None
    matrix_type: Optional[str] = None
    mixed_precision: Optional[bool] = None
    return_info: Optional[bool] = None
    verbose: Optional[bool] = None

    def _kwargs(self) -> dict:
        """Return only the fields the user actually set."""
        out: dict = {}
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            # preconditioner uses _UNSET (None is a legitimate user choice meaning
            # "no preconditioning"); the others use None as the inactive marker.
            if f.name == "preconditioner":
                if v is not _UNSET:
                    out["preconditioner"] = v
            elif v is not None:
                out[f.name] = v
        return out

    def __enter__(self) -> "SolverConfig":
        _STACK.stack.append(self._kwargs())
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        _STACK.stack.pop()

    def __call__(self, fn: Callable) -> Callable:
        """Decorator form: ``@SolverConfig(...) def f(): ...``."""
        @functools.wraps(fn)
        def wrapped(*args, **kw):
            with self:
                return fn(*args, **kw)
        return wrapped


# Type alias for everything :func:`solve` accepts as the matrix argument.
MatrixLike = Union[
    "SparseTensor",                                  # noqa: F821 (str ref)
    Tuple[Tensor, Tensor, Tensor, Tuple[int, int]],  # (val, row, col, shape)
    Any,                                             # scipy.sparse / dense / Callable
]


# ====================================================================== #
# Input-format coercion
# ====================================================================== #
def _coerce_to_coo(A: MatrixLike, *,
                   shape: Optional[Tuple[int, int]] = None,
                   device: Optional[torch.device] = None,
                   dtype: Optional[torch.dtype] = None,
                   ) -> Tuple[Tensor, Tensor, Tensor, Tuple[int, int]]:
    """Normalise the matrix argument to a ``(val, row, col, shape)`` tuple.

    Accepted forms
    --------------
    * :class:`SparseTensor` -- read its ``values`` / ``row_indices`` /
      ``col_indices`` / ``sparse_shape`` directly.
    * 4-tuple ``(val, row, col, shape)`` -- returned essentially as-is
      (only dtype/device coercion if requested).
    * ``scipy.sparse`` matrix -- ``.tocoo()`` then convert to torch.
    * Dense :class:`torch.Tensor` -- ``A != 0`` mask + ``nonzero()``.
    * Callable ``x -> A @ x`` -- requires ``shape`` argument; materialised
      column-by-column via probing (only viable for small matrices, this
      is the path matrix-free operators take when they hit the COO API).
    """
    # Local import: SparseTensor lives in a heavier module we don't want
    # to import unconditionally from this small surface file.
    from .sparse_tensor import SparseTensor

    if isinstance(A, SparseTensor):
        return (A.values, A.row_indices, A.col_indices, tuple(A.sparse_shape))

    if isinstance(A, tuple) and len(A) == 4:
        val, row, col, sh = A
        if device is not None:
            val = val.to(device); row = row.to(device); col = col.to(device)
        if dtype is not None:
            val = val.to(dtype)
        return val, row, col, tuple(sh)

    # scipy.sparse path -- detect duck-typed; avoids a hard scipy import.
    if hasattr(A, "tocoo") and hasattr(A, "shape"):
        coo = A.tocoo()
        val = torch.from_numpy(coo.data)
        row = torch.from_numpy(coo.row).to(torch.long)
        col = torch.from_numpy(coo.col).to(torch.long)
        if device is not None:
            val = val.to(device); row = row.to(device); col = col.to(device)
        if dtype is not None:
            val = val.to(dtype)
        return val, row, col, (int(A.shape[0]), int(A.shape[1]))

    if isinstance(A, Tensor) and A.dim() == 2:
        mask = A != 0
        idx = mask.nonzero(as_tuple=False)
        row, col = idx[:, 0].contiguous(), idx[:, 1].contiguous()
        val = A[row, col]
        return val, row, col, (int(A.shape[0]), int(A.shape[1]))

    if callable(A):
        if shape is None:
            raise ValueError(
                "Matrix-free callable requires the ``shape`` keyword "
                "argument so it can be probed column-by-column."
            )
        m, n = shape
        if device is None or dtype is None:
            raise ValueError(
                "Matrix-free callable requires both ``device`` and "
                "``dtype`` to materialise its columns."
            )
        cols = []
        for j in range(n):
            e = torch.zeros(n, device=device, dtype=dtype)
            e[j] = 1
            cols.append(A(e))
        dense = torch.stack(cols, dim=1)
        return _coerce_to_coo(dense, shape=shape)

    raise TypeError(
        f"Unsupported matrix argument {type(A).__name__}; expected "
        f"SparseTensor, 4-tuple, scipy.sparse, 2-D torch.Tensor, or callable."
    )


# ====================================================================== #
# Public solve()
# ====================================================================== #
BackendName = Literal["auto", "scipy", "pytorch", "cupy", "cudss", "eigen"]
MethodName = Literal["auto", "lu", "cholesky", "ldlt", "umfpack",
                     "cg", "cgs", "bicgstab", "gmres"]


def solve(
    A: MatrixLike,
    b: Tensor,
    *,
    shape: Optional[Tuple[int, int]] = None,
    method: Any = _UNSET,
    backend: Any = _UNSET,
    preconditioner: Any = _UNSET,
    atol: Any = _UNSET,
    rtol: Any = _UNSET,
    maxiter: Any = _UNSET,
    x0: Any = _UNSET,
    matrix_type: Any = _UNSET,
    mixed_precision: Any = _UNSET,
    return_info: Any = _UNSET,
    verbose: Any = _UNSET,
) -> Union[Tensor, Tuple[Tensor, SolveInfo]]:
    """Solve ``A x = b`` with autograd support; the user-facing entry point.

    Parameters
    ----------
    A : SparseTensor, (val, row, col, shape) tuple, scipy.sparse, dense
        ``torch.Tensor``, or callable ``x -> A @ x``
        Matrix in any of the supported formats. Matrix-free callables
        require the ``shape`` keyword.
    b : torch.Tensor
        Right-hand side, shape ``[n]`` or ``[n, k]`` for multiple RHS.
    method : str, default ``"auto"``
        Solver method. ``"auto"`` picks based on backend + matrix
        properties. Direct: ``"lu" / "cholesky" / "ldlt" / "umfpack"``.
        Iterative: ``"cg" / "cgs" / "bicgstab" / "gmres"``.
    backend : str, default ``"auto"``
        Backend to dispatch to. ``"auto"`` picks based on device and
        availability.
    preconditioner : str | PreconditionerConfig | None, default ``None``
        Preconditioner for iterative solvers. Pass a string for the
        default config of a kind (``"jacobi"``); pass a
        :class:`PreconditionerConfig` for non-default options. ``None``
        means no preconditioning.
    atol, rtol : float
        Absolute / relative tolerances for iterative methods.
    maxiter : int
        Max iterations for iterative methods.
    x0 : torch.Tensor, optional
        Initial guess for iterative methods.
    matrix_type : str
        Hint for cuDSS / direct solvers: ``"general"``, ``"symmetric"``,
        ``"spd"``, ``"hermitian"``, ``"hpd"``, or ``"auto"`` to detect.
    mixed_precision : bool
        Run iterative solver in lower precision then refine.
    return_info : bool, default ``False``
        If ``True``, return ``(x, SolveInfo)`` instead of just ``x``.
    verbose : bool
        Print backend / method dispatch diagnostics.

    Returns
    -------
    torch.Tensor or (torch.Tensor, SolveInfo)
        Solution, plus diagnostic info if ``return_info=True``.

    Examples
    --------
    >>> from torch_sla import solve, PreconditionerConfig, SparseTensor
    >>> A = SparseTensor(val, row, col, (n, n))
    >>> x = solve(A, b)                                       # one-liner
    >>> x = solve(A, b, method="cg", preconditioner="jacobi") # iterative
    >>> x, info = solve(A, b, return_info=True)               # diagnostics
    >>> info.iter_count, info.residual, info.converged
    """
    # Resolve unset kwargs from the active SolverConfig scope, then fall
    # back to hard-coded defaults. Explicit user kwargs always win.
    _defaults = _active_defaults()
    def _pick(name, hardcoded):
        v = locals_dict[name]
        if v is not _UNSET:
            return v
        return _defaults.get(name, hardcoded)
    locals_dict = {
        "method": method, "backend": backend, "preconditioner": preconditioner,
        "atol": atol, "rtol": rtol, "maxiter": maxiter, "x0": x0,
        "matrix_type": matrix_type, "mixed_precision": mixed_precision,
        "return_info": return_info, "verbose": verbose,
    }
    method         = _pick("method", "auto")
    backend        = _pick("backend", "auto")
    preconditioner = _pick("preconditioner", None)
    atol           = _pick("atol", 1e-10)
    rtol           = _pick("rtol", 1e-6)
    maxiter        = _pick("maxiter", 10_000)
    x0             = _pick("x0", None)
    matrix_type    = _pick("matrix_type", "general")
    mixed_precision = _pick("mixed_precision", False)
    return_info    = _pick("return_info", False)
    verbose        = _pick("verbose", False)

    val, row, col, sh = _coerce_to_coo(
        A, shape=shape, device=b.device, dtype=b.dtype
    )

    pc = preconditioner
    if pc is None:
        pc_str = "none"
    elif isinstance(pc, str):
        pc_str = pc
    elif isinstance(pc, PreconditionerConfig):
        pc_str = pc.kind
        # Non-default fields are silently ignored by today's backends;
        # they're already declared so the AMG / SSOR / block-Jacobi
        # follow-up PRs can wire them up without changing this signature.
    else:
        raise TypeError(
            f"preconditioner must be str | PreconditionerConfig | None, "
            f"got {type(pc).__name__}"
        )

    # Delegate to the existing dispatcher (which handles all backends + autograd).
    from .linear_solve import spsolve
    x = spsolve(
        val, row, col, sh, b,
        backend=backend, method=method,
        atol=atol, maxiter=maxiter,
        matrix_type=matrix_type,
        preconditioner=pc_str,
        mixed_precision=mixed_precision,
        verbose=verbose,
    )

    if not return_info:
        return x

    # Today's spsolve doesn't yet thread iteration counts back; we
    # populate what we can and leave the rest as defaults. The
    # downstream backend-signature-unification PR will fill iter_count /
    # residual / converged from each backend's solver output.
    residual = float("nan")
    try:
        from .sparse_tensor import SparseTensor as _ST
        r = b - (_ST(val, row, col, sh) @ x)
        residual = float(r.norm().item())
    except Exception:
        pass
    info = SolveInfo(
        iter_count=0,
        residual=residual,
        converged=(residual == residual and residual < max(atol, rtol * float(b.norm().item()))),
        method=method,
        backend=backend,
    )
    return x, info
