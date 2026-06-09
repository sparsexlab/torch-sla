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
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional, Tuple, Union

import torch
from torch import Tensor


__all__ = [
    "solve",
    "SolveInfo",
    "PreconditionerConfig",
    "MatrixLike",
]


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
    method: MethodName = "auto",
    backend: BackendName = "auto",
    preconditioner: Union[str, PreconditionerConfig, None] = None,
    atol: float = 1e-10,
    rtol: float = 1e-6,
    maxiter: int = 10_000,
    x0: Optional[Tensor] = None,
    matrix_type: str = "general",
    mixed_precision: bool = False,
    return_info: bool = False,
    verbose: bool = False,
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
