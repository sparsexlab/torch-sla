"""Chainable :class:`~torch_sla.SolverConfig` preset builders.

The user-facing entry points fall into three groups:

1. **Auto-detect** (recommended)::

       from torch_sla import SolverConfig, solve

       with SolverConfig.auto(A):
           x = solve(A, b)

2. **Explicit per-axis builder**. Each axis is its own chainable method,
   so the three orthogonal axes (matrix kind, device, direct/iterative)
   compose cleanly::

       SolverConfig.spd().gpu()                # SPD on GPU, iterative
       SolverConfig.spd().gpu().direct()       # SPD on GPU, direct (cuDSS Cholesky)
       SolverConfig.general().cpu().direct()   # general on CPU, SciPy SuperLU
       SolverConfig.indefinite().gpu()         # AmgX + block-Jacobi
       SolverConfig.convection_diffusion().gpu()  # AmgX FGMRES + DILU

   The matrix-kind methods (``spd``, ``general``, ``indefinite``,
   ``convection_diffusion``) are classmethod starts. The device and
   direct/iterative methods (``gpu``, ``cpu``, ``direct``, ``iterative``)
   are instance modifiers that re-look-up the preset table with the new
   axis value, so chain order doesn't matter:
   ``SolverConfig.spd().gpu().direct()`` and
   ``SolverConfig.spd().direct().gpu()`` produce the same config.

3. **Modifier chain on any preset**::

       SolverConfig.auto(A).high_accuracy()                 # tighten tolerances
       SolverConfig.spd().gpu().replace(maxiter=10_000)     # override a field

Picking the (backend, method, preconditioner) triple is half the battle
for sparse iterative solvers; the wrong preconditioner can make the
difference between converging in 30 iterations and stalling forever. The
heuristics encoded here come from the standard references (Saad's
*Iterative Methods*, AmgX's stock JSONs, PETSc's PCType docs, Trilinos'
Belos+Ifpack recommendations) and are *advisory* -- explicit kwargs on
:func:`solve` and inner :class:`SolverConfig` scopes still win per the
PR #17 precedence chain.

Backend availability is checked when :meth:`auto` runs, so it never
returns a config whose backend isn't installed -- e.g. on macOS it
skips ``amgx`` / ``cudss`` and falls back to ``pyamg`` / ``scipy``.
"""
from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Optional, Union

import torch
from torch import Tensor

if TYPE_CHECKING:  # pragma: no cover -- type-only
    from .solve import SolverConfig
    from .sparse_tensor import SparseTensor


# ---------------------------------------------------------------------- #
# Backend-availability shims
# ---------------------------------------------------------------------- #
def _amgx_ok() -> bool:
    try:
        from .backends import is_amgx_available
        return bool(is_amgx_available())
    except Exception:
        return False


def _cudss_ok() -> bool:
    try:
        from .backends import is_cudss_available
        return bool(is_cudss_available())
    except Exception:
        return False


def _pyamg_ok() -> bool:
    try:
        from .backends import is_pyamg_available
        return bool(is_pyamg_available())
    except Exception:
        return False


# ---------------------------------------------------------------------- #
# Device + size heuristics
# ---------------------------------------------------------------------- #
_DIRECT_DENSE_SIZE_CROSSOVER = 100_000


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _normalize_device(device: Union[str, torch.device, None]) -> str:
    """Normalize ``device`` to ``"cuda"`` or ``"cpu"``. ``"gpu"`` is an
    accepted alias for ``"cuda"`` (user-vocab vs torch-vocab)."""
    if device is None:
        return _default_device()
    if isinstance(device, torch.device):
        return device.type
    s = str(device).lower()
    if s in ("gpu", "cuda"):
        return "cuda"
    if s == "cpu":
        return "cpu"
    raise ValueError(
        f"device must be 'cpu' / 'cuda' / 'gpu' / torch.device, got {device!r}"
    )


# ====================================================================== #
# The preset lookup table: (kind, device, direct) -> ready-to-go config.
# Centralised so every chainable method funnels through one source of
# truth rather than re-encoding the heuristics.
# ====================================================================== #
def _lookup(kind: str, device: str, direct: bool) -> "SolverConfig":
    from .solve import SolverConfig

    if kind == "spd":
        if device == "cuda":
            if direct:
                return SolverConfig(backend="cudss", method="cholesky",
                                    _kind=kind, _device=device, _direct=direct)
            return SolverConfig(backend="amgx", method="pcg",
                                preconditioner="amg",
                                atol=1e-9, maxiter=500,
                                _kind=kind, _device=device, _direct=direct)
        # cpu
        if direct:
            return SolverConfig(backend="scipy", method="cholesky",
                                _kind=kind, _device=device, _direct=direct)
        return SolverConfig(backend="pyamg", method="amg",
                            atol=1e-9, maxiter=500,
                            _kind=kind, _device=device, _direct=direct)

    if kind == "general":
        if device == "cuda":
            if direct:
                return SolverConfig(backend="cudss", method="lu",
                                    _kind=kind, _device=device, _direct=direct)
            return SolverConfig(backend="amgx", method="pbicgstab",
                                preconditioner="multicolor_dilu",
                                atol=1e-8, maxiter=500,
                                _kind=kind, _device=device, _direct=direct)
        # cpu
        if direct:
            return SolverConfig(backend="scipy", method="lu",
                                _kind=kind, _device=device, _direct=direct)
        return SolverConfig(backend="scipy", method="bicgstab",
                            preconditioner="jacobi",
                            atol=1e-8, maxiter=500,
                            _kind=kind, _device=device, _direct=direct)

    if kind == "indefinite":
        if device == "cuda":
            if direct:
                return SolverConfig(backend="cudss", method="ldlt",
                                    _kind=kind, _device=device, _direct=direct)
            return SolverConfig(backend="amgx", method="pbicgstab",
                                preconditioner="block_jacobi",
                                atol=1e-8, maxiter=500,
                                _kind=kind, _device=device, _direct=direct)
        # cpu
        if direct:
            return SolverConfig(backend="scipy", method="lu",
                                _kind=kind, _device=device, _direct=direct)
        return SolverConfig(backend="scipy", method="minres",
                            atol=1e-8, maxiter=500,
                            _kind=kind, _device=device, _direct=direct)

    if kind == "convection_diffusion":
        # No direct-solve mode -- conv-diff problems are almost always
        # large enough that direct is unrealistic. ``.direct()`` on a
        # conv-diff config falls back to general(direct).
        if direct:
            return _lookup("general", device, True)
        if device == "cuda":
            return SolverConfig(backend="amgx", method="fgmres",
                                preconditioner="multicolor_dilu",
                                atol=1e-8, maxiter=500,
                                _kind=kind, _device=device, _direct=direct)
        return SolverConfig(backend="scipy", method="gmres",
                            preconditioner="jacobi",
                            atol=1e-8, maxiter=500,
                            _kind=kind, _device=device, _direct=direct)

    raise ValueError(
        f"Unknown preset kind {kind!r}; expected one of spd / general / "
        "indefinite / convection_diffusion."
    )


# ====================================================================== #
# Classmethod factories -- start of the axis chain.
# ====================================================================== #
def _spd() -> "SolverConfig":
    """SPD / HPD systems. Auto-picks GPU if CUDA available, iterative.

    Chain ``.gpu()`` / ``.cpu()`` to force a device; ``.direct()`` to
    swap iterative for a direct factorisation.
    """
    return _lookup("spd", _default_device(), False)


def _general() -> "SolverConfig":
    """General non-symmetric systems. Auto-picks GPU + iterative."""
    return _lookup("general", _default_device(), False)


def _indefinite() -> "SolverConfig":
    """Symmetric / Hermitian indefinite (saddle-point structures). Auto-
    picks GPU + iterative."""
    return _lookup("indefinite", _default_device(), False)


def _convection_diffusion() -> "SolverConfig":
    """Convection-dominated PDEs. FGMRES + multi-colour DILU on GPU."""
    return _lookup("convection_diffusion", _default_device(), False)


# ====================================================================== #
# Instance modifiers -- axis chain steps.
# Each one re-looks-up the preset table with the new axis value, so
# chain order is irrelevant:
#     SolverConfig.spd().gpu().direct() == SolverConfig.spd().direct().gpu()
# ====================================================================== #
def _gpu(self: "SolverConfig") -> "SolverConfig":
    """Switch to GPU (CUDA)."""
    kind = self._kind
    if kind is None:
        # No matrix kind chosen yet -- just stash the device hint and
        # leave backend/method untouched. A later ``.spd()`` / ``.general()``
        # call will pick it up.
        return dataclasses.replace(self, _device="cuda")
    return _lookup(kind, "cuda", bool(self._direct))


def _cpu(self: "SolverConfig") -> "SolverConfig":
    """Switch to CPU."""
    kind = self._kind
    if kind is None:
        return dataclasses.replace(self, _device="cpu")
    return _lookup(kind, "cpu", bool(self._direct))


def _direct_method(self: "SolverConfig") -> "SolverConfig":
    """Use a direct factorisation (LU / Cholesky / LDLᵀ) instead of an
    iterative Krylov method."""
    kind = self._kind
    if kind is None:
        return dataclasses.replace(self, _direct=True)
    dev = self._device or _default_device()
    return _lookup(kind, dev, True)


def _iterative(self: "SolverConfig") -> "SolverConfig":
    """Use an iterative Krylov / multigrid method (the default)."""
    kind = self._kind
    if kind is None:
        return dataclasses.replace(self, _direct=False)
    dev = self._device or _default_device()
    return _lookup(kind, dev, False)


# ====================================================================== #
# auto(A) -- the recommended entry point
# ====================================================================== #
def _auto(A: "SparseTensor",
          *,
          device: Union[str, torch.device, None] = None,
          size_hint: Optional[int] = None) -> "SolverConfig":
    """Pick a preset for matrix ``A`` based on its detected type, size,
    and the device + backend availability.

    Decision flow:

    * **SPD / HPD** + CUDA + AmgX + ``n > 100k`` -> AmgX PCG-AMG
    * **SPD / HPD** + CUDA + cuDSS -> cuDSS Cholesky
    * **SPD / HPD** + CPU + PyAMG + ``n > 100k`` -> PyAMG AMG
    * **SPD / HPD** + CPU otherwise -> SciPy Cholesky
    * **symmetric / hermitian** (indefinite) + CUDA + AmgX -> block-Jacobi BiCGStab
    * **symmetric / hermitian** otherwise -> SciPy MINRES
    * **general** + CUDA + AmgX -> multicolour-DILU PBiCGStab
    * **general** + CPU + small -> SciPy SuperLU
    * **general** + CPU + large -> SciPy BiCGStab + Jacobi
    """
    mtype = A.detect_matrix_type()
    n = size_hint if size_hint is not None else int(A.shape[0])
    dev = _normalize_device(device) if device is not None \
          else ("cuda" if A.values.is_cuda else "cpu")
    on_cuda = dev == "cuda"

    if mtype in ("spd", "hpd"):
        if on_cuda and _amgx_ok() and n > _DIRECT_DENSE_SIZE_CROSSOVER:
            return _lookup("spd", "cuda", False)
        if on_cuda and _cudss_ok():
            return _lookup("spd", "cuda", True)
        if _pyamg_ok() and n > _DIRECT_DENSE_SIZE_CROSSOVER:
            return _lookup("spd", "cpu", False)
        return _lookup("spd", "cpu", True)

    if mtype in ("symmetric", "hermitian"):
        if on_cuda and _amgx_ok():
            return _lookup("indefinite", "cuda", False)
        return _lookup("indefinite", "cpu", False)

    # general non-symmetric
    if on_cuda and _amgx_ok():
        return _lookup("general", "cuda", False)
    if not on_cuda and n <= _DIRECT_DENSE_SIZE_CROSSOVER:
        return _lookup("general", "cpu", True)
    return _lookup("general", "cpu", False)


# ====================================================================== #
# Modifier instance methods (precision + ad-hoc field overrides).
# ====================================================================== #
def _high_accuracy(self: "SolverConfig",
                   *,
                   atol: float = 1e-12,
                   rtol: float = 1e-10,
                   maxiter: int = 5000) -> "SolverConfig":
    """Tighten precision knobs while preserving the rest of the config.

    >>> SolverConfig.auto(A).high_accuracy()
    >>> SolverConfig.spd().gpu().high_accuracy(atol=1e-13)
    """
    return dataclasses.replace(self, atol=atol, rtol=rtol, maxiter=maxiter)


def _replace(self: "SolverConfig", **kwargs) -> "SolverConfig":
    """Generic field-override escape hatch.

    >>> SolverConfig.auto(A).replace(method="fgmres")
    >>> SolverConfig.spd().gpu().replace(maxiter=10_000)
    """
    return dataclasses.replace(self, **kwargs)


# ====================================================================== #
# Injection: attach factories + axis modifiers + utility modifiers as
# methods on SolverConfig.
# ====================================================================== #
def _install_on_solver_config() -> None:
    from .solve import SolverConfig
    # Classmethod factories -- start of the axis chain
    SolverConfig.spd                  = staticmethod(_spd)
    SolverConfig.general              = staticmethod(_general)
    SolverConfig.indefinite           = staticmethod(_indefinite)
    SolverConfig.convection_diffusion = staticmethod(_convection_diffusion)
    SolverConfig.auto                 = staticmethod(_auto)
    # Instance modifiers -- axis chain steps
    SolverConfig.gpu                  = _gpu
    SolverConfig.cpu                  = _cpu
    SolverConfig.direct               = _direct_method
    SolverConfig.iterative            = _iterative
    # Instance modifiers -- precision / generic field overrides
    SolverConfig.high_accuracy        = _high_accuracy
    SolverConfig.replace              = _replace


_install_on_solver_config()
