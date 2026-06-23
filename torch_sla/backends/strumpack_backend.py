"""STRUMPACK backend: portable sparse **direct** solver via ``torch-strumpack``.

STRUMPACK (https://github.com/pghysels/STRUMPACK) is a sparse direct solver
(multifrontal LU) with native **CUDA *and* HIP/ROCm** GPU support -- which is
exactly the gap NVIDIA-only cuDSS leaves on AMD hardware. ``torch-strumpack``
(https://github.com/sparsexlab/torch-strumpack) wraps it for PyTorch and is
designed to be driven as a torch-sla backend: torch-sla owns autograd and calls
the autograd-free primitives ``factor`` / ``solve`` / ``solve_transpose`` in
``torch_strumpack._core`` directly (no double differentiation).

Notes (mirroring torch-strumpack's core):
* **Direct solver only** -- method is ``lu``; no iterations / preconditioner.
* **Real float64 and complex128** -- STRUMPACK builds both ``double`` and
  ``complex<double>``; the core dispatches on dtype. The complex adjoint
  (``A^H``) is handled in torch-sla's autograd Function.
* The compiled extension is shipped per platform (cpu / cuda / rocm) inside the
  matching ``torch-strumpack`` wheel; ``is_strumpack_available()`` is ``True``
  only when that extension actually loads.
"""
from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor


# torch-strumpack is imported lazily so torch-sla imports cleanly on machines
# without it (the common case -- it is an optional, platform-specific wheel).
_ts = None


def _load_torch_strumpack():
    """Lazy import of torch-strumpack with a clear error when it's missing."""
    global _ts
    if _ts is not None:
        return _ts
    try:
        import torch_strumpack
    except ImportError as exc:
        raise ImportError(
            "torch_sla.backends.strumpack_backend requires the optional "
            "dependency torch-strumpack. Install a wheel built for your "
            "platform (cpu / cuda / rocm):\n\n"
            "    pip install torch-strumpack\n\n"
            "The CUDA / ROCm wheels carry a STRUMPACK build with GPU support; "
            "this is the portable (incl. AMD) direct-solver path that cuDSS "
            "(NVIDIA-only) does not cover."
        ) from exc
    _ts = torch_strumpack
    return _ts


def is_strumpack_available() -> bool:
    """``True`` iff torch-strumpack is importable AND its compiled STRUMPACK
    extension actually loads on this machine."""
    try:
        ts = _load_torch_strumpack()
    except ImportError:
        return False
    try:
        return bool(ts.is_available())
    except Exception:
        return False


def _coo_to_csr(val: Tensor, row: Tensor, col: Tensor,
                shape: Tuple[int, int]):
    """COO triple -> CSR ``(crow, col, values)`` via torch. Duplicate (i, j)
    entries are summed by ``coalesce`` (the factorization sees the assembled
    matrix); the autograd gradient is computed on the *original* COO row/col,
    so this reordering does not need to be tracked."""
    A = torch.sparse_coo_tensor(
        torch.stack([row, col], 0), val, shape).coalesce().to_sparse_csr()
    return A.crow_indices(), A.col_indices(), A.values()


# Thin pass-throughs to the autograd-free core primitives. The autograd
# Function in ``linear_solve`` owns differentiation and calls these.
def factor(crow: Tensor, col: Tensor, values: Tensor, n: int):
    return _load_torch_strumpack().factor(crow, col, values, n)


def solve(fac, b: Tensor) -> Tensor:
    return _load_torch_strumpack().solve(fac, b)


def solve_transpose(fac, b: Tensor) -> Tensor:
    return _load_torch_strumpack().solve_transpose(fac, b)
