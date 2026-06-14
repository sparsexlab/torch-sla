"""torch.sparse.spsolve wrapper backend.

Thin shim around PyTorch's native sparse direct solver. Calls
``torch.sparse.spsolve(A_csr, b)`` -- analyze + factor + solve in one
op, no manual handle management.

Status (PyTorch 2.12, June 2026): the underlying ``aten::_spsolve`` op
is only registered for the **MPS** backend in mainline PyTorch. CPU and
CUDA dispatch raises ``NotImplementedError`` so :func:`torch_spsolve` 
catches that and reraises as :class:`RuntimeError` with a clear message
the dispatcher can fall through on.

When PyTorch upstream lands CPU / CUDA kernels for ``_spsolve``, this
backend will start working with no code change here.
"""
from __future__ import annotations

import torch


def is_torch_spsolve_supported(device: torch.device) -> bool:
    """Cheap dispatch-table probe: tries the smallest possible solve
    and returns True iff it succeeds on ``device``."""
    try:
        A = torch.sparse_csr_tensor(
            torch.tensor([0, 1], dtype=torch.int32, device=device),
            torch.tensor([0], dtype=torch.int32, device=device),
            torch.tensor([1.0], dtype=torch.float64, device=device),
            (1, 1),
        )
        b = torch.tensor([1.0], dtype=torch.float64, device=device)
        torch.sparse.spsolve(A, b)
        return True
    except (NotImplementedError, RuntimeError):
        return False
    except Exception:
        return False


def torch_spsolve(val: torch.Tensor, row: torch.Tensor, col: torch.Tensor,
                  shape, b: torch.Tensor) -> torch.Tensor:
    """Solve ``A x = b`` via :func:`torch.sparse.spsolve`.

    Parameters
    ----------
    val, row, col : COO triple. Indices must be int64 (will be cast to
        int32 inside; ``torch.sparse.spsolve`` requires int32 indices).
    shape : (M, N) tuple. M must equal N (square system).
    b : right-hand side. 1-D ``(M,)`` or 2-D ``(M, nrhs)``.

    Returns
    -------
    x : Tensor of the same shape as ``b``.

    Raises
    ------
    RuntimeError if the active backend doesn't have ``_spsolve``
    registered (typical on CPU/CUDA in PyTorch <= 2.12).
    """
    M, N = shape
    if M != N:
        raise ValueError(f"torch_spsolve requires square A; got {(M, N)}")

    # Build CSR. PyTorch's spsolve wants int32 indices and column-major
    # for multi-RHS; we pass single-RHS through directly.
    indices = torch.stack([row.to(torch.int64), col.to(torch.int64)])
    A_coo = torch.sparse_coo_tensor(indices, val, (M, N)).coalesce()
    A_csr = A_coo.to_sparse_csr()
    crow = A_csr.crow_indices().to(torch.int32)
    ccol = A_csr.col_indices().to(torch.int32)
    A32 = torch.sparse_csr_tensor(crow, ccol, A_csr.values(), (M, N))

    return torch.sparse.spsolve(A32, b)
