"""Distributed LOBPCG for :class:`~torch_sla.distributed.DSparseTensor`.

Every rank holds the full ``N x m`` Ritz basis ``X`` (replicated); the
only distributed step per iteration is the column-wise matvec
(``scatter`` + ``D @ x_dt`` + ``full_tensor``). Rayleigh-Ritz on the
small ``m x m`` Gram matrix runs identically on every rank so the same
basis rotation lands everywhere.

The 3-block subspace, CGS2 reorthogonalisation, and pre-allocated
buffers are reused from the single-device core in
:mod:`torch_sla.sparse_tensor.linalg._lobpcg_core` -- this wrapper
just supplies the column-wise distributed matvec.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

from ..sparse_tensor.linalg import _lobpcg_core


def _column_matvec_global(D, x_col: torch.Tensor) -> torch.Tensor:
    return (D @ D.scatter(x_col)).full_tensor()


def eigsh_shard(
    D,
    k: int = 6,
    which: str = "LM",
    maxiter: int = 200,
    tol: float = 1e-8,
    return_eigenvectors: bool = True,
    sigma: Optional[float] = None,
    verbose: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Distributed LOBPCG. ``which`` ∈ ``{LM, LA, SM, SA}``."""
    if not D.is_square:
        raise ValueError("eigsh requires a square matrix")
    if sigma is not None:
        raise NotImplementedError("sigma (shift-invert) not supported")

    N = int(D.shape[0])
    dtype, device = D.dtype, D.device
    largest = which in ("LM", "LA")

    def matvec(B: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(B)
        for j in range(B.shape[1]):
            out[:, j] = _column_matvec_global(D, B[:, j].contiguous())
        return out

    eigvals, X = _lobpcg_core(
        matvec, N, k,
        dtype=dtype, device=device,
        largest=largest, maxiter=maxiter, tol=tol,
        seed=0,
    )
    return eigvals, (X if return_eigenvectors else None)
