"""Distributed LOBPCG for :class:`~torch_sla.distributed.DSparseTensor`.

Every rank holds the full ``N x m`` Ritz basis ``X`` (replicated); the
only distributed step per iteration is the column-wise matvec
(``scatter`` + ``D @ x_dt`` + ``full_tensor``). Rayleigh-Ritz on the
small ``m x m`` Gram matrix runs identically on every rank so the same
basis rotation lands everywhere.

Free function, not a method on DSparseTensor -- mirrors the
``*_shard`` Krylov routines in :mod:`distributed_solve`. The class
exposes :meth:`DSparseTensor.eigsh` as a thin wrapper.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

try:
    import torch.distributed as dist
    _DIST_AVAILABLE = True
except ImportError:
    _DIST_AVAILABLE = False


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
    m = min(max(2 * k, k + 4), N)

    gen_device = "cpu" if device.type == "mps" else device
    g = torch.Generator(device=gen_device).manual_seed(0)
    X = torch.randn(N, m, dtype=dtype, device=gen_device, generator=g).to(device)
    X, _ = torch.linalg.qr(X)

    def _batched_matvec(B):
        out = torch.empty_like(B)
        for j in range(B.shape[1]):
            out[:, j] = _column_matvec_global(D, B[:, j].contiguous())
        return out

    rank0 = not (_DIST_AVAILABLE and dist.is_initialized()) or dist.get_rank() == 0
    eig_prev: Optional[torch.Tensor] = None

    for it in range(maxiter):
        AX = _batched_matvec(X)
        H = X.T @ AX
        H = 0.5 * (H + H.T)
        eigs, V = torch.linalg.eigh(H)
        idx = eigs.argsort(descending=largest)
        eigs, V = eigs[idx], V[:, idx]
        X, AX = X @ V, AX @ V

        if eig_prev is not None:
            diff = (eigs[:k] - eig_prev[:k]).abs()
            ref = eigs[:k].abs().clamp(min=1e-12)
            if torch.all(diff < tol * ref):
                if verbose and rank0:
                    print(f"[eigsh] converged iter {it}")
                break
        eig_prev = eigs.clone()

        R = AX[:, :k] - X[:, :k] * eigs[:k].unsqueeze(0)
        X, _ = torch.linalg.qr(torch.cat([X[:, :k], R], dim=1))
        if X.size(1) < m:
            pad = torch.randn(N, m - X.size(1), dtype=dtype,
                              device=gen_device, generator=g).to(device)
            X, _ = torch.linalg.qr(torch.cat([X, pad], dim=1))

        if verbose and rank0 and it % 10 == 0:
            print(f"[eigsh] iter {it} top {eigs[:min(k, 4)].tolist()}")

    return eigs[:k], (X[:, :k] if return_eigenvectors else None)
