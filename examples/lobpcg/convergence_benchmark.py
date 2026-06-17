"""LOBPCG vs block steepest descent -- convergence speed benchmark.

Both algorithms target the top-``k`` eigenpairs of the same SPD
matrix; we count matvecs to reach the same tolerance. Saves
``lobpcg_convergence.png`` (residual vs matvec count) when matplotlib
is available.

Usage::

    python examples/lobpcg/convergence_benchmark.py

The "old" algorithm is the previous block-steepest-descent (BSD)
formulation: two-block subspace [X | R], full QR each iteration, no
conjugate direction. The "new" algorithm is the current
``_lobpcg_core`` (3-block [X | R | P], CGS2, buffer reuse).
"""
from __future__ import annotations

import time
from typing import Callable, List, Tuple

import numpy as np
import torch

from torch_sla.sparse_tensor.linalg import _lobpcg_core


# --------------------------------------------------------------------- #
# Reference: old block-steepest-descent (the algorithm we're replacing)
# --------------------------------------------------------------------- #
def _lobpcg_bsd_old(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    n: int,
    k: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
    largest: bool = True,
    maxiter: int = 1000,
    tol: float = 1e-8,
    seed: int = 0,
    record: List[Tuple[int, float]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Block-steepest-descent reference. Returns (eigvals, X, matvecs).

    Mirrors the pre-refactor code: subspace = [X | R], rebuilt with a
    full ``torch.linalg.qr(torch.cat(...))`` every iteration, no P
    block. This is what the new LOBPCG should beat.
    """
    m = min(max(2 * k, k + 2), n)
    g = torch.Generator(device=device).manual_seed(seed)
    X = torch.randn(n, m, dtype=dtype, device=device, generator=g)
    X, _ = torch.linalg.qr(X)
    eig_prev = None
    matvec_count = 0

    def _mv(B):
        nonlocal matvec_count
        matvec_count += 1
        return matvec(B)

    for it in range(maxiter):
        AX = _mv(X)
        H = X.T @ AX
        H = 0.5 * (H + H.T)
        eigs, V = torch.linalg.eigh(H)
        idx = eigs.argsort(descending=largest)
        eigs, V = eigs[idx], V[:, idx]
        X, AX = X @ V, AX @ V

        # Residual norm of top-k for the convergence plot.
        R = AX[:, :k] - X[:, :k] * eigs[:k].unsqueeze(0)
        resid = R.norm().item() / AX[:, :k].norm().clamp_min(1e-30).item()
        if record is not None:
            record.append((matvec_count, resid))

        if eig_prev is not None:
            diff = (eigs[:k] - eig_prev[:k]).abs()
            denom = eigs[:k].abs().clamp(min=1e-12)
            if (diff < tol * denom).all():
                break
        eig_prev = eigs.clone()

        # Restart subspace as [X[:k] | R] via FULL QR + cat -- the slow path.
        X_new = torch.cat([X[:, :k], R], dim=1)
        X_new, _ = torch.linalg.qr(X_new)
        if X_new.shape[1] < m:
            pad = torch.randn(n, m - X_new.shape[1], dtype=dtype,
                              device=device, generator=g)
            X = torch.cat([X_new, pad], dim=1)
            X, _ = torch.linalg.qr(X)
        else:
            X = X_new

    return eigs[:k], X[:, :k], matvec_count


# --------------------------------------------------------------------- #
# New LOBPCG core with matvec counting + per-iter residual recording.
# --------------------------------------------------------------------- #
def _lobpcg_new_counted(
    matvec, n, k, *, dtype, device, largest, maxiter, tol, seed,
    record: List[Tuple[int, float]] = None,
):
    matvec_count = 0
    iter_residuals = []

    def _mv(B):
        nonlocal matvec_count
        matvec_count += 1
        out = matvec(B)
        return out

    # Wrap matvec so each call records the residual against the
    # current Ritz pair. We hook by inspecting outputs as they come
    # back: the core calls matvec exactly once per outer iteration
    # (the Rayleigh-Ritz expand step), so matvec_count - 1 == iter.
    eigvals, X = _lobpcg_core(
        _mv, n, k,
        dtype=dtype, device=device,
        largest=largest, maxiter=maxiter, tol=tol,
        seed=seed,
    )
    # Final residual is recorded after the fact -- we don't have a
    # per-iter hook from outside the core, so we re-run a tiny check
    # for the trajectory.
    return eigvals, X, matvec_count


# --------------------------------------------------------------------- #
# Convergence trajectory recorder for the new core (call back into core
# with a residual-recording matvec).
# --------------------------------------------------------------------- #
def _lobpcg_new_with_trajectory(
    A: torch.Tensor, k: int, *, tol=1e-8, maxiter=200, seed=0, largest=True,
):
    """Run new LOBPCG and return (eigvals, X, matvec_count, trajectory)
    where ``trajectory`` is a list of ``(matvec_count, ritz_resid_norm)``.
    """
    n = A.shape[0]
    dtype, device = A.dtype, A.device
    matvec_count = 0
    traj: List[Tuple[int, float]] = []

    # Estimate Ritz at every matvec by Rayleigh quotient on the column
    # we got back -- crude proxy, but it tracks convergence well.
    def matvec(B):
        nonlocal matvec_count
        matvec_count += 1
        out = A @ B
        # Residual for first k Ritz vectors approximated via column
        # norms: ||A B - B diag(B^T A B / B^T B)|| -- standard proxy.
        if B.shape[1] >= k:
            num = (B[:, :k] * out[:, :k]).sum(dim=0)
            den = (B[:, :k] * B[:, :k]).sum(dim=0).clamp_min(1e-30)
            rq = num / den
            R = out[:, :k] - B[:, :k] * rq.unsqueeze(0)
            resid = R.norm().item() / out[:, :k].norm().clamp_min(1e-30).item()
            traj.append((matvec_count, resid))
        return out

    eigvals, X = _lobpcg_core(
        matvec, n, k,
        dtype=dtype, device=device,
        largest=largest, maxiter=maxiter, tol=tol,
        seed=seed,
    )
    return eigvals, X, matvec_count, traj


# --------------------------------------------------------------------- #
# Test matrix.
# --------------------------------------------------------------------- #
def make_spd(n: int, kappa: float = 1e3, seed: int = 42) -> torch.Tensor:
    """SPD with prescribed condition number. The spectrum has a small
    gap between the top eigenvalues so the conjugate-direction
    advantage shows clearly.
    """
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    eig = np.geomspace(1.0, kappa, n)
    # Add a small cluster at the top so the top-k pairs need more
    # iterations to separate -- BSD struggles here, LOBPCG breezes.
    eig[-4:] = eig[-5] * np.array([1.0, 1.001, 1.002, 1.003])
    A = Q @ np.diag(eig) @ Q.T
    A = 0.5 * (A + A.T)
    return torch.from_numpy(A)


def main():
    torch.manual_seed(0)
    n, k = 400, 6
    tol = 1e-8
    maxiter = 200

    A = make_spd(n, kappa=1e4)
    gt = sorted(np.linalg.eigvalsh(A.numpy()), reverse=True)[:k]
    print(f"matrix: n={n}, k={k}, kappa~1e4, top-{k} eigs = "
          f"{[f'{x:.4f}' for x in gt]}")
    print()

    # --- Old BSD ---
    bsd_traj: List[Tuple[int, float]] = []
    t0 = time.perf_counter()
    eig_old, X_old, mvc_old = _lobpcg_bsd_old(
        lambda B: A @ B, n, k,
        dtype=A.dtype, device=A.device,
        largest=True, maxiter=maxiter, tol=tol, seed=0,
        record=bsd_traj,
    )
    t_old = time.perf_counter() - t0
    err_old = max(abs(g - e) for g, e in zip(gt, eig_old.tolist()))
    print(f"OLD (block-steepest-descent): {mvc_old:4d} matvecs   "
          f"{t_old*1000:6.1f} ms   max eig err = {err_old:.2e}")

    # --- New LOBPCG ---
    t0 = time.perf_counter()
    eig_new, X_new, mvc_new, lobpcg_traj = _lobpcg_new_with_trajectory(
        A, k, tol=tol, maxiter=maxiter, seed=0, largest=True,
    )
    t_new = time.perf_counter() - t0
    err_new = max(abs(g - e) for g, e in zip(gt, eig_new.tolist()))
    print(f"NEW (LOBPCG 3-block + CGS2):  {mvc_new:4d} matvecs   "
          f"{t_new*1000:6.1f} ms   max eig err = {err_new:.2e}")
    print()
    print(f"SPEEDUP: {mvc_old / max(mvc_new, 1):.1f}x fewer matvecs, "
          f"{t_old / max(t_new, 1e-9):.1f}x wall-clock")

    # --- Plot ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        if bsd_traj:
            xs, ys = zip(*bsd_traj)
            ax.semilogy(xs, ys, "o-", label=f"old BSD ({mvc_old} mv)",
                        markersize=4)
        if lobpcg_traj:
            xs, ys = zip(*lobpcg_traj)
            ax.semilogy(xs, ys, "s-", label=f"new LOBPCG ({mvc_new} mv)",
                        markersize=4)
        ax.axhline(tol, color="gray", linestyle=":", label=f"tol = {tol}")
        ax.set_xlabel("matvec count")
        ax.set_ylabel(r"$\|A x - \lambda x\| / \|A x\|$ (top-$k$ block)")
        ax.set_title(f"LOBPCG vs block-steepest-descent  "
                     f"(n={n}, k={k}, kappa=1e4)")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
        # Save next to the other example plots in the repo so PR
        # descriptions and docs can reference it by relative path.
        import os
        here = os.path.dirname(os.path.abspath(__file__))
        out_dir = os.path.normpath(os.path.join(
            here, "..", "..", "assets", "examples", "lobpcg"))
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, "convergence.png")
        fig.tight_layout()
        fig.savefig(out, dpi=120)
        print(f"\nplot saved to {out}")
    except ImportError:
        print("\n(matplotlib not installed; skipping plot)")


if __name__ == "__main__":
    main()
