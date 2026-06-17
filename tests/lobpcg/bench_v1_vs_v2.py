"""Before / after the PR #45 fix series: speed AND precision.

This PR fixes two bugs in ``_lobpcg_core`` that were independently
masking each other:

* Convergence test: eigvals-diff → Ritz-residual-norm. The old test
  trips early on clustered spectra and returns eigenpairs with
  residual ~1e-5 for ``tol=1e-8``.
* Re-orthonormalisation: Python-loop CGS2 → ``torch.linalg.qr``.
  Profiling showed the loop took ~80% of per-iter time at typical
  block sizes; one LAPACK call is identical in stability and 5-10x
  faster on CPU.

We bench three variants against ``torch.lobpcg`` on a banded SPD:

  v1   = pre-fix (eigvals-diff convergence + Python-loop CGS2)
  v1.5 = convergence fix only
  v2   = convergence fix + QR swap (this PR)
  ref  = torch.lobpcg(sparse_coo)

Prints a tabular summary. The headline plot lives in
``tests/lobpcg/assets/comparison_all.png``, produced by
``lobpcg_fix_comparison_multi_device.py`` (multi-device sweep) +
``lobpcg_fix_comparison_merge.py`` (merge per-device JSON dumps).
"""
from __future__ import annotations

import sys
import time
from typing import Callable, Optional

import numpy as np
import torch


def make_sparse_spd(n: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    A = np.zeros((n, n))
    for i in range(n):
        A[i, i] = 4.0 + rng.uniform(0, 0.1)
        for offset in [1, 2, 5]:
            if i + offset < n:
                v = -rng.uniform(0.1, 1.0)
                A[i, i + offset] = v
                A[i + offset, i] = v
    A = 0.5 * (A + A.T) + n * np.eye(n) * 0.5
    indices = np.array(A.nonzero())
    values = A[indices[0], indices[1]]
    A_coo = torch.sparse_coo_tensor(
        torch.from_numpy(indices).long(),
        torch.from_numpy(values),
        size=(n, n),
    ).coalesce()
    return A_coo, torch.from_numpy(A)


# --------------------------------------------------------------------- #
# v1: pre-fix (eigvals-diff convergence + Python-loop CGS2)
# --------------------------------------------------------------------- #
def _cgs2_python_loop(Z: torch.Tensor) -> torch.Tensor:
    eps = torch.finfo(Z.dtype).eps * 100
    for _ in range(2):
        for j in range(Z.shape[1]):
            if j > 0:
                coeff = Z[:, :j].T @ Z[:, j]
                Z[:, j] -= Z[:, :j] @ coeff
            nrm = Z[:, j].norm()
            if nrm > eps:
                Z[:, j] /= nrm
            else:
                Z[:, j].zero_()
    col_norms = Z.norm(dim=0)
    valid = col_norms > 0.5
    if not bool(valid.all()):
        Z = Z[:, valid]
    return Z


def _qr_orthonormalize(Z: torch.Tensor) -> torch.Tensor:
    # MPS torch.linalg.qr on tall-skinny (n, m) blocks scales as
    # O(n^2) -- 62 ms at n=2000, m=36. CPU round-trip is 0.9 ms,
    # ~70x faster. Same pattern as _eigh_with_mps_fallback.
    if Z.device.type == "mps":
        Q_cpu, R_cpu = torch.linalg.qr(Z.cpu())
        diag = R_cpu.diagonal().abs()
        eps = torch.finfo(Z.dtype).eps * 100 * diag.max().clamp(min=1)
        keep = diag > eps
        if not bool(keep.all()):
            Q_cpu = Q_cpu[:, keep]
        return Q_cpu.to(Z.device)
    Q, R = torch.linalg.qr(Z)
    diag = R.diagonal().abs()
    eps = torch.finfo(Z.dtype).eps * 100 * diag.max().clamp(min=1)
    keep = diag > eps
    if not bool(keep.all()):
        Q = Q[:, keep]
    return Q


def _eigh_with_mps_fallback(H: torch.Tensor, device: torch.device):
    """``aten::_linalg_eigh.eigenvalues`` isn't implemented on MPS in
    current PyTorch. H is tiny here (3m x 3m <= 36 x 36), so the
    round-trip to CPU is essentially free. Lets us bench LOBPCG on
    Apple Silicon Metal until upstream fills the gap."""
    if device.type == "mps":
        eigvals, V = torch.linalg.eigh(H.cpu())
        return eigvals.to(device), V.to(device)
    return torch.linalg.eigh(H)


def lobpcg_param(
    matvec: Callable, n: int, k: int, *, dtype, device,
    largest: bool = True, maxiter: int = 300, tol: float = 1e-8,
    convergence: str = "residual",       # "residual" or "eigvals_diff"
    orthonormalize: str = "qr",          # "qr" or "cgs2_loop"
    seed: int = 0,
):
    """Parametric LOBPCG. ``convergence`` and ``orthonormalize`` switch
    between the buggy and fixed variants so we can A/B them on
    identical inputs."""
    ortho = _qr_orthonormalize if orthonormalize == "qr" else _cgs2_python_loop
    m = min(max(2 * k, k + 2), n)

    g = torch.Generator(device=device).manual_seed(seed)
    X = torch.randn(n, m, dtype=dtype, device=device, generator=g)
    X, _ = torch.linalg.qr(X)
    AX = torch.empty_like(X)
    R = torch.empty_like(X)
    P = torch.zeros_like(X)
    Z = torch.empty(n, 3 * m, dtype=dtype, device=device)
    eigenvalues = torch.empty(m, dtype=dtype, device=device)
    eigenvalues_prev: Optional[torch.Tensor] = None

    AX.copy_(matvec(X))
    H = X.T @ AX
    H = 0.5 * (H + H.T)
    eigvals, V = _eigh_with_mps_fallback(H, device)
    idx = eigvals.argsort(descending=largest)
    eigvals, V = eigvals[idx], V[:, idx]
    X.copy_(X @ V); AX.copy_(AX @ V)
    eigenvalues.copy_(eigvals[:m])

    for iteration in range(maxiter):
        torch.mul(X, eigenvalues.unsqueeze(0), out=R); R.neg_(); R.add_(AX)
        if convergence == "residual":
            res_norms = R[:, :k].norm(dim=0)
            denom = eigenvalues[:k].abs().clamp(min=1e-10)
            if (res_norms < tol * denom).all():
                break

        ncols = 2 * m if iteration == 0 else 3 * m
        Z[:, :m].copy_(X); Z[:, m:2 * m].copy_(R)
        if iteration > 0:
            Z[:, 2 * m:3 * m].copy_(P)

        Z_active = ortho(Z[:, :ncols])
        ncols_eff = Z_active.shape[1]

        AZ_active = matvec(Z_active)
        H = Z_active.T @ AZ_active
        H = 0.5 * (H + H.T)
        eigvals, V = _eigh_with_mps_fallback(H, device)
        idx = eigvals.argsort(descending=largest)
        eigvals, V = eigvals[idx], V[:, idx]
        Vk = V[:, :m]
        X.copy_(Z_active @ Vk); AX.copy_(AZ_active @ Vk)
        if ncols_eff > m:
            P.copy_(Z_active[:, m:] @ Vk[m:, :])
        else:
            P.zero_()
        new_eigvals = eigvals[:m]

        if convergence == "eigvals_diff":
            if eigenvalues_prev is not None:
                diff = (new_eigvals[:k] - eigenvalues_prev[:k]).abs()
                denom = new_eigvals[:k].abs().clamp(min=1e-10)
                if (diff < tol * denom).all():
                    eigenvalues.copy_(new_eigvals); break
            if eigenvalues_prev is None:
                eigenvalues_prev = new_eigvals.clone()
            else:
                eigenvalues_prev.copy_(new_eigvals)
        eigenvalues.copy_(new_eigvals)

    return eigenvalues[:k], X[:, :k]


def time_call(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return time.perf_counter() - t0, out


def main():
    sizes = [200, 400, 700, 1000, 1500, 2000]
    k = 6
    variants = {
        "v1 (pre-fix)":           dict(convergence="eigvals_diff", orthonormalize="cgs2_loop"),
        "v1.5 (conv. fix only)":  dict(convergence="residual",      orthonormalize="cgs2_loop"),
        "v2 (this PR)":           dict(convergence="residual",      orthonormalize="qr"),
    }

    times = {label: [] for label in variants}
    errs = {label: [] for label in variants}
    times["torch.lobpcg"] = []
    errs["torch.lobpcg"] = []

    print(f"{'n':>5s}  {'variant':22s}  {'time_ms':>9s} {'max_err':>10s}")
    for n in sizes:
        A_coo, A_dense = make_sparse_spd(n)
        gt = sorted(np.linalg.eigvalsh(A_dense.numpy()), reverse=True)[:k]

        # torch.lobpcg baseline
        torch.manual_seed(0)
        t, (vals_t, _) = time_call(torch.lobpcg, A_coo, k=k, largest=True,
                                    niter=300, tol=1e-8)
        err = max(abs(g - e) for g, e in zip(gt, vals_t.tolist()))
        times["torch.lobpcg"].append(t * 1000)
        errs["torch.lobpcg"].append(err)
        print(f"{n:>5d}  {'torch.lobpcg':22s}  {t*1000:>9.2f} {err:>10.2e}")

        for label, opts in variants.items():
            torch.manual_seed(0)
            t, (vals, _) = time_call(
                lobpcg_param,
                lambda B, _A=A_coo: torch.sparse.mm(_A, B),
                n, k,
                dtype=A_coo.dtype, device=A_coo.device,
                largest=True, maxiter=300, tol=1e-8, seed=0,
                **opts,
            )
            err = max(abs(g - e) for g, e in zip(gt, vals.tolist()))
            times[label].append(t * 1000)
            errs[label].append(err)
            print(f"{n:>5d}  {label:22s}  {t*1000:>9.2f} {err:>10.2e}")
        print()

    print("\nFor the headline CPU + CUDA combined plot, run:")
    print("  python tests/lobpcg/bench_multi_device.py   # collect")
    print("  python tests/lobpcg/merge_plots.py          # render")
    print("Plot lands at tests/lobpcg/assets/comparison_all.png")


if __name__ == "__main__":
    main()
