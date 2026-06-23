"""Regression test for the fp32 rank-deficiency crash in ``_lobpcg_core``.

In float32 the LOBPCG subspace ``[X | R | P]`` can become (near-)rank-
deficient -- e.g. the residual block R collapses into span(X) near
convergence, or clustered Ritz vectors in X turn collinear. The QR-based
``_qr_orthonormalize`` then drops the offending columns and returns
``ncols_eff < m`` (fewer columns than the internal block size). The old
code unconditionally wrote the resulting ``ncols_eff``-wide Ritz update
into the ``m``-wide ``X``/``AX``/``P``/``eigenvalues`` buffers and raised
``RuntimeError: The size of tensor a (..) must match the size of tensor b
(..)``. The bug was intermittent because it depends on the random initial
block (the ``seed=`` kwarg).

This test loops many seeds in float32 and asserts (a) no exception and
(b) a correct top eigenpair (small relative Ritz residual). It also pins
the failure deterministically by forcing ``_qr_orthonormalize`` to drop
columns below ``m`` -- this reproduces the exact crash on the old code
regardless of luck with the RNG. fp64 over the same seeds stays at full
~1e-9 accuracy.
"""
import torch

import torch_sla.sparse_tensor.linalg as L
from torch_sla.sparse_tensor.linalg import _lobpcg_core


def _poisson_2d(N: int, dtype):
    """Dense 2D 5-point Poisson on an ``N x N`` grid (n = N*N)."""
    n = N * N
    A = torch.zeros(n, n, dtype=dtype)

    def idx(i, j):
        return i * N + j

    for i in range(N):
        for j in range(N):
            p = idx(i, j)
            A[p, p] = 4.0
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ii, jj = i + di, j + dj
                if 0 <= ii < N and 0 <= jj < N:
                    A[p, idx(ii, jj)] = -1.0
    return A, n


def _top_residual(A, vals, vecs):
    worst = 0.0
    for i in range(vecs.shape[1]):
        v = vecs[:, i]
        lam = vals[i].item()
        r = (A @ v - lam * v).norm().item() / max(abs(lam), 1e-12)
        worst = max(worst, r)
    return worst


def test_lobpcg_fp32_seed_sweep_no_crash_and_correct():
    """fp32 must not raise over a sweep of seeds and must return correct
    top-k eigenpairs."""
    cpu = torch.device("cpu")
    A32, n = _poisson_2d(16, torch.float32)
    matvec = lambda Z: A32 @ Z  # noqa: E731

    worst = 0.0
    for seed in range(300):
        vals, vecs = _lobpcg_core(
            matvec, n=n, k=2, dtype=torch.float32, device=cpu,
            largest=True, seed=seed, tol=1e-8,
        )
        assert vals.shape == (2,)
        assert vecs.shape == (n, 2)
        worst = max(worst, _top_residual(A32, vals, vecs))
    # fp32 Ritz residual on a PDE operator caps well under 1e-3.
    assert worst < 1e-3, f"fp32 top eigenpair residual too large: {worst:.2e}"


def test_lobpcg_fp64_seed_sweep_tight():
    """fp64 stays at ~1e-9 accuracy over the same seeds (no regression)."""
    cpu = torch.device("cpu")
    A64, n = _poisson_2d(16, torch.float64)
    matvec = lambda Z: A64 @ Z  # noqa: E731

    worst = 0.0
    for seed in range(60):
        vals, vecs = _lobpcg_core(
            matvec, n=n, k=2, dtype=torch.float64, device=cpu,
            largest=True, seed=seed, tol=1e-8,
        )
        worst = max(worst, _top_residual(A64, vals, vecs))
    assert worst < 1e-7, f"fp64 top eigenpair residual regressed: {worst:.2e}"


def test_lobpcg_fp32_forced_rank_deficiency(monkeypatch):
    """Deterministically force ``ncols_eff < m`` (the exact crash path) by
    truncating the orthonormalised subspace, and assert the core stays
    robust and accurate.

    This is the deterministic version of the intermittent fp32 bug: on the
    unpatched code it raised ``RuntimeError`` at ``X.copy_(X_new)``.
    """
    cpu = torch.device("cpu")
    n = 256
    d = torch.linspace(1.0, 5.0, n, dtype=torch.float32)
    matvec = lambda Z: d.unsqueeze(1) * Z  # noqa: E731

    orig = L._qr_orthonormalize
    calls = {"i": 0}

    def truncating_qr(Z):
        Q = orig(Z)
        calls["i"] += 1
        # Force fewer columns than the block size m on two in-loop
        # orthonormalisations of the [X|R|P] subspace (width >= 8 here).
        if Z.shape[1] >= 8 and calls["i"] in (2, 3):
            Q = Q[:, :2]
        return Q

    # _lobpcg_core resolves the symbol via module globals (aliased as
    # _cgs2_inplace); patch both names to be safe.
    monkeypatch.setitem(_lobpcg_core.__globals__, "_cgs2_inplace", truncating_qr)
    monkeypatch.setitem(_lobpcg_core.__globals__, "_qr_orthonormalize", truncating_qr)

    vals, vecs = _lobpcg_core(
        matvec, n=n, k=2, dtype=torch.float32, device=cpu,
        largest=True, seed=0, tol=1e-8,
    )
    assert vals.shape == (2,)
    assert vecs.shape == (n, 2)
    # Diagonal operator: exact top eigenvalues are 5.0 and the next entry.
    A = torch.diag(d)
    worst = _top_residual(A, vals, vecs)
    assert worst < 1e-3, f"residual after forced rank-deficiency: {worst:.2e}"
    # Top eigenvalue must be recovered.
    assert abs(vals[0].item() - 5.0) < 1e-3
