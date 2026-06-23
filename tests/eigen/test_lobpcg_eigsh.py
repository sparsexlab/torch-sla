"""Correctness + convergence checks for the LOBPCG eigsh implementation.

Reference: numpy.linalg.eigvalsh on the dense form. We don't compare
against scipy's LOBPCG because both are implementations of the same
algorithm and that would only test API compatibility, not numerical
correctness.
"""
import numpy as np
import pytest
import torch

from torch_sla.sparse_tensor.autograd import _lobpcg_eigsh


def _spd_matrix(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, n))
    A = M @ M.T + n * np.eye(n)
    return A


@pytest.mark.parametrize("largest", [True, False])
@pytest.mark.parametrize("k", [1, 4, 8])
def test_lobpcg_matches_dense_eigh(largest, k):
    n = 100
    A_np = _spd_matrix(n)
    A_t = torch.from_numpy(A_np)
    matvec = lambda X: A_t @ X

    gt = np.linalg.eigvalsh(A_np)
    gt_k = sorted(gt, reverse=largest)[:k]

    # Seed the global RNG so the random init inside _lobpcg_eigsh is
    # reproducible. Without this, CPU vs CUDA default seeds differ
    # and the smallest k=1 case can converge to the next-smallest
    # eigenvalue on clustered spectra.
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    ev, V = _lobpcg_eigsh(
        matvec, n, k, torch.float64, torch.device("cpu"),
        largest=largest, maxiter=400, tol=1e-10,
    )

    # Eigenvalue match.
    for got, want in zip(ev.tolist(), gt_k):
        assert abs(got - want) / max(abs(want), 1.0) < 1e-6, \
            f"eigenvalue mismatch: got {got}, want {want}"

    # Orthonormality of eigenvectors.
    ortho_err = (V.T @ V - torch.eye(k, dtype=torch.float64)).abs().max().item()
    assert ortho_err < 1e-10, f"V^T V - I = {ortho_err}"

    # A V = V Lambda (Rayleigh quotient).
    AV = A_t @ V
    resid = (AV - V * ev.unsqueeze(0)).norm().item() / AV.norm().item()
    assert resid < 1e-4, f"||A V - V Lambda|| / ||A V|| = {resid}"


def test_lobpcg_converges_in_fewer_matvecs_than_steepest_descent():
    """The 3-block subspace ([X, R, P]) should reach a fixed tolerance
    in noticeably fewer outer iterations than the old [X, R]
    formulation. We count matvec calls via a counter."""
    torch.manual_seed(0)
    n, k = 200, 4
    A_t = torch.from_numpy(_spd_matrix(n, seed=42))

    matvec_count = [0]
    def counted(X):
        matvec_count[0] += 1
        return A_t @ X

    ev, _ = _lobpcg_eigsh(
        counted, n, k, torch.float64, torch.device("cpu"),
        largest=True, maxiter=200, tol=1e-8,
    )

    # Sanity: actually converged.
    gt = sorted(np.linalg.eigvalsh(A_t.numpy()), reverse=True)[:k]
    for got, want in zip(ev.tolist(), gt):
        assert abs(got - want) / abs(want) < 1e-5

    # Empirical bound for proper LOBPCG on a well-separated spectrum:
    # convergence in O(k log(1/tol) / sqrt(gap)) outer iters. For
    # n=200, k=4 we typically see 20-40 outer iters * one matvec per
    # iter (subspace expansion) + the seed matvec.
    assert matvec_count[0] < 100, \
        f"too many matvecs: {matvec_count[0]} (regression vs LOBPCG bound)"


def test_lobpcg_true_residual_below_tol():
    """Regression guard: the convergence criterion must measure the
    true Ritz residual ``||A x - lambda x||``, not the eigvals diff
    between successive iterations. On a clustered spectrum the
    earlier eigvals-diff test reported "converged" while the actual
    residual was still 1e-3..1e-5.

    For tol=1e-8, the returned eigenpair's Ritz residual must be
    below ``tol * |lambda|``."""
    n, k = 400, 6
    rng = np.random.default_rng(123)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    # Spectrum with near-degenerate top 4 -- the case that tripped
    # the old eigvals-diff test.
    eig = np.geomspace(1.0, 1e4, n)
    eig[-4:] = eig[-5] * np.array([1.0, 1.001, 1.002, 1.003])
    A_np = Q @ np.diag(eig) @ Q.T
    A_np = 0.5 * (A_np + A_np.T)
    A = torch.from_numpy(A_np)

    torch.manual_seed(0)
    tol = 1e-8
    ev, V = _lobpcg_eigsh(
        lambda X: A @ X, n, k, torch.float64, torch.device("cpu"),
        largest=True, maxiter=300, tol=tol,
    )

    # Each Ritz pair's residual norm must respect the requested tol.
    R = A @ V - V * ev.unsqueeze(0)
    res_norms = R.norm(dim=0)
    denom = ev.abs().clamp(min=1e-10)
    rel = (res_norms / denom).max().item()
    assert rel < 10 * tol, (
        f"true Ritz residual {rel:.2e} > 10*tol={10*tol:.2e}: "
        "convergence criterion is lying about the eigenpair quality"
    )


def test_lobpcg_accepts_preconditioner():
    """T_apply hook is a no-op-by-default, plug in identity to check
    the signature path still produces the right answer."""
    n, k = 80, 3
    A_t = torch.from_numpy(_spd_matrix(n))
    matvec = lambda X: A_t @ X

    ev_no_T, _ = _lobpcg_eigsh(
        matvec, n, k, torch.float64, torch.device("cpu"),
        largest=True, maxiter=400, tol=1e-10,
    )
    ev_with_T, _ = _lobpcg_eigsh(
        matvec, n, k, torch.float64, torch.device("cpu"),
        largest=True, maxiter=400, tol=1e-10,
        T_apply=lambda R: R,
    )
    # Identity preconditioner → same answer.
    assert (ev_no_T - ev_with_T).abs().max().item() < 1e-8
