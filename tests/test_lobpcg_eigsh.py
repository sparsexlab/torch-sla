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
