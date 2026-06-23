"""Regression tests for complex-valued backward of det() and eigsh().

History: complex `det` backward was silently wrong (missing a conjugate on
the geometric factor `det * A^{-T}` in the adjoint), giving forward-correct
but gradient-wrong results. `eigsh` backward was fine all along -- a naive
`gradcheck` on the raw COO of a *symmetric* matrix gives false failures
because perturbing a single off-diagonal entry breaks the symmetry the
solver assumes. These tests compare against closed-form / dense-autograd
references instead of `gradcheck`.

CPU only; the CUDA backward paths are verified separately on a GPU box.
"""
from __future__ import annotations

import pytest
import torch

from torch_sla import SparseTensor

torch.manual_seed(0)

# Full coverage matrix: every test runs on cpu and (when present) the GPU --
# CUDA on NVIDIA, or ROCm/HIP which also presents as "cuda". det's CUDA backward
# uses cuDSS when available and falls back to a dense inverse otherwise (so ROCm
# exercises the fallback, a real-GPU NVIDIA box exercises the cuDSS path).
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _coo(A):
    idx = A.nonzero(as_tuple=False).t().contiguous()
    return A[idx[0], idx[1]], idx[0], idx[1], A.shape


def _hpd(n, dtype, device):
    M = torch.randn(n, n, dtype=dtype, device=device)
    return M @ M.conj().t() + n * torch.eye(n, dtype=dtype, device=device)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
def test_det_backward_matches_dense_autograd(dtype, device):
    """Sparse det() VJP must equal torch.linalg.det's autograd, incl. a
    non-trivial complex upstream grad (L = Re(det(A) * w))."""
    torch.set_default_dtype(torch.float64)
    n = 6
    A0 = _hpd(n, dtype, device)
    w = torch.randn((), dtype=dtype, device=device)
    v, r, c, s = _coo(A0)

    vs = v.clone().requires_grad_(True)
    (SparseTensor(vs, r, c, s).det() * w).real.backward()

    Ad = A0.clone().requires_grad_(True)
    (torch.linalg.det(Ad) * w).real.backward()
    gd = torch.stack([Ad.grad[r[i], c[i]] for i in range(len(r))])

    rel = (vs.grad - gd).abs().max() / gd.abs().max()
    assert rel < 1e-9, f"det backward mismatch ({dtype}, {device}): rel={rel:.2e}"


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
def test_eigsh_backward_matches_analytic(dtype, device):
    """d(lambda_max)/dA = v v^H for a simple top eigenvalue; the sparse
    eigsh VJP gathered at A's nnz must match this closed form."""
    torch.set_default_dtype(torch.float64)
    # Regression guard: complex eigsh (LOBPCG) once failed to converge on GPU
    # (CUDA + ROCm) because the Rayleigh-Ritz Gram matrix used plain transpose
    # X.T@AX (non-Hermitian for complex) -> torch.linalg.eigh undefined on
    # non-Hermitian input, LAPACK/cuSOLVER disagreed. Fixed by conjugate
    # transpose X.mH@AX in _lobpcg_core. This case must now pass on GPU too.
    n = 6
    # well-separated spectrum so the top eigenpair is simple/stable
    diag = torch.diag(torch.tensor([10., 8., 6., 4., 2., 1.], dtype=dtype, device=device))
    A0 = diag + 0.05 * _hpd(n, dtype, device)
    A0 = (A0 + A0.conj().t()) / 2
    v, r, c, s = _coo(A0)

    vs = v.clone().requires_grad_(True)
    w, _ = SparseTensor(vs, r, c, s).eigsh(k=1, which="LM")
    w.real.sum().backward()

    # Per-entry analytic for a simple eigenvalue: d(lambda)/dA_ij = conj(v_i) v_j
    # (global-phase invariant, unlike v_i v_j). No-op conj on real symmetric.
    wd, Vd = torch.linalg.eigh(A0)
    vec = Vd[:, -1]
    expected = torch.stack([vec[r[i]].conj() * vec[c[i]] for i in range(len(r))])

    rel = (vs.grad - expected).abs().max() / expected.abs().max()
    assert rel < 1e-5, f"eigsh backward mismatch ({dtype}, {device}): rel={rel:.2e}"
