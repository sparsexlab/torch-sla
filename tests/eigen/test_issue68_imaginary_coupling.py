"""Regression for issue #68: complex Hermitian eigsh backward sign bug.

The imaginary off-diagonal coupling ``A(t) = [[2, i*t], [-i*t, 3]]`` has a
real top eigenvalue whose derivative w.r.t. ``t`` is ``2*t/sqrt(1+4*t^2)``.
The forward eigenvalue was correct, but the backward returned the negated
derivative because it used ``conj(v[row]) * v[col]`` (the analytic Wirtinger
coefficient) instead of PyTorch's VJP convention ``v[row] * conj(v[col])``.

The second test generalizes the first: a random Hermitian perturbation of a
well-separated spectrum, checked against dense ``torch.linalg.eigvalsh``
autograd and a finite-difference reference.
"""
from __future__ import annotations

import pytest
import torch

from torch_sla import SparseTensor


def sparse_evals(a, k, which):
    n = len(a)
    row = torch.arange(n).repeat_interleave(n)
    col = torch.arange(n).repeat(n)
    return SparseTensor(a.flatten(), row, col, (n, n)).eigsh(k=k, which=which)[0]


@pytest.mark.parametrize("t0", [-0.7, 0.2, 0.7])
def test_imaginary_coupling_analytic(t0):
    t = torch.tensor(t0, dtype=torch.float64, requires_grad=True)
    z = t * 0
    a = torch.stack([torch.stack([z + 2, 1j * t]), torch.stack([-1j * t, z + 3])])
    y = sparse_evals(a, 1, "LA").sum()
    g, = torch.autograd.grad(y, t)
    expected = 2 * t0 / (1 + 4 * t0 * t0) ** 0.5
    torch.testing.assert_close(g, torch.tensor(expected, dtype=torch.float64), rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
@pytest.mark.parametrize("seed", [21, 42])
@pytest.mark.parametrize("k", [1, 2])
@pytest.mark.parametrize("which", ["LA", "SA"])
def test_hermitian_parameter_dense_and_finite_difference(dtype, seed, k, which):
    gen = torch.Generator().manual_seed(seed)
    m = torch.randn(5, 5, dtype=dtype, generator=gen)
    a0 = (m + m.mH) / 2 + torch.diag(torch.arange(1.0, 6.0, dtype=torch.float64) * 3)
    h = torch.randn(5, 5, dtype=dtype, generator=gen)
    h = (h + h.mH) / 2
    weights = torch.arange(1.0, k + 1.0, dtype=torch.float64)

    def objective(t, sparse):
        a = a0 + t * h
        if sparse:
            ev = sparse_evals(a, k, which)
        else:
            ev = torch.linalg.eigvalsh(a)
            ev = ev[-k:].flip(0) if which == "LA" else ev[:k]
        return (ev.sort().values * weights).sum()

    t = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    ys, yd = objective(t, True), objective(t, False)
    gs, = torch.autograd.grad(ys, t)
    gd, = torch.autograd.grad(yd, t)
    eps = 1e-5
    fd = (objective(t.detach() + eps, True) - objective(t.detach() - eps, True)) / (2 * eps)
    torch.testing.assert_close(ys, yd, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(gd, fd, rtol=1e-7, atol=1e-8)
    torch.testing.assert_close(gs, gd, rtol=1e-9, atol=1e-9)
