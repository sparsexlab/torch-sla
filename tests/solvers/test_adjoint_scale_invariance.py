"""The adjoint solve must not care how large the incoming gradient is.

Backward solves ``A^H lambda = dL/dx``, which is linear in its right-hand side,
so ``grad(c * g)`` has to equal ``c * grad(g)`` for every ``c``. Iterative
backends stop on ``max(atol, rtol * ||b||)`` and the absolute floor does not
scale, so before this was fixed a shrinking ``dL/dx`` -- that is, a converging
optimisation -- silently drove the gradient to zero. Measured on this problem
before the fix:

======================  =========  =========  =========  =========
backend+method            c=1e+00    c=1e-04    c=1e-08    c=1e-12
======================  =========  =========  =========  =========
scipy+lu                  5.5e-16    8.1e-16    5.7e-16    3.9e-16
cudss+cholesky            2.4e-15    2.3e-15    2.3e-15    2.4e-15
pyamg+ruge_stuben         9.5e-11    9.5e-11    9.5e-11    9.5e-11
scipy+cg                  1.2e-05    1.2e-05    6.1e-04    1.0e+00
pytorch+cg                5.8e-11    2.4e-08    6.1e-04    1.0e+00
amgx+pcg                  1.9e-12    4.4e-08    1.2e-03    1.0e+00
======================  =========  =========  =========  =========

A relative error of 1.0 means the gradient came back identically zero -- finite,
correctly shaped and wrong. Note the ``c=1e-08`` column: the failure is not a
cliff at the tolerance, it degrades a decade or more earlier while the result
still looks entirely healthy, which is why an "assert the gradient is nonzero"
check would not have caught it.

These tests assert the linearity directly rather than comparing against a
reference solver, so they keep their meaning if a backend's stopping rule
changes, and they cover both the soft edge and the cliff.
"""

import numpy as np
import pytest
import scipy.sparse as sp
import torch
import torch.multiprocessing as mp

from torch_sla import SparseTensor
from torch_sla.backends import (
    is_amgx_available,
    is_cudss_available,
    is_pyamg_available,
)

# c = 1 is the reference; each of these must reproduce it after rescaling.
SCALES = (1e-4, 1e-8, 1e-12)
# Loose enough for the iterative backends' own convergence noise (~1e-8 here),
# tight enough to catch the 6e-4 degradation and the 1.0 collapse above.
LINEARITY_RTOL = 1e-6

GRID = 32  # 2-D Poisson, 1024 DOF -- above the size where AMG degenerates to a
           # direct solve (at 225 DOF kappa(PA) is 1.0000017, so P would be an
           # exact inverse and the test would not exercise a real V-cycle)


def _poisson_2d_coo(device, dtype=torch.float64):
    ident = sp.identity(GRID, format="csr")
    stencil = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(GRID, GRID), format="csr")
    A = (sp.kron(ident, stencil) + sp.kron(stencil, ident)).tocoo()
    n = A.shape[0]
    val = torch.tensor(A.data, dtype=dtype, device=device, requires_grad=True)
    row = torch.tensor(A.row, dtype=torch.int64, device=device)
    col = torch.tensor(A.col, dtype=torch.int64, device=device)
    return val, row, col, n


def _grad_at_scale(backend, method, device, gscale, seed=0):
    """d(loss)/d(val) with the incoming gradient scaled by ``gscale``."""
    val, row, col, n = _poisson_2d_coo(device)
    A = SparseTensor(val, row, col, (n, n))
    b = torch.ones(n, dtype=torch.float64, device=device)
    x = A.solve(b, backend=backend, method=method)
    g = torch.from_numpy(
        np.random.default_rng(seed).standard_normal(n) * gscale
    ).to(device=device, dtype=torch.float64)
    (grad,) = torch.autograd.grad(x, val, grad_outputs=g)
    return grad.detach().cpu().numpy()


def _assert_scale_invariant(backend, method, device):
    reference = _grad_at_scale(backend, method, device, 1.0)
    denom = np.linalg.norm(reference)
    assert denom > 0, "reference gradient is zero; the test problem is degenerate"
    for c in SCALES:
        scaled = _grad_at_scale(backend, method, device, c)
        rel = np.linalg.norm(scaled - reference * c) / (denom * c)
        assert rel < LINEARITY_RTOL, (
            f"{backend}+{method} on {device}: adjoint is not scale invariant. "
            f"grad(c*g) / c differs from grad(g) by {rel:.2e} at c={c:.0e} "
            f"(all-zero={not scaled.any()}). The adjoint right-hand side is a "
            f"gradient, so an absolute stopping floor must not reach it."
        )


_CPU_CASES = [("scipy", "lu"), ("scipy", "cg"), ("pytorch", "cg")]


@pytest.mark.parametrize("backend,method", _CPU_CASES)
def test_adjoint_scale_invariant_cpu(backend, method):
    _assert_scale_invariant(backend, method, "cpu")


@pytest.mark.skipif(not is_pyamg_available(), reason="PyAMG not installed")
def test_adjoint_scale_invariant_pyamg():
    _assert_scale_invariant("pyamg", "ruge_stuben", "cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_adjoint_scale_invariant_pytorch_cuda():
    _assert_scale_invariant("pytorch", "cg", "cuda")


@pytest.mark.skipif(not is_cudss_available(), reason="cuDSS not available")
def test_adjoint_scale_invariant_cudss():
    _assert_scale_invariant("cudss", "cholesky", "cuda")


# ---------------------------------------------------------------------------
# AmgX runs in its own process.
#
# torch-amgx aborts the interpreter (SIGABRT, "trying to free non-empty
# temporary device pool") as soon as a second AmgX solver is destroyed while
# another is alive -- which SOLVER_CACHE guarantees, since a solve and its
# adjoint each build one. In-process that abort takes the whole pytest session
# down mid-run, so the measurement is made in a spawned child that reports its
# numbers back through a queue and then leaves without interpreter cleanup.
# Remove this indirection once the lifecycle bug is fixed upstream.
# ---------------------------------------------------------------------------
def _amgx_child(out_queue):
    import os
    import sys

    try:
        reference = _grad_at_scale("amgx", "pcg", "cuda", 1.0)
        denom = float(np.linalg.norm(reference))
        rels = {}
        for c in SCALES:
            scaled = _grad_at_scale("amgx", "pcg", "cuda", c)
            rels[c] = float(np.linalg.norm(scaled - reference * c) / (denom * c))
        out_queue.put(("ok", denom, rels))
    except BaseException as exc:  # noqa: BLE001 -- reported to the parent verbatim
        out_queue.put(("err", 0.0, f"{type(exc).__name__}: {exc}"))
    finally:
        # Flush the queue's feeder thread before skipping cleanup, or the parent
        # sees an empty queue and times out.
        out_queue.close()
        out_queue.join_thread()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


@pytest.mark.skipif(not is_amgx_available(), reason="AmgX (torch-amgx) not available")
def test_adjoint_scale_invariant_amgx():
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_amgx_child, args=(queue,))
    proc.start()
    try:
        status, denom, payload = queue.get(timeout=300)
    finally:
        proc.join(timeout=30)
        if proc.is_alive():
            proc.terminate()

    assert status == "ok", f"AmgX child failed: {payload}"
    assert denom > 0, "reference gradient is zero; the test problem is degenerate"
    for c, rel in payload.items():
        assert rel < LINEARITY_RTOL, (
            f"amgx+pcg: adjoint is not scale invariant. grad(c*g) / c differs "
            f"from grad(g) by {rel:.2e} at c={c:.0e}."
        )
