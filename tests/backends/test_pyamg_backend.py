"""Tests for the PyAMG-hybrid backend (CPU AMG setup + torch.sparse V-cycle).

Runs wherever ``pip install pyamg`` works -- Windows, Linux, macOS.
Cross-platform AMG is the entire point of this backend, so the test
suite *is* the cross-platform claim.

Every test driven by a matrix uses the existing benchmark catalogue
(:data:`torch_sla.datasets`) -- no hand-rolled Poisson/anisotropic
generators inside the test file. ``Benchmark.evaluate`` does the
round-trip vs the catalogued ``x_ref``.
"""
from __future__ import annotations

import pytest
import torch

from torch_sla import SparseTensor, solve, spsolve
from torch_sla.backends import is_pyamg_available
from torch_sla.datasets import Synthetic


pytestmark = pytest.mark.skipif(
    not is_pyamg_available(),
    reason="pyamg is an optional dependency; install with `pip install pyamg`."
)


def _amg_solver(val, row, col, shape, rhs):
    """Convenience solver callable passed to :meth:`Benchmark.evaluate`."""
    return spsolve(val, row, col, shape, rhs,
                   backend="pyamg", atol=1e-7, maxiter=50)


# =====================================================================
# Convergence sweep over the catalogue
# =====================================================================
# Catalogued real-symmetric matrices that classical Ruge-Stuben AMG is
# *not* a good match for, mostly because of pathological conditioning
# rather than a structural issue. The catalogue keeps them around for
# direct-solver tests; AMG is intentionally not asked to solve them here.
_PATHOLOGICAL_FOR_AMG = {
    "HB/bcsstk16",   # structural stiffness, kappa ~ 1e9; needs SA + null-space hint
}


def test_pyamg_converges_on_real_benchmarks(benchmark_real):
    """Standalone AMG converges on every catalogued real-dtype benchmark
    that classical Ruge-Stuben is suited for (i.e. symmetric, not too
    large for a unit-test timeout, not pathologically ill-conditioned)."""
    b = benchmark_real
    if b.math_kind == "general":
        pytest.skip(
            f"{b.name}: classical AMG (Ruge-Stuben) needs symmetric; "
            f"convection-diffusion and other general matrices are out of scope"
        )
    if b.shape[0] > 50_000:
        pytest.skip(
            f"{b.name}: {b.shape[0]} dof; skipped from the unit-test sweep "
            f"(setup time on this scale belongs in the perf benchmarks)"
        )
    if b.name in _PATHOLOGICAL_FOR_AMG:
        pytest.skip(
            f"{b.name}: known AMG-hard (condition number ~1e9); kept in the "
            f"catalogue for direct-solver tests, not for classical RS-AMG"
        )

    errs = b.evaluate(_amg_solver, metric="rel_l2")
    assert max(errs) < 1e-4, f"{b.name}: rel L2 errs = {errs}"


def test_pyamg_handles_anisotropic_diffusion():
    """The eps=0.01 anisotropic 2D Laplacian is AMG's home turf -- the
    classical strength-of-connection measure was *designed* for this
    kind of M-matrix with strong/weak couplings in different directions.

    Convergence to ~1e-4 in 50 V-cycles is the realistic target;
    machine precision needs hundreds of cycles or a finer-tuned cycle
    type (F or W), out of scope for a smoke test."""
    b = Synthetic["anisotropic_2d_64_eps_001"]
    errs = b.evaluate(_amg_solver, metric="rel_l2")
    assert max(errs) < 5e-4, f"anisotropic: rel L2 errs = {errs}"


# =====================================================================
# Integration with the new ``torch_sla.solve`` API
# =====================================================================
def test_pyamg_via_new_solve_api(benchmark_small_real):
    """The kwargs+dataclass ``solve()`` entry point also routes to pyamg
    and returns a populated :class:`SolveInfo` when asked."""
    b = benchmark_small_real
    rhs = b[0]["b"]
    x_ref = b[0]["x"]

    A = SparseTensor(b.val, b.row, b.col, b.shape)
    x, info = solve(A, rhs, backend="pyamg", atol=1e-8, maxiter=30,
                    return_info=True)
    assert info.converged
    rel_err = ((x - x_ref).norm() / x_ref.norm()).item()
    assert rel_err < 1e-6, f"{b.name}: rel_err = {rel_err}"


def test_pyamg_smoothed_aggregation_method(benchmark_small_real):
    """``method='sa'`` selects the smoothed-aggregation coarsening
    variant. SA without a user-supplied near-null-space converges more
    slowly than Ruge-Stuben on a 5-point Laplacian (where RS's
    strength-of-connection matches the M-matrix structure exactly); we
    only assert SA makes *real progress* (~100x residual reduction)."""
    b = benchmark_small_real
    rhs = b[0]["b"]
    x_ref = b[0]["x"]

    x = spsolve(b.val, b.row, b.col, b.shape, rhs,
                backend="pyamg", method="sa", atol=1e-8, maxiter=30)
    rel_err = ((x - x_ref).norm() / x_ref.norm()).item()
    assert rel_err < 1e-2, f"SA-AMG did not make progress: rel_err={rel_err}"


# =====================================================================
# Gradient (Wirtinger adjoint through AMG)
# =====================================================================
def test_pyamg_backward_produces_finite_gradient(benchmark_small_real):
    """Backward pass via the adjoint solve produces a finite, non-trivial
    gradient on ``val``. For real symmetric Poisson the adjoint is the
    same AMG hierarchy solved on the conjugate transpose, which is
    structurally identical for real symmetric problems."""
    b = benchmark_small_real
    rhs = b[0]["b"]
    val = b.val.clone().requires_grad_(True)

    x = spsolve(val, b.row, b.col, b.shape, rhs,
                backend="pyamg", atol=1e-8, maxiter=20)
    loss = (x ** 2).sum()
    loss.backward()

    assert val.grad is not None
    assert torch.isfinite(val.grad).all().item()
    assert val.grad.abs().max().item() > 1e-6


# =====================================================================
# Hierarchy-level API
# =====================================================================
def test_pyamg_hierarchy_reused_for_multiple_rhs(benchmark_small_real):
    """Build a hierarchy once, use it as a callable preconditioner for
    multiple right-hand sides. The LRU-cache pattern that motivates the
    upcoming solver-caching PR (#15)."""
    from torch_sla.backends.pyamg_backend import PyAMGHierarchy

    bench = benchmark_small_real
    A = SparseTensor(bench.val, bench.row, bench.col, bench.shape)
    H = PyAMGHierarchy.from_coo(bench.val, bench.row, bench.col, bench.shape)

    for case in bench:
        rhs = case["b"]
        x = torch.zeros_like(rhs)
        # Each V-cycle is a fixed contraction (~0.1 for classical AMG on
        # Poisson); 10 cycles reaches working precision reliably.
        for _ in range(10):
            r = rhs - (A @ x)
            x = x + H(r)
        rel_resid = ((rhs - (A @ x)).norm() / rhs.norm()).item()
        assert rel_resid < 1e-6, (
            f"seed={case['seed']}: rel residual after 10 V-cycles = "
            f"{rel_resid}"
        )


def test_pyamg_hierarchy_levels_diminish():
    """A well-formed hierarchy has at least two levels for a non-trivial
    problem, and coarse-grid sizes strictly diminish. Uses a fixed
    catalogued benchmark (poisson_2d_64) so the diminish-rate is
    reproducible across runs."""
    from torch_sla.backends.pyamg_backend import PyAMGHierarchy
    b = Synthetic["poisson_2d_64"]
    H = PyAMGHierarchy.from_coo(b.val, b.row, b.col, b.shape)
    assert len(H.levels) >= 2, (
        f"expected multi-level hierarchy, got {len(H.levels)} levels"
    )
    sizes = [L.A.shape[0] for L in H.levels]
    assert all(sizes[i] > sizes[i + 1] for i in range(len(sizes) - 1)), (
        f"coarse levels not strictly diminishing: {sizes}"
    )


# =====================================================================
# Cross-device (CUDA when available)
# =====================================================================
@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="CUDA not available")
def test_pyamg_hierarchy_runs_on_cuda(benchmark_small_real):
    """Build hierarchy on CPU (PyAMG), transfer operators to CUDA, run
    the V-cycle on GPU. This is the core hybrid-platform claim:
    coarsening stays on CPU, every per-solve cost lifts to the device."""
    from torch_sla.backends.pyamg_backend import PyAMGHierarchy
    b = benchmark_small_real
    val = b.val.cuda(); row = b.row.cuda(); col = b.col.cuda()
    H = PyAMGHierarchy.from_coo(val, row, col, b.shape, device=val.device)
    assert H.device.type == "cuda"
    rhs = b[0]["b"].cuda()
    x = H.v_cycle(rhs)
    assert x.device.type == "cuda"
    assert torch.isfinite(x).all().item()


# =====================================================================
# Availability sentinel
# =====================================================================
def test_pyamg_available_returns_bool():
    """``is_pyamg_available`` is a stable boolean -- safe for backend
    selection logic. (At collection time it returned ``True``; this
    re-validates inside the test.)"""
    assert is_pyamg_available() is True
