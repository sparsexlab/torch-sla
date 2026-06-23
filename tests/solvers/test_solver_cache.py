"""End-to-end tests for the LRU solver cache.

The cache is transparent infrastructure -- the class, the key
dataclass, and the make-key helper are all package-internal, so there
are no public-surface unit tests to write. The behaviour we *do* need
to lock down is what users actually observe through the public
:func:`torch_sla.solve` API: repeated solves on the same matrix reuse
setup state, distinct matrices each pay setup once, and the
``SOLVER_CACHE.clear()`` escape hatch forces a rebuild on the next
solve.

Every matrix comes from ``torch_sla.datasets`` -- no hand-rolled
stencils.
"""
from __future__ import annotations

import pytest

from torch_sla import SOLVER_CACHE, SparseTensor, solve
from torch_sla.backends import is_pyamg_available
from torch_sla.datasets import Synthetic


pytestmark = pytest.mark.skipif(
    not is_pyamg_available(),
    reason="pyamg is an optional dependency; install with `pip install pyamg`."
)


# =====================================================================
# Cache works transparently through solve()
# =====================================================================
def test_solve_reuses_setup_across_rhs_on_same_matrix(benchmark_small_real):
    """Solving the same A with several different right-hand sides
    only builds the AMG hierarchy once. The user notices the speedup
    without ever touching the cache surface."""
    SOLVER_CACHE.clear()
    before = SOLVER_CACHE.stats()
    bench = benchmark_small_real
    A = SparseTensor(bench.val, bench.row, bench.col, bench.shape)

    for case in bench:
        solve(A, case["b"], backend="pyamg", atol=1e-8, maxiter=30)

    after = SOLVER_CACHE.stats()
    misses = after["misses"] - before["misses"]
    hits = after["hits"] - before["hits"]
    assert misses == 1, f"expected 1 build, got {misses}"
    assert hits == len(bench) - 1, (
        f"expected {len(bench) - 1} hits, got {hits}"
    )


def test_solve_misses_on_distinct_catalogue_matrices(benchmark_small_real):
    """Two structurally different catalogued matrices both miss on
    their first solve and the cache holds both entries afterwards."""
    SOLVER_CACHE.clear()
    a = benchmark_small_real
    Aa = SparseTensor(a.val, a.row, a.col, a.shape)
    b = Synthetic["poisson_2d_64"]
    Ab = SparseTensor(b.val, b.row, b.col, b.shape)

    solve(Aa, a[0]["b"], backend="pyamg", maxiter=30)
    solve(Ab, b[0]["b"], backend="pyamg", maxiter=30)

    stats = SOLVER_CACHE.stats()
    assert stats["misses"] >= 2
    assert stats["size"] >= 2


def test_cache_clear_forces_rebuild_on_next_solve(benchmark_small_real):
    """``SOLVER_CACHE.clear()`` is the public escape hatch for the rare
    case where users need to force a rebuild (testing, benchmarking,
    deliberate invalidation). Verifies the miss counter strictly
    increases after a clear."""
    bench = benchmark_small_real
    A = SparseTensor(bench.val, bench.row, bench.col, bench.shape)

    SOLVER_CACHE.clear()
    baseline = SOLVER_CACHE.stats()["misses"]
    solve(A, bench[0]["b"], backend="pyamg", maxiter=30)
    assert SOLVER_CACHE.stats()["misses"] - baseline == 1

    SOLVER_CACHE.clear()
    solve(A, bench[0]["b"], backend="pyamg", maxiter=30)
    assert SOLVER_CACHE.stats()["misses"] - baseline == 2, (
        "after clear the second solve must miss again"
    )


def test_pyamg_preconditioner_factory_reuses_cache(benchmark_small_real):
    """The standalone ``pyamg_preconditioner`` factory returns the
    cached hierarchy when called twice on the same matrix -- the
    factory and the solver share one cache."""
    from torch_sla.backends.pyamg_backend import pyamg_preconditioner
    SOLVER_CACHE.clear()
    b = benchmark_small_real
    H1 = pyamg_preconditioner(b.val, b.row, b.col, b.shape)
    H2 = pyamg_preconditioner(b.val, b.row, b.col, b.shape)
    assert H1 is H2, (
        "second factory call should return the cached hierarchy"
    )
