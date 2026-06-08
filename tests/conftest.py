"""Shared pytest fixtures backed by ``torch_sla.datasets``.

Fixtures are deliberately split by source / kind so individual tests can
opt into the appropriate tier instead of every test paying for every
matrix:

==============================  ==================================  =======
Fixture                         Yields                              Network
==============================  ==================================  =======
``benchmark_suitesparse``       every SuiteSparse benchmark         yes
``benchmark_dimacs``            every DIMACS10 graph Laplacian      yes
``benchmark_synthetic``         every Synthetic PDE stencil         no
``benchmark_real``              every real-dtype benchmark          partial
``benchmark_complex``           every complex-dtype benchmark       partial
``benchmark_small_real``        a single tiny synthetic (n^2=256)   no
==============================  ==================================  =======

When a matrix cannot be downloaded (no internet, mirror down, ...) the
test is skipped via :func:`pytest.skip` rather than failing -- so the
library is air-gapped-CI friendly.

Tests that genuinely want exhaustive coverage can iterate
``torch_sla.datasets.all_benchmarks()`` directly.
"""
from __future__ import annotations

import pytest

from torch_sla.benchmark import Benchmark
from torch_sla.datasets import (
    DIMACS10,
    DatasetUnavailable,
    SuiteSparse,
    Synthetic,
)


def _get(registry, key: str) -> Benchmark:
    try:
        return registry[key]
    except DatasetUnavailable as e:
        pytest.skip(f"{key!r}: {e}")


# ---------------------------------------------------------------------- #
# By source
# ---------------------------------------------------------------------- #
@pytest.fixture(params=list(SuiteSparse.keys()))
def benchmark_suitesparse(request) -> Benchmark:
    """Each catalogued SuiteSparse matrix, one per test invocation.

    Needs network on first call; subsequent runs hit the cache.
    """
    return _get(SuiteSparse, request.param)


@pytest.fixture(params=list(DIMACS10.keys()))
def benchmark_dimacs(request) -> Benchmark:
    """Each catalogued DIMACS10 graph Laplacian.

    Downloaded from the SuiteSparse mirror (group ``DIMACS10``) and
    converted to ``L + eps*I`` so the solve target is SPD.
    """
    return _get(DIMACS10, request.param)


@pytest.fixture(params=list(Synthetic.keys()))
def benchmark_synthetic(request) -> Benchmark:
    """Each Synthetic PDE stencil. No network required."""
    return _get(Synthetic, request.param)


# ---------------------------------------------------------------------- #
# By dtype (mix all three sources)
# ---------------------------------------------------------------------- #
# Hardcoded carve-out at the catalogue-key level, so we don't have to
# instantiate every benchmark (which would trigger network downloads
# during pytest collection) just to read its dtype.
_COMPLEX_SUITESPARSE = {k for k in SuiteSparse if "complex" in k}
_COMPLEX_SYNTHETIC = {k for k in Synthetic if "helmholtz" in k}

_REAL_KEYS = (
    [("suitesparse", k) for k in SuiteSparse if k not in _COMPLEX_SUITESPARSE]
    + [("dimacs10", k) for k in DIMACS10]
    + [("synthetic", k) for k in Synthetic if k not in _COMPLEX_SYNTHETIC]
)
_COMPLEX_KEYS = (
    [("suitesparse", k) for k in _COMPLEX_SUITESPARSE]
    + [("synthetic", k) for k in _COMPLEX_SYNTHETIC]
)
_REGISTRIES = {"suitesparse": SuiteSparse, "dimacs10": DIMACS10,
               "synthetic": Synthetic}


@pytest.fixture(params=_REAL_KEYS, ids=[f"{s}:{k}" for s, k in _REAL_KEYS])
def benchmark_real(request) -> Benchmark:
    """Every real-dtype benchmark across all three sources."""
    source, key = request.param
    return _get(_REGISTRIES[source], key)


@pytest.fixture(params=_COMPLEX_KEYS, ids=[f"{s}:{k}" for s, k in _COMPLEX_KEYS])
def benchmark_complex(request) -> Benchmark:
    """Every complex-dtype benchmark across all three sources."""
    source, key = request.param
    return _get(_REGISTRIES[source], key)


# ---------------------------------------------------------------------- #
# Fast smoke
# ---------------------------------------------------------------------- #
@pytest.fixture
def benchmark_small_real() -> Benchmark:
    """A single 16x16 = 256-DOF synthetic Poisson 2D -- for very fast
    correctness smoke checks (gradcheck, residual sweep, ...)."""
    return Synthetic["poisson_2d_16"]
