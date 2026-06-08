"""Shared pytest fixtures for torch-sla tests.

Exposes parametrised fixtures backed by :class:`torch_sla.datasets.SuiteSparse`,
the public benchmark API. Tests that need a sparse benchmark just declare
the ``benchmark`` (any matrix) or ``benchmark_complex`` (complex only)
parameter -- no in-test downloading boilerplate.

If a matrix cannot be downloaded (no internet, mirror down, ...) the
test is skipped via :func:`pytest.skip` rather than failing, so CI on
air-gapped runners degrades gracefully.
"""
from __future__ import annotations

import pytest

from torch_sla.benchmark import Benchmark
from torch_sla.datasets import SuiteSparse, DatasetUnavailable


def _get(key: str) -> Benchmark:
    try:
        return SuiteSparse[key]
    except DatasetUnavailable as e:
        pytest.skip(f"{key!r}: {e}")


_ALL = list(SuiteSparse.keys())
_COMPLEX = [k for k in _ALL if "complex" in k]


@pytest.fixture(params=_ALL)
def benchmark(request) -> Benchmark:
    """Parametrised: yields one :class:`Benchmark` per catalogued matrix."""
    return _get(request.param)


@pytest.fixture(params=_COMPLEX)
def benchmark_complex(request) -> Benchmark:
    """Parametrised over complex-only matrices."""
    return _get(request.param)


@pytest.fixture
def benchmark_complex_small() -> Benchmark:
    """Smallest complex matrix (qc324, n=324) — for slow tests like gradcheck."""
    return _get("complex_sym")
