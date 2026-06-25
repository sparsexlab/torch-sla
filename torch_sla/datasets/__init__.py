"""``torch_sla.datasets`` -- a single, reusable source of benchmark / test
problems.

Two families live here:

* **Analytical / PDE problems** (:mod:`torch_sla.datasets.problems`) -- the
  generators :func:`laplacian_1d`, :func:`poisson_2d`, :func:`bratu_1d`,
  :func:`helmholtz_1d`, ... each returning a :class:`SparseProblem`, plus the
  :data:`PROBLEMS` registry / :func:`get` / :func:`default_sizes` helpers.
* **Public datasets** (:mod:`torch_sla.datasets.suitesparse`) --
  :func:`load_suitesparse`, :func:`is_suitesparse_available`, and the curated
  :data:`RECOMMENDED` list.

The legacy :class:`~torch_sla.benchmark.Benchmark`-based collections
(:data:`Synthetic`, :data:`SuiteSparse`, :data:`DIMACS10`, :data:`Benchmarks`,
:func:`iter_benchmarks`, :class:`DatasetUnavailable`) are preserved and
re-exported for backward compatibility.
"""

from __future__ import annotations

# --- new problem package -------------------------------------------------- #
from .sparse_problem import SparseProblem
from .problems import (
    laplacian_1d,
    laplacian_2d,
    laplacian_3d,
    poisson_2d,
    poisson_3d,
    bratu_1d,
    helmholtz_1d,
    anisotropic_diffusion_2d,
    advection_diffusion_2d,
    laplacian_1d_eigenvalues,
    laplacian_2d_eigenvalues,
    PROBLEMS,
    list_problems,
    get,
    default_sizes,
)

from .suitesparse import (
    load_suitesparse,
    is_suitesparse_available,
    RECOMMENDED,
)

# --- legacy Benchmark collections (backward compatible) ------------------- #
from ._benchmarks import (
    Benchmarks,
    SuiteSparse,
    DIMACS10,
    Synthetic,
    DatasetUnavailable,
    cache_dir,
    iter_benchmarks,
)

__all__ = [
    # new
    "SparseProblem",
    "laplacian_1d",
    "laplacian_2d",
    "laplacian_3d",
    "poisson_2d",
    "poisson_3d",
    "bratu_1d",
    "helmholtz_1d",
    "anisotropic_diffusion_2d",
    "advection_diffusion_2d",
    "laplacian_1d_eigenvalues",
    "laplacian_2d_eigenvalues",
    "PROBLEMS",
    "list_problems",
    "get",
    "default_sizes",
    "load_suitesparse",
    "is_suitesparse_available",
    "RECOMMENDED",
    # legacy
    "Benchmarks",
    "SuiteSparse",
    "DIMACS10",
    "Synthetic",
    "DatasetUnavailable",
    "cache_dir",
    "iter_benchmarks",
]
