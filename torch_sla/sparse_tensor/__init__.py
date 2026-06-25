"""SparseTensor stack. Public surface re-exported from :mod:`.core`."""
from .core import SparseTensor, LUFactorization
from .list import SparseTensorList
from .utils import (
    auto_select_method,
    estimate_direct_solver_memory,
    get_available_gpu_memory,
)
from .autograd import (
    DetAdjoint,
    EigshAdjoint,
    SparseSolveFunction,
    SparseSparseMatmulFunction,
)

# Now that every submodule is imported (no circular-import risk), copy the
# rich operation docstrings from their implementations (linalg / graph) onto
# the thin delegating wrappers on ``SparseTensor`` so Sphinx autodoc renders
# them. See ``core._propagate_op_docstrings``.
from .core import _propagate_op_docstrings as _propagate_op_docstrings
_propagate_op_docstrings()

__all__ = [
    "SparseTensor",
    "SparseTensorList",
    "LUFactorization",
    "DetAdjoint",
    "EigshAdjoint",
    "SparseSolveFunction",
    "SparseSparseMatmulFunction",
    "auto_select_method",
    "estimate_direct_solver_memory",
    "get_available_gpu_memory",
]
