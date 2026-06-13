"""SparseTensor stack. Public surface re-exported from :mod:`.core`."""
from .core import (
    SparseTensor,
    SparseTensorList,
    LUFactorization,
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
