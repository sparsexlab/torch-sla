"""
torch-sla: PyTorch Sparse Linear Algebra

A differentiable sparse linear equation solver library for PyTorch with multiple backends.

Backends
--------
- CPU: SciPy (LU, UMFPACK), Eigen (CG, BiCGStab)
- CUDA: CuPy (LU, CG, GMRES), cuDSS (LU, Cholesky, LDLT)

Features
--------
- Full gradient support via torch.autograd
- Backend/method separation for flexible solver selection
- Batch solving (same-layout and different-layout)
- SparseTensor class with norm, eigs, solve methods
- Distributed sparse matrices with halo exchange

Usage
-----
>>> import torch
>>> from torch_sla import spsolve, SparseTensor
>>>
>>> # Method 1: Direct function call with auto backend selection
>>> val = torch.tensor([4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0], dtype=torch.float64)
>>> row = torch.tensor([0, 0, 1, 1, 1, 2, 2])
>>> col = torch.tensor([0, 1, 0, 1, 2, 1, 2])
>>> b = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
>>> x = spsolve(val, row, col, (3, 3), b)  # Auto-selects scipy+lu on CPU
>>>
>>> # Method 2: Specify backend and method
>>> x = spsolve(val, row, col, (3, 3), b, backend='scipy', method='lu')
>>>
>>> # Method 3: SparseTensor class
>>> A = SparseTensor(val, row, col, (3, 3))
>>> is_sym = A.is_symmetric()  # Auto-computed, returns tensor
>>> x = A.solve(b)  # Auto-selects best method
>>> x = A.solve(b, backend='scipy', method='cg')  # Specify backend/method
>>> eigenvalues, eigenvectors = A.eigsh(k=2)
>>>
>>> # CUDA usage
>>> val_cuda = val.cuda()
>>> b_cuda = b.cuda()
>>> x = spsolve(val_cuda, row.cuda(), col.cuda(), (3, 3), b_cuda)  # Auto-selects cudss
>>> x = spsolve(val_cuda, row.cuda(), col.cuda(), (3, 3), b_cuda, backend='cudss', method='lu')
"""

from .linear_solve import (
    spsolve,
    spsolve_coo,
    spsolve_csr,
)

from .solve import (
    solve,
    SolveInfo,
    PreconditionerConfig,
    SolverConfig,
    MatrixLike,
)
from .det import DetConfig

# Side-effect import: installs SolverConfig.for_spd_gpu(), for_matrix(),
# etc. classmethods. Kept in its own module so the SolverConfig dataclass
# itself stays focused on the scope-stack machinery.
from . import presets as _presets  # noqa: F401

from .solver_cache import SOLVER_CACHE  # singleton; the class itself is internal

from .batch_solve import (
    spsolve_batch_same_layout,
    spsolve_batch_different_layout,
    spsolve_batch_coo_same_layout,
    spsolve_batch_coo_different_layout,
    ParallelBatchSolver,
)

from .sparse_tensor import (
    SparseTensor,
    SparseTensorList,
    LUFactorization,
    auto_select_method,
    estimate_direct_solver_memory,
    get_available_gpu_memory,
)

from .backends import (
    # Backend utilities
    get_available_backends,
    show_backends,
    get_backend_methods,
    get_default_method,
    select_backend,
    select_method,
    # Availability checks
    is_scipy_available,
    is_eigen_available,
    is_cupy_available,
    is_cudss_available,
    # Backend-method mappings
    BACKEND_METHODS,
    DEFAULT_METHODS,
    # Type aliases
    BackendType,
    MethodType,
)

from .distributed import (
    DSparseTensor,
    Partition,
    partition_graph_metis,
    partition_coordinates,
    partition_simple,
    # DTensor-mirror placement vocabulary
    DSparseSpec,
    VertexShard,
    VertexShardReplicated,
    BatchShard,
    Replicated,
    SparseShard,  # deprecated alias (factory)
)

from .io import (
    save_sparse,
    load_sparse,
    load_metadata,
    # Matrix Market format
    save_mtx,
    load_mtx,
    load_mtx_info,
    # Distributed I/O
    save_dsparse,
    load_dsparse,
    save_sparse_sharded,
    load_sparse_shard,
)

from .nonlinear_solve import (
    nonlinear_solve,
    adjoint_solve,
    NonlinearSolveAdjoint,
)

__version__ = "0.3.1"
__author__ = "Mingyuan Chi, Shizheng Wen"

__all__ = [
    # Single solve
    "spsolve",
    "spsolve_coo",
    "spsolve_csr",
    # Batch solve
    "spsolve_batch_same_layout",
    "spsolve_batch_different_layout",
    "spsolve_batch_coo_same_layout",
    "spsolve_batch_coo_different_layout",
    "ParallelBatchSolver",
    # SparseTensor class
    "SparseTensor",
    "SparseTensorList",
    "LUFactorization",
    "auto_select_method",
    "estimate_direct_solver_memory",
    "get_available_gpu_memory",
    # Backend utilities
    "get_available_backends",
    "show_backends",
    "get_backend_methods",
    "get_default_method",
    "select_backend",
    "select_method",
    "is_scipy_available",
    "is_eigen_available",
    "is_cupy_available",
    "is_cudss_available",
    "BACKEND_METHODS",
    "DEFAULT_METHODS",
    "BackendType",
    "MethodType",
    # Distributed
    "DSparseTensor",
    "Partition",
    "partition_graph_metis",
    "partition_coordinates",
    "partition_simple",
    # I/O
    "save_sparse",
    "load_sparse",
    "load_metadata",
    # Matrix Market format
    "save_mtx",
    "load_mtx",
    "load_mtx_info",
    # Distributed I/O
    "save_dsparse",
    "load_dsparse",
    "save_sparse_sharded",
    "load_sparse_shard",
    # Nonlinear solve
    "nonlinear_solve",
    "adjoint_solve",
    "NonlinearSolveAdjoint",
    # Version
    "__version__",
]
