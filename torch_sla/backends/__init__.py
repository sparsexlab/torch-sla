"""
Backend management for torch-sla

This module provides a unified interface for different sparse linear algebra backends:

Backends:
- 'scipy': SciPy backend (CPU only) - Uses LU for direct solvers
- 'eigen': Eigen backend (CPU only) - Iterative solvers (CG, BiCGStab)
- 'pytorch': PyTorch-native (CPU & CUDA) - Iterative solvers with Jacobi preconditioning
- 'cupy': CuPy backend (CUDA only) - Direct and iterative solvers via cupyx.scipy
- 'cudss': NVIDIA cuDSS via nvmath-python (CUDA only) - Direct solvers (LU, Cholesky, LDLT)

Methods (solver algorithms):
- 'lu': LU factorization (scipy, cupy, cudss)
- 'umfpack': UMFPACK direct solver (scipy only, requires scikit-umfpack)
- 'cholesky': Cholesky decomposition (direct, SPD matrices, cudss)
- 'ldlt': LDLT decomposition (direct, symmetric matrices, cudss)
- 'cg': Conjugate Gradient (iterative, for SPD matrices)
- 'bicgstab': BiCGStab (iterative, general matrices)
- 'gmres': GMRES (iterative, general matrices)

Recommended Backends (based on benchmarks):
==========================================

float64 (double precision):
---------------------------
- CPU, DOF < 100K: scipy+lu (best balance of speed and accuracy)
- CPU, DOF >= 100K: scipy+lu (still fast, machine precision ~1e-15)
- CUDA, DOF < 100K: cudss+cholesky (for SPD) or cudss+lu (general)
- CUDA, DOF >= 100K: cudss+cholesky (direct, ~1e-14 precision)
- CUDA, DOF > 2M: pytorch+cg (iterative, ~1e-6 precision, memory efficient)

float32 (single precision):
---------------------------
- CPU: scipy+lu (precision ~1e-6)
- CUDA: cudss+cholesky/ldlt (precision ~1e-6)
- Note: Iterative methods may not converge well with float32

Usage:
    # Auto-select backend based on device and problem size
    x = spsolve(A, b)  # Uses scipy for CPU, cudss for CUDA (small), pytorch for CUDA (large)

    # Specify backend and method
    x = spsolve(A, b, backend='scipy', method='lu')
    x = spsolve(A, b, backend='cudss', method='cholesky')
    x = spsolve(A, b, backend='pytorch', method='cg')  # GPU iterative
    x = spsolve(A, b, backend='cupy', method='spsolve')  # CuPy direct on GPU
"""

from typing import Optional, List, Dict, Literal
import torch

# Type aliases
BackendType = Literal['scipy', 'eigen', 'pytorch', 'cupy', 'cudss', 'pyamg',
                      'amgx', 'auto']
MethodType = Literal[
    'auto',
    # Direct methods
    'lu', 'umfpack', 'cholesky', 'ldlt',
    # Iterative methods
    'cg', 'cgs', 'bicgstab', 'gmres', 'lgmres', 'minres', 'qmr', 'lsqr', 'lsmr',
    # AMG variants (pyamg / amgx / petsc)
    'amg', 'ruge_stuben', 'smoothed_aggregation', 'sa',
]

# Backend -> supported methods mapping
BACKEND_METHODS: Dict[str, List[str]] = {
    'scipy': ['lu', 'umfpack', 'cg', 'bicgstab', 'gmres', 'lgmres', 'minres', 'qmr'],
    'eigen': ['cg', 'bicgstab'],
    'pytorch': ['cg', 'bicgstab'],  # PyTorch-native iterative with Jacobi preconditioning
    'cupy': ['lu', 'cg', 'cgs', 'gmres', 'minres', 'lsqr', 'lsmr'],
    'cudss': ['lu', 'cholesky', 'ldlt'],
    'pyamg': ['amg', 'ruge_stuben', 'smoothed_aggregation', 'sa'],
    'amgx': ['amg', 'cg', 'pcg', 'bicgstab', 'pbicgstab', 'gmres', 'fgmres'],
}

# Default methods for each backend (based on benchmarks)
DEFAULT_METHODS: Dict[str, str] = {
    'scipy': 'lu',            # Best for CPU: fast + machine precision (SuperLU)
    'eigen': 'bicgstab',
    'pytorch': 'cg',         # Use CG for SPD (most common), with Jacobi preconditioning
    'cupy': 'lu',             # GPU direct solver via SuperLU
    'cudss': 'cholesky',     # Best for CUDA: fastest + high precision
    'pyamg': 'ruge_stuben',   # Classical AMG; works for most PDE / SPD problems
    'amgx':  'pbicgstab',     # AmgX's most robust default; AMG-preconditioned
}

# Threshold for switching from direct to iterative on CUDA (DOF)
# Based on benchmark: direct solvers (cudss) work well up to ~2M DOF
CUDA_ITERATIVE_THRESHOLD = 2_000_000

# Backend availability flags
_scipy_available: Optional[bool] = None
_cupy_available: Optional[bool] = None
_cudss_available: Optional[bool] = None
_eigen_available: Optional[bool] = None


def _check_cuda() -> bool:
    """Check if CUDA is available"""
    return torch.cuda.is_available()


def is_scipy_available() -> bool:
    """Check if SciPy backend is available"""
    global _scipy_available
    if _scipy_available is None:
        try:
            import scipy.sparse.linalg
            _scipy_available = True
        except ImportError:
            _scipy_available = False
    return _scipy_available


def is_eigen_available() -> bool:
    """Check if Eigen backend (C++ extension) is available"""
    global _eigen_available
    if _eigen_available is None:
        try:
            _load_eigen_backend()
            _eigen_available = True
        except Exception:
            _eigen_available = False
    return _eigen_available


def is_pytorch_available() -> bool:
    """Check if PyTorch-native backend is available (always True)"""
    return True


def is_cupy_available() -> bool:
    """Check if CuPy backend is available"""
    global _cupy_available
    if _cupy_available is None:
        if not _check_cuda():
            _cupy_available = False
        else:
            try:
                import cupy  # noqa: F401
                import cupyx.scipy.sparse.linalg  # noqa: F401
                _cupy_available = True
            except ImportError:
                _cupy_available = False
    return _cupy_available


def is_cudss_available() -> bool:
    """Check if cuDSS backend is available (via nvmath-python)"""
    global _cudss_available
    if _cudss_available is None:
        if not _check_cuda():
            _cudss_available = False
        else:
            try:
                import nvmath.bindings.cudss  # noqa: F401
                _cudss_available = True
            except ImportError:
                _cudss_available = False
    return _cudss_available


_torch_spsolve_available = None
def is_torch_spsolve_available() -> bool:
    """Check if :func:`torch.sparse.spsolve` is dispatchable on the
    default device. On PyTorch 2.12 the underlying ``aten::_spsolve``
    op is only registered for MPS; CPU/CUDA raise ``NotImplementedError``
    so this returns False on those devices.

    When upstream PyTorch lands CPU/CUDA kernels for ``_spsolve``,
    this will start returning True with no code change here.
    """
    global _torch_spsolve_available
    if _torch_spsolve_available is None:
        try:
            from .torch_spsolve_backend import is_torch_spsolve_supported
            # Probe on the default device (CPU is safest -- works as
            # a "does the dispatch table have an entry" check).
            _torch_spsolve_available = is_torch_spsolve_supported(
                torch.device("cpu"))
        except Exception:
            _torch_spsolve_available = False
    return _torch_spsolve_available


_pyamg_available = None
def is_pyamg_available() -> bool:
    """Check if the PyAMG backend is available (CPU AMG; cross-platform)."""
    global _pyamg_available
    if _pyamg_available is None:
        try:
            import pyamg  # noqa: F401
            _pyamg_available = True
        except ImportError:
            _pyamg_available = False
    return _pyamg_available


_amgx_available = None
def is_amgx_available() -> bool:
    """Check if the AmgX backend is available (Linux + Windows + NVIDIA CUDA).

    The ``torch-amgx`` PyPI package (https://pypi.org/p/torch-amgx) bundles
    the AmgX shared library inside its wheels for Linux + Windows. Install
    via::

        pip install torch-amgx                # or: pip install torch-sla[amgx]

    macOS is not supported (NVIDIA does not ship CUDA on macOS).
    """
    global _amgx_available
    if _amgx_available is None:
        if not _check_cuda():
            _amgx_available = False
        else:
            try:
                import torch_amgx  # noqa: F401
                _amgx_available = torch_amgx.is_available()
            except ImportError:
                _amgx_available = False
    return _amgx_available


def get_available_backends() -> List[str]:
    """Get list of available backends"""
    backends = []

    if is_scipy_available():
        backends.append('scipy')

    if is_eigen_available():
        backends.append('eigen')

    backends.append('pytorch')  # Always available

    if is_cupy_available():
        backends.append('cupy')

    if is_cudss_available():
        backends.append('cudss')

    return backends


# Per-backend status descriptors used by show_backends()
_BACKEND_DESCRIPTIONS: Dict[str, Dict[str, str]] = {
    'scipy':   {'device': 'CPU',  'install': 'pip install scipy'},
    'eigen':   {'device': 'CPU',  'install': 'JIT-compiled C++ extension (requires a C++ compiler)'},
    'pytorch': {'device': 'CPU/CUDA', 'install': 'bundled with torch (always available)'},
    'cupy':    {'device': 'CUDA', 'install': 'pip install torch-sla[cupy]'},
    'cudss':   {'device': 'CUDA', 'install': 'pip install torch-sla[cudss]'},
}


def show_backends() -> None:
    """Print a formatted status report of all backends.

    Shows which backends are available on the current machine and gives
    installation hints for the ones that are not. Useful right after
    ``pip install torch-sla`` to verify the runtime environment.

    Example
    -------
    >>> import torch_sla
    >>> torch_sla.show_backends()
    torch-sla backend status (CUDA: available)
      scipy    [CPU]      available
      eigen    [CPU]      available
      pytorch  [CPU/CUDA] available
      cupy     [CUDA]     not available — pip install torch-sla[cupy]
      cudss    [CUDA]     not available — pip install torch-sla[cudss]
    """
    checks = [
        ('scipy',   is_scipy_available()),
        ('eigen',   is_eigen_available()),
        ('pytorch', is_pytorch_available()),
        ('cupy',    is_cupy_available()),
        ('cudss',   is_cudss_available()),
    ]
    cuda_status = 'available' if _check_cuda() else 'not available'
    print(f"torch-sla backend status (CUDA: {cuda_status})")
    for name, ok in checks:
        info = _BACKEND_DESCRIPTIONS[name]
        device = f"[{info['device']}]"
        if ok:
            status = "available"
        else:
            status = f"not available — {info['install']}"
        print(f"  {name:<8} {device:<11} {status}")


def get_backend_methods(backend: str) -> List[str]:
    """Get list of methods supported by a backend"""
    return BACKEND_METHODS.get(backend, [])


def get_default_method(backend: str) -> str:
    """Get default method for a backend"""
    return DEFAULT_METHODS.get(backend, 'auto')


def select_backend(
    device: torch.device,
    n: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
    prefer_direct: bool = True
) -> str:
    """
    Auto-select the best backend based on device, problem size, and dtype.

    Recommendations based on benchmark results:
    - CPU: scipy+lu (all sizes, fast + machine precision)
    - CUDA (DOF < 2M): cudss+cholesky (fast + high precision)
    - CUDA (DOF >= 2M): pytorch+cg (memory efficient, ~1e-6 precision)

    Parameters
    ----------
    device : torch.device
        Target device (cpu or cuda)
    n : int, optional
        Problem size (DOF). If > CUDA_ITERATIVE_THRESHOLD, prefer iterative.
    dtype : torch.dtype, optional
        Data type.
    prefer_direct : bool
        If True, prefer direct solvers over iterative (when applicable)

    Returns
    -------
    str
        Backend name ('scipy', 'eigen', 'pytorch', 'cupy', or 'cudss')
    """
    if device.type == 'cpu':
        # CPU: scipy is best (SuperLU: fast + machine precision)
        if is_scipy_available():
            return 'scipy'
        if is_eigen_available():
            return 'eigen'
        return 'pytorch'  # Fallback to PyTorch-native

    elif device.type == 'cuda':
        # Large problem: use iterative (PyTorch-native on GPU with Jacobi preconditioning)
        if n is not None and n > CUDA_ITERATIVE_THRESHOLD:
            return 'pytorch'

        # Small/medium problem: prefer direct solvers
        if prefer_direct:
            # cuDSS is best for CUDA (supports both float32 and float64)
            if is_cudss_available():
                return 'cudss'
            # CuPy as fallback (also supports float32 and float64)
            if is_cupy_available():
                return 'cupy'

        # Fallback to iterative
        return 'pytorch'

    else:
        raise ValueError(f"Unsupported device type: {device.type}")


def select_method(
    backend: str,
    is_symmetric: bool = False,
    is_spd: bool = False,
    prefer_direct: bool = True
) -> str:
    """
    Auto-select the best method for a given backend and matrix properties.

    Recommendations based on benchmark results:
    - scipy: lu (direct, best precision) or cg (iterative, for SPD)
    - cudss: cholesky (SPD, fastest) > ldlt (symmetric) > lu (general)
    - cupy: lu (direct) or cg (iterative, for SPD)
    - pytorch: cg (SPD) or bicgstab (general), both with Jacobi preconditioning

    Parameters
    ----------
    backend : str
        Backend name
    is_symmetric : bool
        Whether the matrix is symmetric
    is_spd : bool
        Whether the matrix is symmetric positive definite
    prefer_direct : bool
        If True, prefer direct solvers

    Returns
    -------
    str
        Method name
    """
    methods = BACKEND_METHODS.get(backend, [])

    if backend == 'scipy':
        if prefer_direct:
            return 'lu'  # Best: fast + machine precision (SuperLU)
        elif is_spd:
            return 'cg'
        else:
            return 'bicgstab'

    elif backend == 'eigen':
        return 'cg' if is_spd else 'bicgstab'

    elif backend == 'pytorch':
        # Iterative with Jacobi preconditioning
        return 'cg' if is_spd else 'bicgstab'

    elif backend == 'cupy':
        if prefer_direct:
            return 'lu'
        elif is_spd:
            return 'cg'
        else:
            return 'gmres'

    elif backend == 'cudss':
        # Recommendation: cholesky > ldlt > lu (based on benchmarks)
        if is_spd and 'cholesky' in methods:
            return 'cholesky'  # Fastest for SPD
        elif is_symmetric and 'ldlt' in methods:
            return 'ldlt'
        return 'lu'

    return DEFAULT_METHODS.get(backend, methods[0] if methods else 'auto')


def _load_eigen_backend():
    """Load Eigen backend (C++ iterative solvers)"""
    global _eigen_module

    if _eigen_module is not None:
        return _eigen_module

    import os
    from torch.utils.cpp_extension import load

    try:
        _eigen_module = load(
            name="spsolve",
            sources=[os.path.abspath(os.path.join(
                os.path.dirname(__file__), "..", "..", "csrc", "spsolve", "spsolve.cpp"
            ))],
            verbose=False
        )
        return _eigen_module
    except Exception as e:
        raise RuntimeError(f"Failed to load Eigen backend: {e}")


# Lazy-loaded modules
_eigen_module = None


# Convenience functions for getting modules
def get_cpu_module():
    """Get Eigen backend module (legacy name for compatibility)"""
    return get_eigen_module()


def get_eigen_module():
    """Get Eigen backend module"""
    return _load_eigen_backend()


def get_cudss_module():
    """Get cuDSS backend module (via nvmath-python)"""
    from .nvmath_backend import _NvmathCudssModule
    return _NvmathCudssModule()
