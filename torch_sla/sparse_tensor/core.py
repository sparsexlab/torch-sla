"""
SparseTensor wrapper class for PyTorch sparse tensors.

Supports batched and block sparse tensors with shape [...batch, M, N, ...block]:
- Leading dimensions: batch dimensions [B1, B2, ...]
- Matrix dimensions: (M, N) at positions (sparse_dim[0], sparse_dim[1]), default (-2, -1)
- Trailing dimensions: block dimensions [K1, K2, ...]

Key Features:
- Automatic symmetry and positive definiteness detection
- Sparse linear equation solving with gradient support
- Sparse-sparse multiplication with sparse gradients
- Batched operations for all methods
- CUDA support with LOBPCG for eigenvalue computation

Examples
--------
>>> # Create a simple sparse matrix
>>> val = torch.tensor([4.0, -1.0, -1.0, 4.0])
>>> row = torch.tensor([0, 0, 1, 1])
>>> col = torch.tensor([0, 1, 0, 1])
>>> A = SparseTensor(val, row, col, (2, 2))
>>>
>>> # Check properties (returns boolean tensor for batched)
>>> is_sym = A.is_symmetric()  # tensor(True)
>>> is_pd = A.is_positive_definite()  # tensor(True)
>>>
>>> # Solve linear system
>>> b = torch.tensor([1.0, 2.0])
>>> x = A.solve(b)
>>>
>>> # Matrix operations
>>> y = A @ x  # Sparse @ Dense
>>> C = A @ A  # Sparse @ Sparse (sparse gradient)
"""

import os
import torch
from torch.autograd.function import Function
from typing import Tuple, Optional, Union, Literal, List, Dict
import warnings
import math

from ..backends import (
    is_scipy_available,
    is_eigen_available,
    is_cupy_available,
    is_cudss_available,
    select_backend,
    select_method,
    BackendType,
    MethodType,
)
from ..backends.scipy_backend import (
    scipy_solve,
    scipy_eigs,
    scipy_eigsh,
    scipy_svds,
    scipy_norm,
    scipy_lu,
    scipy_det,
)

from .autograd import (
    DetAdjoint,
    EigshAdjoint,
    SparseSolveFunction,
    SparseSparseMatmulFunction,
    _sparse_sparse_matmul_with_sparse_grad,
)


# =============================================================================
# Utility Functions
# =============================================================================

def estimate_direct_solver_memory(nnz: int, n: int, dtype: torch.dtype) -> int:
    """
    Estimate memory required for direct sparse solver.
    
    Parameters
    ----------
    nnz : int
        Number of non-zero elements.
    n : int
        Matrix dimension.
    dtype : torch.dtype
        Data type of the matrix.
    
    Returns
    -------
    int
        Estimated memory in bytes.
    """
    bytes_per_element = 8 if dtype == torch.float64 else 4
    fill_factor = min(10, max(2, n / 100))
    factor_memory = int(nnz * fill_factor * bytes_per_element)
    workspace_memory = n * bytes_per_element * 10
    return factor_memory + workspace_memory


def get_available_gpu_memory() -> int:
    """
    Get available GPU memory in bytes.
    
    Returns
    -------
    int
        Available GPU memory in bytes, or 0 if CUDA is not available.
    """
    if not torch.cuda.is_available():
        return 0
    try:
        free_memory, total_memory = torch.cuda.mem_get_info()
        return free_memory
    except Exception:
        return torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()


def auto_select_method(
    nnz: int, n: int, dtype: torch.dtype, is_cuda: bool, is_spd: bool = False,
    memory_threshold: float = 0.8
) -> Tuple[str, str]:
    """
    Automatically select the best backend and method.
    
    Parameters
    ----------
    nnz : int
        Number of non-zero elements.
    n : int
        Matrix dimension.
    dtype : torch.dtype
        Data type of the matrix.
    is_cuda : bool
        Whether the matrix is on CUDA.
    is_spd : bool, optional
        Whether the matrix is symmetric positive definite. Default: False.
    memory_threshold : float, optional
        Fraction of GPU memory to use. Default: 0.8.
        
    Returns
    -------
    Tuple[str, str]
        (backend, method) tuple.
    """
    if not is_cuda:
        if is_scipy_available():
            return ("scipy", "lu")
        elif is_eigen_available():
            return ("eigen", "cg" if is_spd else "bicgstab")
        else:
            raise RuntimeError("No CPU backend available")
    
    estimated_memory = estimate_direct_solver_memory(nnz, n, dtype)
    available_memory = get_available_gpu_memory()
    
    if available_memory > 0 and estimated_memory < available_memory * memory_threshold:
        if is_cudss_available():
            return ("cudss", "cholesky" if is_spd else "lu")
        elif is_cupy_available():
            return ("cupy", "lu")
    
    if is_scipy_available():
        return ("scipy", "lu")
    
    raise RuntimeError("No suitable backend available")


def _power_iteration_svd(
    A_matvec,
    At_matvec,
    m: int,
    n: int,
    k: int,
    dtype: torch.dtype,
    device: torch.device,
    maxiter: int = 100,
    tol: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Power iteration based SVD for sparse matrices on any device.
    
    Parameters
    ----------
    A_matvec : callable
        Function that computes A @ x.
    At_matvec : callable
        Function that computes A^T @ x.
    m, n : int
        Matrix dimensions (m rows, n columns).
    k : int
        Number of singular values to compute.
    dtype : torch.dtype
        Data type.
    device : torch.device
        Device to compute on.
    maxiter : int, optional
        Maximum iterations. Default: 100.
    tol : float, optional
        Convergence tolerance. Default: 1e-6.
    
    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        (U, S, Vt) with shapes [m, k], [k], [k, n].
    """
    V = torch.randn(n, k, dtype=dtype, device=device)
    V, _ = torch.linalg.qr(V)
    
    for _ in range(maxiter):
        U = A_matvec(V)
        U, R = torch.linalg.qr(U)
        V_new = At_matvec(U)
        S = V_new.norm(dim=0)
        V_new = V_new / S.unsqueeze(0).clamp(min=1e-10)
        diff = (V_new - V).norm()
        V = V_new
        if diff < tol:
            break
    
    return U, S, V.T


# =============================================================================
# SparseTensor Class
# =============================================================================

class SparseTensor:
    """
    Wrapper class for PyTorch sparse tensors with batched and block support.
    
    Supports tensors with shape [...batch, M, N, ...block] where:
    - Leading dimensions [...batch] are batch dimensions
    - (M, N) are the sparse matrix dimensions (at sparse_dim positions)
    - Trailing dimensions [...block] are block dimensions
    
    Parameters
    ----------
    values : torch.Tensor
        Non-zero values with shape:
        - Simple: [nnz]
        - Batched: [...batch, nnz] 
        - Block: [nnz, *block_shape]
        - Batched+Block: [...batch, nnz, *block_shape]
    row_indices : torch.Tensor
        Row indices with shape [nnz]. Must be on the same device as values.
    col_indices : torch.Tensor
        Column indices with shape [nnz]. Must be on the same device as values.
    shape : Tuple[int, ...]
        Full tensor shape [...batch, M, N, *block_shape].
    sparse_dim : Tuple[int, int], optional
        Which dimensions are sparse (M, N). Default: (-2, -1) meaning last two
        before any block dimensions.
    
    Attributes
    ----------
    values : torch.Tensor
        The non-zero values.
    row_indices : torch.Tensor
        Row indices of non-zeros.
    col_indices : torch.Tensor
        Column indices of non-zeros.
    shape : Tuple[int, ...]
        Full tensor shape.
    sparse_shape : Tuple[int, int]
        The (M, N) dimensions.
    batch_shape : Tuple[int, ...]
        The batch dimensions.
    block_shape : Tuple[int, ...]
        The block dimensions.
    
    Examples
    --------
    **1. Simple 2D Sparse Matrix [M, N]**
    
    >>> import torch
    >>> from torch_sla import SparseTensor
    >>> 
    >>> # Create a 3x3 tridiagonal matrix in COO format
    >>> val = torch.tensor([4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0])
    >>> row = torch.tensor([0, 0, 1, 1, 1, 2, 2])
    >>> col = torch.tensor([0, 1, 0, 1, 2, 1, 2])
    >>> A = SparseTensor(val, row, col, (3, 3))
    >>> print(A)
    SparseTensor(shape=(3, 3), sparse=(3, 3), nnz=7, dtype=torch.float64, device=cpu)
    >>> 
    >>> # Solve Ax = b
    >>> b = torch.tensor([1.0, 2.0, 3.0])
    >>> x = A.solve(b)
    
    **2. Batched Sparse Matrices [B, M, N]**
    
    Same sparsity pattern, different values for each batch.
    
    >>> # 4 matrices, each 3x3, same structure
    >>> batch_size = 4
    >>> val_batch = val.unsqueeze(0).expand(batch_size, -1).clone()  # [4, 7]
    >>> for i in range(batch_size):
    ...     val_batch[i] = val * (1.0 + 0.1 * i)  # Scale each matrix
    >>> 
    >>> A_batch = SparseTensor(val_batch, row, col, (4, 3, 3))
    >>> print(A_batch.batch_shape)  # (4,)
    >>> print(A_batch.sparse_shape)  # (3, 3)
    >>> 
    >>> # Batched solve
    >>> b_batch = torch.randn(4, 3)
    >>> x_batch = A_batch.solve(b_batch)  # [4, 3]
    
    **3. Multi-Dimensional Batch [B1, B2, M, N]**
    
    >>> B1, B2 = 2, 3  # e.g., 2 materials x 3 temperatures
    >>> val_batch = val.unsqueeze(0).unsqueeze(0).expand(B1, B2, -1).clone()  # [2, 3, 7]
    >>> A_multi = SparseTensor(val_batch, row, col, (B1, B2, 3, 3))
    >>> print(A_multi.batch_shape)  # (2, 3)
    >>> 
    >>> b_multi = torch.randn(B1, B2, 3)
    >>> x_multi = A_multi.solve(b_multi)  # [2, 3, 3]
    
    **4. Block Sparse Matrix [M, N, K, K] (Block Size K)**
    
    Each non-zero entry is a KxK dense block instead of a scalar.
    
    >>> # 2x2 block matrix with 2x2 blocks = 4x4 total
    >>> block_size = 2
    >>> nnz = 3  # 3 non-zero blocks
    >>> 
    >>> # Values: [nnz, K, K] = [3, 2, 2]
    >>> val_block = torch.randn(nnz, block_size, block_size)
    >>> row_block = torch.tensor([0, 0, 1])  # Block row indices
    >>> col_block = torch.tensor([0, 1, 1])  # Block col indices
    >>> 
    >>> # Shape: (num_block_rows, num_block_cols, block_size, block_size)
    >>> A_block = SparseTensor(val_block, row_block, col_block, (2, 2, 2, 2))
    >>> print(A_block.block_shape)  # (2, 2)
    >>> print(A_block.sparse_shape)  # (2, 2) - number of blocks
    >>> print(A_block.shape)  # (2, 2, 2, 2) - full shape
    
    **5. Batched Block Sparse [B, M, N, K, K]**
    
    >>> batch_size = 4
    >>> val_batch_block = torch.randn(batch_size, nnz, block_size, block_size)  # [4, 3, 2, 2]
    >>> A_batch_block = SparseTensor(val_batch_block, row_block, col_block, (4, 2, 2, 2, 2))
    >>> print(A_batch_block.batch_shape)  # (4,)
    >>> print(A_batch_block.block_shape)  # (2, 2)
    
    **6. Create from Dense Matrix**
    
    >>> A_dense = torch.randn(100, 100)
    >>> A_dense[A_dense.abs() < 0.5] = 0  # Sparsify
    >>> A = SparseTensor.from_dense(A_dense)
    
    **7. Create from PyTorch Sparse Tensor**
    
    >>> A_torch = torch.randn(100, 100).to_sparse_coo()
    >>> A = SparseTensor.from_torch_sparse(A_torch)
    
    **8. Property Detection**
    
    >>> A = SparseTensor(val, row, col, (3, 3))
    >>> A.is_symmetric()  # tensor(True) - returns tensor for batch support
    >>> A.is_positive_definite()  # tensor(True)
    >>> A.is_positive_definite('cholesky')  # Use Cholesky factorization check
    
    **9. Matrix Operations**
    
    >>> # Matrix-vector multiply
    >>> y = A @ x  # SparseTensor @ dense vector
    >>> 
    >>> # Sparse-sparse multiply (returns SparseTensor with sparse gradients)
    >>> C = A @ A
    >>> 
    >>> # Norms
    >>> A.norm('fro')  # Frobenius norm
    >>> 
    >>> # Eigenvalues (symmetric matrices)
    >>> eigenvalues, eigenvectors = A.eigsh(k=2, which='LM')
    
    **10. CUDA Support**
    
    >>> A_cuda = A.cuda()
    >>> x = A_cuda.solve(b.cuda())  # Uses cuDSS or CuPy
    """
    
    def __init__(
        self,
        values: torch.Tensor,
        row_indices: torch.Tensor,
        col_indices: torch.Tensor,
        shape: Tuple[int, ...],
        sparse_dim: Tuple[int, int] = (-2, -1),
    ):
        self.values = values
        self.row_indices = row_indices
        self.col_indices = col_indices
        self._shape = tuple(shape)
        self._sparse_dim = self._normalize_sparse_dim(sparse_dim, len(shape))
        
        # Cache for computed properties
        self._is_symmetric_cache = None
        self._is_hermitian_cache = None
        self._is_positive_definite_cache = None
        
        self._validate()
    
    def _normalize_sparse_dim(self, sparse_dim: Tuple[int, int], ndim: int) -> Tuple[int, int]:
        """Normalize negative indices in sparse_dim."""
        dim_m = sparse_dim[0] if sparse_dim[0] >= 0 else ndim + sparse_dim[0]
        dim_n = sparse_dim[1] if sparse_dim[1] >= 0 else ndim + sparse_dim[1]
        return (dim_m, dim_n)
    
    def _validate(self):
        """Validate tensor dimensions and indices."""
        ndim = len(self._shape)
        dim_m, dim_n = self._sparse_dim
        if ndim < 2:
            raise ValueError(f"Shape must have at least 2 dimensions, got {ndim}")
        if not (0 <= dim_m < ndim and 0 <= dim_n < ndim):
            raise ValueError(f"sparse_dim {self._sparse_dim} out of range for shape {self._shape}")
        if dim_m == dim_n:
            raise ValueError(f"sparse_dim dimensions must be different")
    
    # =========================================================================
    # Class Methods
    # =========================================================================
    
    @classmethod
    def from_dense(
        cls, 
        A: torch.Tensor, 
        sparse_dim: Tuple[int, int] = (-2, -1)
    ) -> "SparseTensor":
        """
        Create SparseTensor from dense tensor.
        
        Parameters
        ----------
        A : torch.Tensor
            Dense tensor with shape [...batch, M, N, ...block].
        sparse_dim : Tuple[int, int], optional
            Which dimensions are sparse. Default: (-2, -1).
        
        Returns
        -------
        SparseTensor
            Sparse representation of A.
        
        Examples
        --------
        >>> A_dense = torch.randn(3, 3)
        >>> A_dense[A_dense.abs() < 0.5] = 0
        >>> A = SparseTensor.from_dense(A_dense)
        """
        ndim = A.dim()
        dim_m = sparse_dim[0] if sparse_dim[0] >= 0 else ndim + sparse_dim[0]
        dim_n = sparse_dim[1] if sparse_dim[1] >= 0 else ndim + sparse_dim[1]
        
        if ndim == 2 and dim_m == 0 and dim_n == 1:
            A_sparse = A.to_sparse_coo().coalesce()
            indices = A_sparse.indices()
            values = A_sparse.values()
            return cls(values, indices[0], indices[1], tuple(A.shape), sparse_dim=sparse_dim)
        
        perm = [i for i in range(ndim) if i not in (dim_m, dim_n)] + [dim_m, dim_n]
        A_perm = A.permute(*perm)
        batch_shape = A_perm.shape[:-2]
        M, N = A_perm.shape[-2], A_perm.shape[-1]
        A_flat = A_perm.reshape(-1, M, N)
        
        A_2d = A_flat[0].to_sparse_coo()
        indices = A_2d._indices()
        row = indices[0]
        col = indices[1]
        nnz = row.size(0)
        
        values = A_flat[:, row, col]
        if len(batch_shape) > 0:
            values = values.reshape(*batch_shape, nnz)
        else:
            values = values.squeeze(0)
        
        return cls(values, row, col, tuple(A.shape), sparse_dim=sparse_dim)
    
    @classmethod
    def from_torch_sparse(cls, A: torch.Tensor) -> "SparseTensor":
        """
        Create SparseTensor from PyTorch sparse tensor.
        
        Parameters
        ----------
        A : torch.Tensor
            PyTorch sparse COO or CSR tensor (2D only).
        
        Returns
        -------
        SparseTensor
            SparseTensor representation.
        
        Examples
        --------
        >>> A_coo = torch.randn(3, 3).to_sparse_coo()
        >>> A = SparseTensor.from_torch_sparse(A_coo)
        """
        if A.layout == torch.sparse_csr:
            A = A.to_sparse_coo()
        A = A.coalesce()
        indices = A.indices()
        values = A.values()
        return cls(values, indices[0], indices[1], tuple(A.shape))

    @classmethod
    def eye(cls, n: int, dtype: torch.dtype = torch.float64,
            device: Union[str, torch.device] = "cpu") -> "SparseTensor":
        """Sparse identity ``n x n``."""
        idx = torch.arange(n, dtype=torch.int64, device=device)
        return cls(torch.ones(n, dtype=dtype, device=device), idx, idx, shape=(n, n))

    @classmethod
    def diag(cls, values: torch.Tensor,
             device: Optional[Union[str, torch.device]] = None) -> "SparseTensor":
        """Sparse diagonal matrix from a 1-D vector."""
        if values.dim() != 1:
            raise ValueError(f"diag needs a 1-D tensor, got shape {tuple(values.shape)}")
        n = int(values.numel())
        device = device if device is not None else values.device
        idx = torch.arange(n, dtype=torch.int64, device=device)
        return cls(values.to(device), idx, idx, shape=(n, n))

    @classmethod
    def tridiagonal(cls, n: int,
                    diag: Union[float, torch.Tensor] = 2.0,
                    off_diag: Union[float, torch.Tensor] = -1.0,
                    dtype: torch.dtype = torch.float64,
                    device: Union[str, torch.device] = "cpu") -> "SparseTensor":
        """Sparse symmetric tridiagonal ``n x n``. ``diag=4, off=-1`` is the
        canonical SPD test matrix; ``diag=2, off=-1`` is the 1-D Laplacian.
        ``diag`` / ``off_diag`` accept scalars or matching-length tensors."""
        device = torch.device(device)

        def _vec(v, length, name):
            if isinstance(v, torch.Tensor):
                if v.dim() != 1 or v.numel() != length:
                    raise ValueError(f"{name} must have shape ({length},), got {tuple(v.shape)}")
                return v.to(device=device, dtype=dtype)
            return torch.full((length,), float(v), dtype=dtype, device=device)

        diag_v = _vec(diag, n, "diag")
        off_v = _vec(off_diag, n - 1, "off_diag")
        idx = torch.arange(n, dtype=torch.int64, device=device)
        vals = torch.cat([diag_v, off_v, off_v])
        row = torch.cat([idx, idx[1:], idx[:-1]])
        col = torch.cat([idx, idx[:-1], idx[1:]])
        return cls(vals, row, col, shape=(n, n))

    # =========================================================================
    # Properties
    # =========================================================================
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Full tensor shape [...batch, M, N, ...block]."""
        return self._shape
    
    @property
    def sparse_shape(self) -> Tuple[int, int]:
        """The (M, N) sparse matrix dimensions."""
        dim_m, dim_n = self._sparse_dim
        return (self._shape[dim_m], self._shape[dim_n])
    
    @property
    def batch_shape(self) -> Tuple[int, ...]:
        """The batch dimensions before the sparse dimensions."""
        dim_m, dim_n = self._sparse_dim
        min_dim = min(dim_m, dim_n)
        return self._shape[:min_dim]
    
    @property
    def block_shape(self) -> Tuple[int, ...]:
        """The block dimensions after the sparse dimensions."""
        dim_m, dim_n = self._sparse_dim
        max_dim = max(dim_m, dim_n)
        return self._shape[max_dim + 1:]
    
    @property
    def sparse_dim(self) -> Tuple[int, int]:
        """The dimensions that are sparse (M, N)."""
        return self._sparse_dim
    
    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return len(self._shape)
    
    @property
    def nnz(self) -> int:
        """Number of non-zero elements (per batch/block)."""
        return self.row_indices.size(0)
    
    @property
    def dtype(self) -> torch.dtype:
        """Data type of the values."""
        return self.values.dtype
    
    @property
    def device(self) -> torch.device:
        """Device of the tensor."""
        return self.values.device
    
    @property
    def is_cuda(self) -> bool:
        """Whether the tensor is on CUDA."""
        return self.values.is_cuda
    
    @property
    def is_batched(self) -> bool:
        """Whether the tensor has batch dimensions."""
        return len(self.batch_shape) > 0
    
    @property
    def is_block(self) -> bool:
        """Whether the tensor has block dimensions."""
        return len(self.block_shape) > 0
    
    @property
    def batch_size(self) -> int:
        """Total number of batch elements (product of batch_shape)."""
        return math.prod(self.batch_shape) if self.batch_shape else 1
    
    @property
    def is_square(self) -> bool:
        """Whether the sparse dimensions are square (M == N)."""
        M, N = self.sparse_shape
        return M == N
    
    # =========================================================================
    # Device and Type Management
    # =========================================================================
    
    def to(
        self, 
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None
    ) -> "SparseTensor":
        """
        Move tensor to device and/or convert dtype.
        
        Parameters
        ----------
        device : str or torch.device, optional
            Target device (e.g., 'cuda', 'cpu', 'cuda:0').
        dtype : torch.dtype, optional
            Target data type (e.g., torch.float32, torch.float64).
        
        Returns
        -------
        SparseTensor
            New SparseTensor on the target device/dtype.
        
        Examples
        --------
        >>> A = SparseTensor(val, row, col, shape)
        >>> A_cuda = A.to('cuda')
        >>> A_float32 = A.to(dtype=torch.float32)
        >>> A_cuda_float32 = A.to('cuda', torch.float32)
        """
        new_values = self.values
        new_row = self.row_indices
        new_col = self.col_indices
        
        if device is not None:
            new_values = new_values.to(device)
            new_row = new_row.to(device)
            new_col = new_col.to(device)
        
        if dtype is not None:
            new_values = new_values.to(dtype)
        
        result = SparseTensor(
            new_values, new_row, new_col, self._shape,
            sparse_dim=self._sparse_dim
        )
        return result
    
    def cuda(self, device: Optional[int] = None) -> "SparseTensor":
        """
        Move tensor to CUDA device.
        
        Parameters
        ----------
        device : int, optional
            CUDA device index. Default: current device.
        
        Returns
        -------
        SparseTensor
            Tensor on CUDA.
        """
        if device is None:
            return self.to('cuda')
        return self.to(f'cuda:{device}')
    
    def cpu(self) -> "SparseTensor":
        """
        Move tensor to CPU.
        
        Returns
        -------
        SparseTensor
            Tensor on CPU.
        """
        return self.to('cpu')
    
    def float(self) -> "SparseTensor":
        """Convert to float32."""
        return self.to(dtype=torch.float32)
    
    def double(self) -> "SparseTensor":
        """Convert to float64."""
        return self.to(dtype=torch.float64)
    
    def half(self) -> "SparseTensor":
        """Convert to float16."""
        return self.to(dtype=torch.float16)
    
    def to_torch_sparse(self, *args, **kwargs):
        from .convert import to_torch_sparse as _impl
        return _impl(self, *args, **kwargs)

    def to_dense(self, *args, **kwargs):
        from .convert import to_dense as _impl
        return _impl(self, *args, **kwargs)

    def to_csr(self, *args, **kwargs):
        from .convert import to_csr as _impl
        return _impl(self, *args, **kwargs)

    def extract_partition(self, *args, **kwargs):
        from .convert import extract_partition as _impl
        return _impl(self, *args, **kwargs)

    def save_distributed(self, *args, **kwargs):
        from .convert import save_distributed as _impl
        return _impl(self, *args, **kwargs)

    def partition_for_rank(self, *args, **kwargs):
        from .convert import partition_for_rank as _impl
        return _impl(self, *args, **kwargs)

    def detect_matrix_type(self, *args, **kwargs):
        from .convert import detect_matrix_type as _impl
        return _impl(self, *args, **kwargs)

    def T(self, *args, **kwargs):
        from .convert import T as _impl
        return _impl(self, *args, **kwargs)

    def conj(self, *args, **kwargs):
        from .convert import conj as _impl
        return _impl(self, *args, **kwargs)

    def H(self, *args, **kwargs):
        from .convert import H as _impl
        return _impl(self, *args, **kwargs)

    def flatten_blocks(self, *args, **kwargs):
        from .convert import flatten_blocks as _impl
        return _impl(self, *args, **kwargs)

    def unflatten_blocks(self, *args, **kwargs):
        from .convert import unflatten_blocks as _impl
        return _impl(self, *args, **kwargs)

    
    def is_symmetric(self, *args, **kwargs):
        from .structural import is_symmetric as _impl
        return _impl(self, *args, **kwargs)

    def is_hermitian(self, *args, **kwargs):
        from .structural import is_hermitian as _impl
        return _impl(self, *args, **kwargs)

    def is_positive_definite(self, *args, **kwargs):
        from .structural import is_positive_definite as _impl
        return _impl(self, *args, **kwargs)

    def _check_pair_match(self, *args, **kwargs):
        from .structural import _check_pair_match as _impl
        return _impl(self, *args, **kwargs)

    def _check_pd_gershgorin(self, *args, **kwargs):
        from .structural import _check_pd_gershgorin as _impl
        return _impl(self, *args, **kwargs)

    def _check_pd_cholesky(self, *args, **kwargs):
        from .structural import _check_pd_cholesky as _impl
        return _impl(self, *args, **kwargs)

    def _check_pd_eigenvalue(self, *args, **kwargs):
        from .structural import _check_pd_eigenvalue as _impl
        return _impl(self, *args, **kwargs)

    def _batch_indices(self, *args, **kwargs):
        from .structural import _batch_indices as _impl
        return _impl(self, *args, **kwargs)

    def connected_components(self, *args, **kwargs):
        from .graph import connected_components as _impl
        return _impl(self, *args, **kwargs)

    def has_isolated_components(self, *args, **kwargs):
        from .graph import has_isolated_components as _impl
        return _impl(self, *args, **kwargs)

    def to_connected_components(self, *args, **kwargs):
        from .graph import to_connected_components as _impl
        return _impl(self, *args, **kwargs)

    def _spmv_coo(self, *args, **kwargs):
        from .matmul import _spmv_coo as _impl
        return _impl(self, *args, **kwargs)

    def _dense_sparse_mm(self, *args, **kwargs):
        from .matmul import _dense_sparse_mm as _impl
        return _impl(self, *args, **kwargs)

    def _spsp_multiply(self, *args, **kwargs):
        from .matmul import _spsp_multiply as _impl
        return _impl(self, *args, **kwargs)

    def __matmul__(self, *args, **kwargs):
        from .matmul import __matmul__ as _impl
        return _impl(self, *args, **kwargs)

    def __rmatmul__(self, *args, **kwargs):
        from .matmul import __rmatmul__ as _impl
        return _impl(self, *args, **kwargs)

    def solve(self, *args, **kwargs):
        from .linalg import solve as _impl
        return _impl(self, *args, **kwargs)

    def solve_batch(self, *args, **kwargs):
        from .linalg import solve_batch as _impl
        return _impl(self, *args, **kwargs)

    def nonlinear_solve(self, *args, **kwargs):
        from .linalg import nonlinear_solve as _impl
        return _impl(self, *args, **kwargs)

    # =========================================================================
    # Norms
    # =========================================================================
    
    def norm(self, ord: Literal['fro', 1, 2] = 'fro') -> torch.Tensor:
        """
        Compute matrix norm.
        
        For batched tensors, returns norm for each batch element.
        
        Parameters
        ----------
        ord : {'fro', 1, 2}, optional
            Norm type:
            - 'fro': Frobenius norm (default)
            - 1: Maximum absolute column sum
            - 2: Spectral norm (largest singular value)
            
        Returns
        -------
        torch.Tensor
            Norm value(s). Shape [] for non-batched, [*batch_shape] for batched.
        
        Examples
        --------
        >>> A = SparseTensor(val, row, col, (3, 3))
        >>> A.norm('fro')  # tensor(5.0)
        
        >>> A_batch = SparseTensor(val_batch, row, col, (4, 3, 3))
        >>> A_batch.norm('fro')  # tensor([5.0, 5.0, 5.0, 5.0])
        """
        if self.is_batched:
            batch_shape = self.batch_shape
            vals_flat = self.values.reshape(-1, self.nnz)
            norms = []
            for i in range(vals_flat.size(0)):
                if ord == 'fro':
                    norms.append(vals_flat[i].norm())
                else:
                    idx = self._flat_to_batch_idx(i)
                    A_dense = self.to_dense(idx)
                    norms.append(torch.linalg.norm(A_dense, ord=ord))
            return torch.stack(norms).reshape(*batch_shape)
        else:
            if ord == 'fro':
                return self.values.norm()
            if self.is_cuda or not is_scipy_available():
                A = self.to_dense()
                return torch.linalg.norm(A, ord=ord)
            M, N = self.sparse_shape
            return scipy_norm(self.values, self.row_indices, self.col_indices, (M, N), ord=ord)
    
    def _flat_to_batch_idx(self, flat_idx: int) -> Tuple[int, ...]:
        """Convert flat batch index to tuple."""
        idx = []
        for s in reversed(self.batch_shape):
            idx.append(flat_idx % s)
            flat_idx //= s
        return tuple(reversed(idx))
    
    def spy(self, *args, **kwargs):
        """Render the sparsity pattern. See :func:`viz.spy`."""
        from .viz import spy as _spy
        return _spy(self, *args, **kwargs)

    def eigs(self, *args, **kwargs):
        from .linalg import eigs as _impl
        return _impl(self, *args, **kwargs)

    def eigsh(self, *args, **kwargs):
        from .linalg import eigsh as _impl
        return _impl(self, *args, **kwargs)

    def svd(self, *args, **kwargs):
        from .linalg import svd as _impl
        return _impl(self, *args, **kwargs)

    def condition_number(self, *args, **kwargs):
        from .linalg import condition_number as _impl
        return _impl(self, *args, **kwargs)

    def det(self, *args, **kwargs):
        from .linalg import det as _impl
        return _impl(self, *args, **kwargs)

    def lu(self, *args, **kwargs):
        from .linalg import lu as _impl
        return _impl(self, *args, **kwargs)

    # =========================================================================
    # String Representation
    # =========================================================================
    
    def __repr__(self) -> str:
        parts = [f"SparseTensor(shape={self._shape}"]
        if self.is_batched:
            parts.append(f"batch={self.batch_shape}")
        parts.append(f"sparse={self.sparse_shape}")
        if self.is_block:
            parts.append(f"block={self.block_shape}")
        parts.append(f"nnz={self.nnz}")
        parts.append(f"dtype={self.dtype}")
        parts.append(f"device={self.device}")
        return ", ".join(parts) + ")"
    
    def sum(self, *args, **kwargs):
        from .reductions import _sum_impl as _impl
        return _impl(self, *args, **kwargs)

    def _sum_over_sparse(self, *args, **kwargs):
        from .reductions import _sum_over_sparse as _impl
        return _impl(self, *args, **kwargs)

    def _sum_over_batch_block(self, *args, **kwargs):
        from .reductions import _sum_over_batch_block as _impl
        return _impl(self, *args, **kwargs)

    def mean(self, *args, **kwargs):
        from .reductions import _mean_impl as _impl
        return _impl(self, *args, **kwargs)

    def prod(self, *args, **kwargs):
        from .reductions import _prod_impl as _impl
        return _impl(self, *args, **kwargs)

    def max(self, *args, **kwargs):
        from .reductions import _max_impl as _impl
        return _impl(self, *args, **kwargs)

    def min(self, *args, **kwargs):
        from .reductions import _min_impl as _impl
        return _impl(self, *args, **kwargs)

    def _normalize_axis(self, *args, **kwargs):
        from .reductions import _normalize_axis as _impl
        return _impl(self, *args, **kwargs)

    def _get_dim_type(self, *args, **kwargs):
        from .reductions import _get_dim_type as _impl
        return _impl(self, *args, **kwargs)

    def _values_axis_for_dim(self, *args, **kwargs):
        from .reductions import _values_axis_for_dim as _impl
        return _impl(self, *args, **kwargs)

    def _apply_elementwise(self, *args, **kwargs):
        from .ops import _apply_elementwise as _impl
        return _impl(self, *args, **kwargs)

    def __add__(self, *args, **kwargs):
        from .ops import __add__ as _impl
        return _impl(self, *args, **kwargs)

    def __radd__(self, *args, **kwargs):
        from .ops import __radd__ as _impl
        return _impl(self, *args, **kwargs)

    def __sub__(self, *args, **kwargs):
        from .ops import __sub__ as _impl
        return _impl(self, *args, **kwargs)

    def __rsub__(self, *args, **kwargs):
        from .ops import __rsub__ as _impl
        return _impl(self, *args, **kwargs)

    def __mul__(self, *args, **kwargs):
        from .ops import __mul__ as _impl
        return _impl(self, *args, **kwargs)

    def __rmul__(self, *args, **kwargs):
        from .ops import __rmul__ as _impl
        return _impl(self, *args, **kwargs)

    def __truediv__(self, *args, **kwargs):
        from .ops import __truediv__ as _impl
        return _impl(self, *args, **kwargs)

    def __rtruediv__(self, *args, **kwargs):
        from .ops import __rtruediv__ as _impl
        return _impl(self, *args, **kwargs)

    def __floordiv__(self, *args, **kwargs):
        from .ops import __floordiv__ as _impl
        return _impl(self, *args, **kwargs)

    def __pow__(self, *args, **kwargs):
        from .ops import __pow__ as _impl
        return _impl(self, *args, **kwargs)

    def __neg__(self, *args, **kwargs):
        from .ops import __neg__ as _impl
        return _impl(self, *args, **kwargs)

    def __pos__(self, *args, **kwargs):
        from .ops import __pos__ as _impl
        return _impl(self, *args, **kwargs)

    def __abs__(self, *args, **kwargs):
        from .ops import __abs__ as _impl
        return _impl(self, *args, **kwargs)

    def abs(self, *args, **kwargs):
        from .ops import _abs_impl as _impl
        return _impl(self, *args, **kwargs)

    def sqrt(self, *args, **kwargs):
        from .ops import _sqrt_impl as _impl
        return _impl(self, *args, **kwargs)

    def square(self, *args, **kwargs):
        from .ops import _square_impl as _impl
        return _impl(self, *args, **kwargs)

    def exp(self, *args, **kwargs):
        from .ops import _exp_impl as _impl
        return _impl(self, *args, **kwargs)

    def log(self, *args, **kwargs):
        from .ops import _log_impl as _impl
        return _impl(self, *args, **kwargs)

    # =========================================================================
    # Persistence (I/O)
    # =========================================================================
    
    def save(
        self,
        path: Union[str, "os.PathLike"],
        metadata: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Save SparseTensor to safetensors format.
        
        Parameters
        ----------
        path : str or PathLike
            Output file path (should end with .safetensors).
        metadata : dict, optional
            Additional metadata to store.
        
        Example
        -------
        >>> A = SparseTensor(val, row, col, (100, 100))
        >>> A.save("matrix.safetensors")
        """
        from ..io import save_sparse
        save_sparse(self, path, metadata)
    
    @classmethod
    def load(
        cls,
        path: Union[str, "os.PathLike"],
        device: Union[str, torch.device] = "cpu"
    ) -> "SparseTensor":
        """
        Load SparseTensor from safetensors format.
        
        Parameters
        ----------
        path : str or PathLike
            Input file path.
        device : str or torch.device
            Device to load tensors to.
        
        Returns
        -------
        SparseTensor
            The loaded sparse tensor.
        
        Example
        -------
        >>> A = SparseTensor.load("matrix.safetensors", device="cuda")
        """
        from ..io import load_sparse
        return load_sparse(path, device)


# =============================================================================
# LUFactorization Class
# =============================================================================

class LUFactorization:
    """
    LU factorization wrapper for efficient repeated solves.
    
    Created by SparseTensor.lu().
    
    Parameters
    ----------
    lu_factor : scipy.sparse.linalg.SuperLU
        The SciPy LU factorization object.
    shape : Tuple[int, int]
        Matrix shape.
    dtype : torch.dtype
        Data type.
    device : torch.device
        Device.
    
    Examples
    --------
    >>> A = SparseTensor(val, row, col, (10, 10))
    >>> lu = A.lu()
    >>> x1 = lu.solve(b1)  # First solve
    >>> x2 = lu.solve(b2)  # Much faster - reuses factorization
    """
    
    def __init__(self, lu_factor, shape: Tuple[int, int], dtype: torch.dtype, device: torch.device):
        self._lu = lu_factor
        self._shape = shape
        self._dtype = dtype
        self._device = device
    
    def solve(self, b: torch.Tensor) -> torch.Tensor:
        """
        Solve Ax = b using the cached factorization.
        
        Parameters
        ----------
        b : torch.Tensor
            Right-hand side vector.
        
        Returns
        -------
        torch.Tensor
            Solution x.
        """
        import numpy as np
        b_np = b.detach().cpu().numpy()
        x_np = self._lu.solve(b_np)
        return torch.from_numpy(x_np).to(dtype=self._dtype, device=self._device)
    
    def __repr__(self) -> str:
        return f"LUFactorization(shape={self._shape})"


# =============================================================================
# SparseTensorList Class
# =============================================================================

class SparseTensorList:
    """
    A list of SparseTensors with different structures.
    
    Provides a unified interface for batch operations on matrices
    with different sparsity patterns. Unlike batched SparseTensor
    (which requires same structure), SparseTensorList allows
    each matrix to have different shape and sparsity pattern.
        
        Parameters
        ----------
    tensors : List[SparseTensor]
        List of SparseTensor objects.
    
    Attributes
    ----------
    shapes : List[Tuple[int, ...]]
        List of shapes for each tensor.
    device : torch.device
        Device (from first tensor).
    dtype : torch.dtype
        Data type (from first tensor).
    
    Examples
    --------
    >>> # Create matrices with different sizes
    >>> A1 = SparseTensor(val1, row1, col1, (10, 10))
    >>> A2 = SparseTensor(val2, row2, col2, (20, 20))
    >>> A3 = SparseTensor(val3, row3, col3, (30, 30))
    
    >>> # Create list
    >>> matrices = SparseTensorList([A1, A2, A3])
    >>> print(matrices.shapes)  # [(10, 10), (20, 20), (30, 30)]
    
    >>> # Batch solve
    >>> x_list = matrices.solve([b1, b2, b3])
    
    >>> # Check properties for all
    >>> is_sym = matrices.is_symmetric()  # [tensor(True), tensor(True), tensor(True)]
    """
    
    def __init__(self, tensors: List["SparseTensor"]):
        if not tensors:
            raise ValueError("SparseTensorList cannot be empty")
        self._tensors = list(tensors)
    
    @classmethod
    def from_coo_list(
        cls,
        matrices: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[int, ...]]],
    ) -> "SparseTensorList":
        """
        Create from list of COO data tuples.
        
        Parameters
        ----------
        matrices : List[Tuple]
            List of (values, row_indices, col_indices, shape) tuples.
        
        Returns
        -------
        SparseTensorList
            List of SparseTensors.
        
        Examples
        --------
        >>> data = [
        ...     (val1, row1, col1, (10, 10)),
        ...     (val2, row2, col2, (20, 20)),
        ... ]
        >>> matrices = SparseTensorList.from_coo_list(data)
        """
        tensors = [
            SparseTensor(val, row, col, shape)
            for val, row, col, shape in matrices
        ]
        return cls(tensors)
    
    @classmethod
    def from_torch_sparse_list(cls, A_list: List[torch.Tensor]) -> "SparseTensorList":
        """
        Create from list of PyTorch sparse tensors.
        
        Parameters
        ----------
        A_list : List[torch.Tensor]
            List of PyTorch sparse COO tensors.
        
        Returns
        -------
        SparseTensorList
            List of SparseTensors.
        """
        tensors = [SparseTensor.from_torch_sparse(A) for A in A_list]
        return cls(tensors)
    
    @property
    def shapes(self) -> List[Tuple[int, ...]]:
        """List of shapes for each tensor."""
        return [t.shape for t in self._tensors]
    
    @property
    def device(self) -> torch.device:
        """Device of the first tensor."""
        return self._tensors[0].device
    
    @property
    def dtype(self) -> torch.dtype:
        """Data type of the first tensor."""
        return self._tensors[0].dtype
    
    def __len__(self) -> int:
        """Number of tensors in the list."""
        return len(self._tensors)
    
    def __getitem__(self, idx: int) -> "SparseTensor":
        """
        Get tensor by index.
        
        Parameters
        ----------
        idx : int
            Index (supports negative indexing).
        
        Returns
        -------
        SparseTensor
            The tensor at that index.
        """
        if idx < 0:
            idx = len(self._tensors) + idx
        return self._tensors[idx]
    
    def __iter__(self):
        """Iterate over tensors."""
        return iter(self._tensors)
    
    def to(self, device: Union[str, torch.device]) -> "SparseTensorList":
        """
        Move all tensors to device.
        
        Parameters
        ----------
        device : str or torch.device
            Target device.
        
        Returns
        -------
        SparseTensorList
            New list with tensors on target device.
        """
        return SparseTensorList([t.to(device) for t in self._tensors])
    
    def cuda(self) -> "SparseTensorList":
        """Move all tensors to CUDA."""
        return self.to('cuda')
    
    def cpu(self) -> "SparseTensorList":
        """Move all tensors to CPU."""
        return self.to('cpu')
    
    # =========================================================================
    # Arithmetic Operations
    # =========================================================================
    
    def __matmul__(self, x_list: Union[List[torch.Tensor], torch.Tensor]) -> List[torch.Tensor]:
        """
        Batch matrix-vector/matrix multiplication.
        
        Parameters
        ----------
        x_list : List[torch.Tensor] or torch.Tensor
            If List: one vector/matrix per sparse tensor, each with compatible shape.
            If Tensor: broadcasted to all matrices (must have compatible shape for all).
        
        Returns
        -------
        List[torch.Tensor]
            List of results [A1 @ x1, A2 @ x2, ...] or [A1 @ x, A2 @ x, ...]
            
        Examples
        --------
        >>> matrices = SparseTensorList([A1, A2, A3])
        >>> # Per-matrix vectors
        >>> y_list = matrices @ [x1, x2, x3]
        >>> # Broadcast same vector
        >>> y_list = matrices @ x  # x applied to all
        """
        if isinstance(x_list, torch.Tensor):
            # Broadcast same tensor to all
            return [t @ x_list for t in self._tensors]
        
        if len(x_list) != len(self._tensors):
            raise ValueError(f"Expected {len(self._tensors)} vectors, got {len(x_list)}")
        return [t @ x for t, x in zip(self._tensors, x_list)]
    
    def __add__(self, other: Union["SparseTensorList", float, int]) -> "SparseTensorList":
        """
        Element-wise addition.
        
        Parameters
        ----------
        other : SparseTensorList or scalar
            If SparseTensorList: add corresponding matrices (must have same length).
            If scalar: add to all matrices.
            
        Returns
        -------
        SparseTensorList
            Result of addition.
        """
        if isinstance(other, SparseTensorList):
            if len(other) != len(self._tensors):
                raise ValueError(f"Length mismatch: {len(self._tensors)} vs {len(other)}")
            return SparseTensorList([a + b for a, b in zip(self._tensors, other._tensors)])
        # Scalar addition - add to values
        return SparseTensorList([
            SparseTensor(t.values + other, t.row_indices, t.col_indices, t.shape)
            for t in self._tensors
        ])
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other: Union["SparseTensorList", float, int]) -> "SparseTensorList":
        """Element-wise subtraction."""
        if isinstance(other, SparseTensorList):
            if len(other) != len(self._tensors):
                raise ValueError(f"Length mismatch: {len(self._tensors)} vs {len(other)}")
            return SparseTensorList([a - b for a, b in zip(self._tensors, other._tensors)])
        return SparseTensorList([
            SparseTensor(t.values - other, t.row_indices, t.col_indices, t.shape)
            for t in self._tensors
        ])
    
    def __rsub__(self, other):
        return SparseTensorList([
            SparseTensor(other - t.values, t.row_indices, t.col_indices, t.shape)
            for t in self._tensors
        ])
    
    def __mul__(self, other: Union["SparseTensorList", float, int, torch.Tensor]) -> "SparseTensorList":
        """
        Element-wise multiplication.
        
        Parameters
        ----------
        other : SparseTensorList, scalar, or Tensor
            If SparseTensorList: multiply corresponding matrices element-wise.
            If scalar/Tensor: multiply all values.
            
        Returns
        -------
        SparseTensorList
            Result of multiplication.
        """
        if isinstance(other, SparseTensorList):
            if len(other) != len(self._tensors):
                raise ValueError(f"Length mismatch: {len(self._tensors)} vs {len(other)}")
            return SparseTensorList([a * b for a, b in zip(self._tensors, other._tensors)])
        return SparseTensorList([t * other for t in self._tensors])
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other: Union[float, int, torch.Tensor]) -> "SparseTensorList":
        """Element-wise division by scalar."""
        return SparseTensorList([t / other for t in self._tensors])
    
    def __neg__(self) -> "SparseTensorList":
        """Negate all values."""
        return SparseTensorList([-t for t in self._tensors])
    
    def sum(self, axis: Optional[int] = None) -> Union[List[torch.Tensor], torch.Tensor]:
        """
        Sum values in each matrix.
        
        Parameters
        ----------
        axis : int, optional
            If None: sum all values in each matrix, return List[scalar].
            If 0: sum over rows for each matrix.
            If 1: sum over columns for each matrix.
            
        Returns
        -------
        List[torch.Tensor] or torch.Tensor
            If axis is None: List of scalar tensors (one per matrix).
            If axis is 0 or 1: List of 1D tensors.
            
        Examples
        --------
        >>> matrices = SparseTensorList([A1, A2, A3])
        >>> totals = matrices.sum()  # [sum(A1), sum(A2), sum(A3)]
        >>> row_sums = matrices.sum(axis=1)  # [A1.sum(1), A2.sum(1), ...]
        """
        return [t.sum(axis=axis) for t in self._tensors]
    
    def mean(self, axis: Optional[int] = None) -> List[torch.Tensor]:
        """
        Mean of values in each matrix.
        
        Parameters
        ----------
        axis : int, optional
            Same as sum().
            
        Returns
        -------
        List[torch.Tensor]
            List of mean values/vectors.
        """
        return [t.mean(axis=axis) for t in self._tensors]
    
    def max(self) -> List[torch.Tensor]:
        """Maximum value in each matrix."""
        return [t.max() for t in self._tensors]
    
    def min(self) -> List[torch.Tensor]:
        """Minimum value in each matrix."""
        return [t.min() for t in self._tensors]
    
    def abs(self) -> "SparseTensorList":
        """Absolute value of all elements."""
        return SparseTensorList([t.abs() for t in self._tensors])
    
    def clamp(self, min: Optional[float] = None, max: Optional[float] = None) -> "SparseTensorList":
        """Clamp values in all matrices."""
        return SparseTensorList([t.clamp(min=min, max=max) for t in self._tensors])
    
    def pow(self, exponent: float) -> "SparseTensorList":
        """Element-wise power."""
        return SparseTensorList([t.pow(exponent) for t in self._tensors])
    
    def sqrt(self) -> "SparseTensorList":
        """Element-wise square root."""
        return SparseTensorList([t.sqrt() for t in self._tensors])
    
    def exp(self) -> "SparseTensorList":
        """Element-wise exponential."""
        return SparseTensorList([t.exp() for t in self._tensors])
    
    def log(self) -> "SparseTensorList":
        """Element-wise natural logarithm."""
        return SparseTensorList([t.log() for t in self._tensors])
    
    # =========================================================================
    # Linear Algebra
    # =========================================================================
    
    def solve(self, b_list: List[torch.Tensor], **kwargs) -> List[torch.Tensor]:
        """
        Solve linear systems for all matrices.
        
        Parameters
        ----------
        b_list : List[torch.Tensor]
            List of right-hand side vectors, one per matrix.
        **kwargs
            Additional arguments passed to SparseTensor.solve().
        
        Returns
        -------
        List[torch.Tensor]
            List of solutions.
        
        Examples
        --------
        >>> matrices = SparseTensorList([A1, A2, A3])
        >>> x_list = matrices.solve([b1, b2, b3])
        """
        if len(b_list) != len(self._tensors):
            raise ValueError(f"Expected {len(self._tensors)} RHS vectors, got {len(b_list)}")
        return [t.solve(b, **kwargs) for t, b in zip(self._tensors, b_list)]
    
    def is_symmetric(self, **kwargs) -> List[torch.Tensor]:
        """
        Check symmetry for all matrices.
        
        Parameters
        ----------
        **kwargs
            Arguments passed to SparseTensor.is_symmetric().
        
        Returns
        -------
        List[torch.Tensor]
            List of boolean tensors.
        """
        return [t.is_symmetric(**kwargs) for t in self._tensors]
    
    def is_positive_definite(self, **kwargs) -> List[torch.Tensor]:
        """
        Check positive definiteness for all matrices.
        
        Parameters
        ----------
        **kwargs
            Arguments passed to SparseTensor.is_positive_definite().
        
        Returns
        -------
        List[torch.Tensor]
            List of boolean tensors.
        """
        return [t.is_positive_definite(**kwargs) for t in self._tensors]
    
    def norm(self, ord: Literal['fro', 1, 2] = 'fro') -> List[torch.Tensor]:
        """
        Compute norms for all matrices.
        
        Parameters
        ----------
        ord : {'fro', 1, 2}
            Norm type.
        
        Returns
        -------
        List[torch.Tensor]
            List of norm values.
        """
        return [t.norm(ord=ord) for t in self._tensors]
    
    def eigs(self, k: int = 6, **kwargs) -> List[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """
        Compute eigenvalues for all matrices.
        
        Parameters
        ----------
        k : int
            Number of eigenvalues.
        **kwargs
            Additional arguments.
        
        Returns
        -------
        List[Tuple[torch.Tensor, Optional[torch.Tensor]]]
            List of (eigenvalues, eigenvectors) tuples.
        """
        return [t.eigs(k=k, **kwargs) for t in self._tensors]
    
    def eigsh(self, k: int = 6, **kwargs) -> List[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """
        Compute eigenvalues for symmetric matrices.
        
        Parameters
        ----------
        k : int
            Number of eigenvalues.
        **kwargs
            Additional arguments.
        
        Returns
        -------
        List[Tuple[torch.Tensor, Optional[torch.Tensor]]]
            List of (eigenvalues, eigenvectors) tuples.
        """
        return [t.eigsh(k=k, **kwargs) for t in self._tensors]
    
    def svd(self, k: int = 6) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Compute SVD for all matrices.
        
        Parameters
        ----------
        k : int
            Number of singular values.
        
        Returns
        -------
        List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
            List of (U, S, Vt) tuples.
        """
        return [t.svd(k=k) for t in self._tensors]
    
    def condition_number(self, ord: int = 2) -> List[torch.Tensor]:
        """
        Compute condition numbers for all matrices.
        
        Parameters
        ----------
        ord : int
            Norm order.
        
        Returns
        -------
        List[torch.Tensor]
            List of condition numbers.
        """
        return [t.condition_number(ord=ord) for t in self._tensors]
    
    def det(self) -> List[torch.Tensor]:
        """
        Compute determinants for all matrices.
        
        Returns
        -------
        List[torch.Tensor]
            List of determinant values.
            
        Examples
        --------
        >>> matrices = SparseTensorList([A1, A2, A3])
        >>> dets = matrices.det()
        >>> print([d.item() for d in dets])
        """
        return [t.det() for t in self._tensors]
    
    def spy(
        self,
        indices: Optional[List[int]] = None,
        ncols: int = 3,
        figsize: Optional[Tuple[float, float]] = None,
        **kwargs
    ):
        """
        Visualize sparsity patterns for multiple matrices in a grid.
        
        Parameters
        ----------
        indices : List[int], optional
            Which matrices to visualize. Default: all.
        ncols : int, optional
            Number of columns in subplot grid. Default: 3.
        figsize : Tuple[float, float], optional
            Figure size. Auto-computed if None.
        **kwargs
            Additional arguments passed to SparseTensor.spy().
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
            
        Examples
        --------
        >>> matrices = SparseTensorList([A1, A2, A3, A4])
        >>> matrices.spy()  # Visualize all in grid
        >>> matrices.spy(indices=[0, 2])  # Visualize specific ones
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for spy(). Install with: pip install matplotlib")
        
        if indices is None:
            indices = list(range(len(self._tensors)))
        
        n = len(indices)
        nrows = (n + ncols - 1) // ncols
        
        if figsize is None:
            figsize = (4 * ncols, 4 * nrows)
        
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        
        for i, idx in enumerate(indices):
            row, col = i // ncols, i % ncols
            ax = axes[row, col]
            self._tensors[idx].spy(ax=ax, show_colorbar=False, **kwargs)
            M, N = self._tensors[idx].sparse_shape
            ax.set_title(f'[{idx}] {M}×{N}, nnz={self._tensors[idx].nnz:,}', fontsize=10)
        
        # Hide unused subplots
        for i in range(n, nrows * ncols):
            row, col = i // ncols, i % ncols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        return fig
    
    # =========================================================================
    # Conversion Methods
    # =========================================================================
    
    def to_block_diagonal(self) -> "SparseTensor":
        """
        Merge all matrices into a single block-diagonal SparseTensor.
        
        Creates a sparse matrix where each input matrix appears as a block
        on the diagonal: diag(A1, A2, ..., An).
        
        Returns
        -------
        SparseTensor
            Block-diagonal matrix with shape (sum(M_i), sum(N_i)).
            
        Notes
        -----
        The resulting matrix has the structure:
        
        ```
        [A1  0  0  ...]
        [ 0 A2  0  ...]
        [ 0  0 A3  ...]
        [... ... ... ]
        ```
        
        Examples
        --------
        >>> A1 = SparseTensor(val1, row1, col1, (10, 10))
        >>> A2 = SparseTensor(val2, row2, col2, (20, 20))
        >>> stl = SparseTensorList([A1, A2])
        >>> A_block = stl.to_block_diagonal()  # Shape (30, 30)
        """
        if len(self._tensors) == 0:
            raise ValueError("Cannot convert empty SparseTensorList to block diagonal")
        
        if len(self._tensors) == 1:
            return self._tensors[0]
        
        # Compute offsets
        row_offsets = [0]
        col_offsets = [0]
        
        for t in self._tensors:
            M, N = t.sparse_shape
            row_offsets.append(row_offsets[-1] + M)
            col_offsets.append(col_offsets[-1] + N)
        
        total_rows = row_offsets[-1]
        total_cols = col_offsets[-1]
        
        # Concatenate all COO data with offsets
        all_values = []
        all_rows = []
        all_cols = []
        
        for i, t in enumerate(self._tensors):
            all_values.append(t.values)
            all_rows.append(t.row_indices + row_offsets[i])
            all_cols.append(t.col_indices + col_offsets[i])
        
        values = torch.cat(all_values)
        rows = torch.cat(all_rows)
        cols = torch.cat(all_cols)
        
        return SparseTensor(values, rows, cols, (total_rows, total_cols))
    
    @classmethod
    def from_block_diagonal(
        cls,
        sparse: "SparseTensor",
        sizes: List[Tuple[int, int]]
    ) -> "SparseTensorList":
        """
        Split a block-diagonal SparseTensor into a list of matrices.
        
        Parameters
        ----------
        sparse : SparseTensor
            Block-diagonal matrix to split.
        sizes : List[Tuple[int, int]]
            List of (rows, cols) for each block. Must sum to sparse.shape.
            
        Returns
        -------
        SparseTensorList
            List of extracted blocks.
            
        Examples
        --------
        >>> A_block = SparseTensor(val, row, col, (30, 30))
        >>> stl = SparseTensorList.from_block_diagonal(A_block, [(10, 10), (20, 20)])
        >>> print(len(stl))  # 2
        """
        if sparse.is_batched:
            raise NotImplementedError("from_block_diagonal not supported for batched tensors")
        
        # Validate sizes
        total_rows = sum(s[0] for s in sizes)
        total_cols = sum(s[1] for s in sizes)
        
        if (total_rows, total_cols) != sparse.sparse_shape:
            raise ValueError(
                f"Sizes sum to ({total_rows}, {total_cols}) but sparse has shape {sparse.sparse_shape}"
            )
        
        # Compute offsets
        row_offsets = [0]
        col_offsets = [0]
        for m, n in sizes:
            row_offsets.append(row_offsets[-1] + m)
            col_offsets.append(col_offsets[-1] + n)
        
        tensors = []
        row = sparse.row_indices
        col = sparse.col_indices
        val = sparse.values
        
        for i, (m, n) in enumerate(sizes):
            r_start, r_end = row_offsets[i], row_offsets[i + 1]
            c_start, c_end = col_offsets[i], col_offsets[i + 1]
            
            # Find entries in this block
            mask = (row >= r_start) & (row < r_end) & (col >= c_start) & (col < c_end)
            
            block_row = row[mask] - r_start
            block_col = col[mask] - c_start
            block_val = val[mask]
            
            tensors.append(SparseTensor(block_val, block_row, block_col, (m, n)))
        
        return cls(tensors)
    
    @property
    def block_sizes(self) -> List[Tuple[int, int]]:
        """
        Get the (rows, cols) size of each matrix.
        
        Returns
        -------
        List[Tuple[int, int]]
            List of (M, N) tuples.
        """
        return [t.sparse_shape for t in self._tensors]
    
    @property
    def total_nnz(self) -> int:
        """Total number of non-zeros across all matrices."""
        return sum(t.nnz for t in self._tensors)
    
    @property
    def total_shape(self) -> Tuple[int, int]:
        """Shape of the block-diagonal representation."""
        total_rows = sum(t.sparse_shape[0] for t in self._tensors)
        total_cols = sum(t.sparse_shape[1] for t in self._tensors)
        return (total_rows, total_cols)
    
    def __repr__(self) -> str:
        return f"SparseTensorList(n={len(self._tensors)}, device={self.device})"
