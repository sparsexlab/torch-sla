"""Reductions for SparseTensor (sum / mean / prod / max / min / norm)."""
from __future__ import annotations
import warnings
from typing import Optional, Tuple, Union, Literal
import torch

from .core import SparseTensor  # noqa: E402  # cross-module use


def _normalize_axis(self, axis: Optional[Union[int, Tuple[int, ...]]]) -> Optional[Tuple[int, ...]]:
    """Normalize axis to tuple of positive indices."""
    if axis is None:
        return None
    if isinstance(axis, int):
        axis = (axis,)
    ndim = len(self._shape)
    return tuple(a if a >= 0 else ndim + a for a in axis)

def _get_dim_type(self, dim: int) -> str:
    """Get the type of dimension: 'batch', 'sparse_m', 'sparse_n', or 'block'."""
    dim_m, dim_n = self._sparse_dim
    min_sparse = min(dim_m, dim_n)
    max_sparse = max(dim_m, dim_n)
    
    if dim < min_sparse:
        return 'batch'
    elif dim == dim_m:
        return 'sparse_m'
    elif dim == dim_n:
        return 'sparse_n'
    else:
        return 'block'

def _values_axis_for_dim(self, dim: int) -> int:
    """
    Map tensor dimension to values tensor dimension.
    
    Values shape: [...batch, nnz, ...block]
    Tensor shape: [...batch, M, N, ...block]
    """
    dim_m, dim_n = self._sparse_dim
    min_sparse = min(dim_m, dim_n)
    max_sparse = max(dim_m, dim_n)
    
    if dim < min_sparse:
        # Batch dimension - same position
        return dim
    elif dim == dim_m or dim == dim_n:
        # Sparse dimension - maps to nnz axis
        return min_sparse  # nnz is at the position of first sparse dim
    else:
        # Block dimension - after nnz axis
        # Shift by -1 because we replaced (M, N) with (nnz,)
        return dim - 1

def _sum_impl(
    self, 
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdim: bool = False
) -> Union[torch.Tensor, "SparseTensor"]:
    """
    Sum of sparse tensor elements over specified axis.
    
    Parameters
    ----------
    axis : int, tuple of ints, or None
        Axis or axes along which to sum. Axes correspond to:
        - Batch dimensions: [...batch] at the beginning
        - Sparse dimensions: (M, N) at sparse_dim positions
        - Block dimensions: [...block] at the end
        
        If None, sum over all elements (returns scalar tensor).
    keepdim : bool
        Whether to keep the reduced dimensions.
        
    Returns
    -------
    torch.Tensor or SparseTensor
        - If reducing over sparse dimensions: returns dense tensor
        - If reducing over batch/block dimensions only: returns SparseTensor
        - If axis=None: returns scalar tensor
    
    Examples
    --------
    >>> # Shape: [batch=2, M=10, N=10, block=3]
    >>> A = SparseTensor(val, row, col, (2, 10, 10, 3))
    >>> 
    >>> A.sum()           # Scalar: sum all elements
    >>> A.sum(axis=0)     # Sum over batch -> [10, 10, 3]
    >>> A.sum(axis=1)     # Sum over M (rows) -> [2, 10, 3] (dense)
    >>> A.sum(axis=2)     # Sum over N (cols) -> [2, 10, 3] (dense)
    >>> A.sum(axis=3)     # Sum over block -> SparseTensor [2, 10, 10]
    >>> A.sum(axis=(1,2)) # Sum over M and N -> [2, 3] (dense)
    """
    if axis is None:
        # Sum over all elements
        return self.values.sum()
    
    axes = self._normalize_axis(axis)
    dim_types = [self._get_dim_type(d) for d in axes]
    
    # Check if we're reducing over sparse dimensions
    has_sparse_reduction = any(dt in ('sparse_m', 'sparse_n') for dt in dim_types)
    
    if has_sparse_reduction:
        # Need to convert to dense for sparse reduction
        return self._sum_over_sparse(axes, keepdim)
    else:
        # Only batch/block reduction - can stay sparse
        return self._sum_over_batch_block(axes, keepdim)

def _sum_over_sparse(
    self, 
    axes: Tuple[int, ...], 
    keepdim: bool
) -> torch.Tensor:
    """Sum that involves sparse dimensions - returns dense."""
    M, N = self.sparse_shape
    dim_m, dim_n = self._sparse_dim
    row, col = self.row_indices, self.col_indices
    
    # Separate sparse and non-sparse axes
    sparse_axes = [a for a in axes if self._get_dim_type(a) in ('sparse_m', 'sparse_n')]
    other_axes = [a for a in axes if self._get_dim_type(a) not in ('sparse_m', 'sparse_n')]
    
    reduce_m = dim_m in axes
    reduce_n = dim_n in axes
    
    if self.is_batched:
        B = self.batch_size
        batch_shape = self.batch_shape
        vals_flat = self.values.reshape(B, self.nnz, *self.block_shape) if self.is_block else self.values.reshape(B, self.nnz)
        
        if reduce_m and reduce_n:
            # Sum all sparse entries per batch
            result = vals_flat.sum(dim=1)  # [B, *block]
            result = result.reshape(*batch_shape, *self.block_shape) if self.is_block else result.reshape(*batch_shape)
        elif reduce_m:
            # Sum over rows -> result is [B, N, *block]
            result = torch.zeros(B, N, *self.block_shape, dtype=self.dtype, device=self.device)
            col_idx = col.unsqueeze(0).expand(B, -1)
            if self.is_block:
                for i in range(B):
                    result[i].scatter_add_(0, col_idx[i].unsqueeze(-1).expand(-1, *self.block_shape), vals_flat[i])
            else:
                result.scatter_add_(1, col_idx, vals_flat)
            result = result.reshape(*batch_shape, N, *self.block_shape) if self.is_block else result.reshape(*batch_shape, N)
        else:  # reduce_n
            # Sum over cols -> result is [B, M, *block]
            result = torch.zeros(B, M, *self.block_shape, dtype=self.dtype, device=self.device)
            row_idx = row.unsqueeze(0).expand(B, -1)
            if self.is_block:
                for i in range(B):
                    result[i].scatter_add_(0, row_idx[i].unsqueeze(-1).expand(-1, *self.block_shape), vals_flat[i])
            else:
                result.scatter_add_(1, row_idx, vals_flat)
            result = result.reshape(*batch_shape, M, *self.block_shape) if self.is_block else result.reshape(*batch_shape, M)
    else:
        vals = self.values
        
        if reduce_m and reduce_n:
            result = vals.sum(dim=0) if self.is_block else vals.sum()
        elif reduce_m:
            result = torch.zeros(N, *self.block_shape, dtype=self.dtype, device=self.device) if self.is_block else torch.zeros(N, dtype=self.dtype, device=self.device)
            if self.is_block:
                result.scatter_add_(0, col.unsqueeze(-1).expand(-1, *self.block_shape), vals)
            else:
                result.scatter_add_(0, col, vals)
        else:  # reduce_n
            result = torch.zeros(M, *self.block_shape, dtype=self.dtype, device=self.device) if self.is_block else torch.zeros(M, dtype=self.dtype, device=self.device)
            if self.is_block:
                result.scatter_add_(0, row.unsqueeze(-1).expand(-1, *self.block_shape), vals)
            else:
                result.scatter_add_(0, row, vals)
    
    # Handle other axes reduction
    if other_axes:
        result_axes = [self._values_axis_for_dim(a) for a in other_axes]
        result = result.sum(dim=tuple(result_axes), keepdim=keepdim)
    
    return result

def _sum_over_batch_block(
    self, 
    axes: Tuple[int, ...], 
    keepdim: bool
) -> "SparseTensor":
    """Sum over batch/block dimensions only - stays sparse."""
    # Map tensor axes to values axes
    val_axes = tuple(self._values_axis_for_dim(a) for a in axes)
    new_values = self.values.sum(dim=val_axes, keepdim=keepdim)
    
    # Compute new shape
    new_shape = list(self._shape)
    if keepdim:
        for a in axes:
            new_shape[a] = 1
    else:
        for a in sorted(axes, reverse=True):
            del new_shape[a]
    
    # Adjust sparse_dim if needed
    new_sparse_dim = list(self._sparse_dim)
    if not keepdim:
        removed_before_m = sum(1 for a in axes if a < self._sparse_dim[0])
        removed_before_n = sum(1 for a in axes if a < self._sparse_dim[1])
        new_sparse_dim[0] -= removed_before_m
        new_sparse_dim[1] -= removed_before_n
    
    return SparseTensor(
        new_values, self.row_indices, self.col_indices, 
        tuple(new_shape), sparse_dim=tuple(new_sparse_dim)
    )

def _mean_impl(
    self, 
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdim: bool = False
) -> Union[torch.Tensor, "SparseTensor"]:
    """
    Mean of sparse tensor elements over specified axis.
    
    Note: For sparse dimensions, this computes mean of non-zero values only,
    NOT the mean over all M*N elements. For full mean, use to_dense().mean().
    
    Parameters
    ----------
    axis : int, tuple of ints, or None
        Axis or axes along which to compute mean.
    keepdim : bool
        Whether to keep the reduced dimensions.
        
    Returns
    -------
    torch.Tensor or SparseTensor
        Mean values.
    
    Examples
    --------
    >>> A = SparseTensor(val, row, col, (10, 10))
    >>> A.mean()           # Mean of all non-zero values
    >>> A.mean(axis=0)     # Mean over batch dimension
    """
    if axis is None:
        return self.values.mean()
    
    axes = self._normalize_axis(axis)
    
    # For sparse dims, we compute sum/count of nnz (not M*N)
    sum_result = self.sum(axis=axis, keepdim=keepdim)
    
    # Compute divisor based on axes
    divisor = 1
    for a in axes:
        divisor *= self._shape[a]
    
    # But for sparse dimensions, divisor should be nnz not M*N
    dim_types = [self._get_dim_type(a) for a in axes]
    if 'sparse_m' in dim_types or 'sparse_n' in dim_types:
        # For sparse reduction, we're averaging over nnz values
        sparse_divisor = 1
        if 'sparse_m' in dim_types:
            sparse_divisor *= self.sparse_shape[0]
        if 'sparse_n' in dim_types:
            sparse_divisor *= self.sparse_shape[1]
        # Replace M*N with nnz
        divisor = divisor // sparse_divisor * self.nnz
    
    if isinstance(sum_result, SparseTensor):
        return SparseTensor(
            sum_result.values / divisor,
            sum_result.row_indices,
            sum_result.col_indices,
            sum_result.shape,
            sparse_dim=sum_result.sparse_dim
        )
    return sum_result / divisor

def _prod_impl(
    self, 
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdim: bool = False
) -> Union[torch.Tensor, "SparseTensor"]:
    """
    Product of sparse tensor elements over specified axis.
    
    Warning: For sparse matrices, zero elements are not included in the product.
    This means prod() computes the product of non-zero values only.
    
    Parameters
    ----------
    axis : int, tuple of ints, or None
        Axis or axes along which to compute product.
    keepdim : bool
        Whether to keep the reduced dimensions.
        
    Returns
    -------
    torch.Tensor or SparseTensor
        Product values.
    
    Examples
    --------
    >>> A = SparseTensor(val, row, col, (10, 10))
    >>> A.prod()           # Product of all non-zero values
    >>> A.prod(axis=0)     # Product over batch dimension
    """
    if axis is None:
        return self.values.prod()
    
    axes = self._normalize_axis(axis)
    dim_types = [self._get_dim_type(d) for d in axes]
    
    # Check if we're reducing over sparse dimensions
    has_sparse_reduction = any(dt in ('sparse_m', 'sparse_n') for dt in dim_types)
    
    if has_sparse_reduction:
        # For sparse reduction, prod is complex - convert to dense
        warnings.warn(
            "prod() over sparse dimensions converts to dense. "
            "This may use significant memory for large matrices."
        )
        dense = self.to_dense()
        return dense.prod(dim=axes, keepdim=keepdim)
    else:
        # Only batch/block reduction
        val_axes = tuple(self._values_axis_for_dim(a) for a in axes)
        new_values = self.values.prod(dim=val_axes, keepdim=keepdim)
        
        new_shape = list(self._shape)
        if keepdim:
            for a in axes:
                new_shape[a] = 1
        else:
            for a in sorted(axes, reverse=True):
                del new_shape[a]
        
        new_sparse_dim = list(self._sparse_dim)
        if not keepdim:
            removed_before_m = sum(1 for a in axes if a < self._sparse_dim[0])
            removed_before_n = sum(1 for a in axes if a < self._sparse_dim[1])
            new_sparse_dim[0] -= removed_before_m
            new_sparse_dim[1] -= removed_before_n
        
        return SparseTensor(
            new_values, self.row_indices, self.col_indices,
            tuple(new_shape), sparse_dim=tuple(new_sparse_dim)
        )

def _max_impl(
    self, 
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdim: bool = False
) -> Union[torch.Tensor, "SparseTensor"]:
    """Max of non-zero values over specified axis."""
    if axis is None:
        return self.values.max()
    
    axes = self._normalize_axis(axis)
    dim_types = [self._get_dim_type(d) for d in axes]
    has_sparse_reduction = any(dt in ('sparse_m', 'sparse_n') for dt in dim_types)
    
    if has_sparse_reduction:
        dense = self.to_dense()
        return dense.max(dim=axes[0], keepdim=keepdim).values if len(axes) == 1 else dense.amax(dim=axes, keepdim=keepdim)
    else:
        val_axes = tuple(self._values_axis_for_dim(a) for a in axes)
        new_values = self.values.amax(dim=val_axes, keepdim=keepdim)
        
        new_shape = list(self._shape)
        if keepdim:
            for a in axes:
                new_shape[a] = 1
        else:
            for a in sorted(axes, reverse=True):
                del new_shape[a]
        
        new_sparse_dim = list(self._sparse_dim)
        if not keepdim:
            removed_before_m = sum(1 for a in axes if a < self._sparse_dim[0])
            removed_before_n = sum(1 for a in axes if a < self._sparse_dim[1])
            new_sparse_dim[0] -= removed_before_m
            new_sparse_dim[1] -= removed_before_n
        
        return SparseTensor(
            new_values, self.row_indices, self.col_indices,
            tuple(new_shape), sparse_dim=tuple(new_sparse_dim)
        )

def _min_impl(
    self, 
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdim: bool = False
) -> Union[torch.Tensor, "SparseTensor"]:
    """Min of non-zero values over specified axis."""
    if axis is None:
        return self.values.min()
    
    axes = self._normalize_axis(axis)
    dim_types = [self._get_dim_type(d) for d in axes]
    has_sparse_reduction = any(dt in ('sparse_m', 'sparse_n') for dt in dim_types)
    
    if has_sparse_reduction:
        dense = self.to_dense()
        return dense.min(dim=axes[0], keepdim=keepdim).values if len(axes) == 1 else dense.amin(dim=axes, keepdim=keepdim)
    else:
        val_axes = tuple(self._values_axis_for_dim(a) for a in axes)
        new_values = self.values.amin(dim=val_axes, keepdim=keepdim)
        
        new_shape = list(self._shape)
        if keepdim:
            for a in axes:
                new_shape[a] = 1
        else:
            for a in sorted(axes, reverse=True):
                del new_shape[a]
        
        new_sparse_dim = list(self._sparse_dim)
        if not keepdim:
            removed_before_m = sum(1 for a in axes if a < self._sparse_dim[0])
            removed_before_n = sum(1 for a in axes if a < self._sparse_dim[1])
            new_sparse_dim[0] -= removed_before_m
            new_sparse_dim[1] -= removed_before_n
        
        return SparseTensor(
            new_values, self.row_indices, self.col_indices,
            tuple(new_shape), sparse_dim=tuple(new_sparse_dim)
        )

