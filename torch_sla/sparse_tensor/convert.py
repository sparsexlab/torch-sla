"""Conversion / sparsity-pattern reshaping for SparseTensor."""
from __future__ import annotations
import warnings
from typing import Optional, Tuple, Union
import torch

from .core import SparseTensor  # noqa: E402  # forward calls back


def to_torch_sparse(self, batch_idx: Optional[Tuple[int, ...]] = None) -> torch.Tensor:
    """
    Convert to PyTorch sparse COO tensor.
    
    Parameters
    ----------
    batch_idx : Tuple[int, ...], optional
        For batched tensors, which batch element to convert.
        Default: (0, 0, ...) for first batch element.
    
    Returns
    -------
    torch.Tensor
        PyTorch sparse COO tensor.
    """
    if self.is_batched:
        if batch_idx is None:
            batch_idx = (0,) * len(self.batch_shape)
        vals = self.values[batch_idx]
    else:
        vals = self.values
    
    M, N = self.sparse_shape
    indices = torch.stack([self.row_indices, self.col_indices], dim=0)
    
    if self.is_block:
        return torch.sparse_coo_tensor(indices, vals, (M, N) + self.block_shape)
    else:
        return torch.sparse_coo_tensor(indices, vals, (M, N))

def to_dense(self, batch_idx: Optional[Tuple[int, ...]] = None) -> torch.Tensor:
    """
    Convert to dense tensor.
    
    Parameters
    ----------
    batch_idx : Tuple[int, ...], optional
        For batched tensors, which batch element to convert.
    
    Returns
    -------
    torch.Tensor
        Dense tensor.
    """
    return self.to_torch_sparse(batch_idx).to_dense()

def to_csr(self, batch_idx: Optional[Tuple[int, ...]] = None) -> torch.Tensor:
    """
    Convert to CSR format.
    
    Parameters
    ----------
    batch_idx : Tuple[int, ...], optional
        For batched tensors, which batch element to convert.
    
    Returns
    -------
    torch.Tensor
        PyTorch sparse CSR tensor.
    """
    return self.to_torch_sparse(batch_idx).to_sparse_csr()

def extract_partition(self, partition) -> "SparseTensor":
    """Extract this rank's local subdomain as a plain
    :class:`SparseTensor` in local coordinates.

    Given a :class:`~torch_sla.distributed.Partition` (the irregular
    owned/halo map produced by METIS / RCB / Hilbert), build a local
    ``(num_local, num_local)`` :class:`SparseTensor` whose COO
    triples are in **local** indexing: rows ``0..num_owned-1`` are
    the rows this rank owns, rows ``num_owned..num_local-1`` are
    the halo rows (zero-valued in the matrix; they exist so the
    x vector that matvec consumes can be the same num_local size
    used by the halo-exchange machinery).

    Used internally by :meth:`DSparseTensor.partition` to build the
    local SparseTensor backing each rank holds.

    Parameters
    ----------
    partition : Partition
        Irregular partition map -- ``owned_nodes`` / ``halo_nodes`` /
        ``global_to_local``. Usually produced by
        :func:`~torch_sla.distributed.partition_graph_metis` etc.

    Returns
    -------
    SparseTensor
        Local subdomain. ``.shape == (num_local, num_local)``;
        values live only in the owned-row slice.
    """
    if self.is_batched:
        raise ValueError("extract_partition() does not support "
                         "batched SparseTensor.")

    device = self.values.device
    g2l = partition.global_to_local.to(device=device,
                                        dtype=torch.int64)
    owned = partition.owned_nodes.to(device=device, dtype=torch.int64)
    halo  = partition.halo_nodes.to(device=device, dtype=torch.int64)
    num_owned = int(owned.numel())
    num_halo  = int(halo.numel())
    num_local = num_owned + num_halo

    # Map global row/col to local indices via g2l. ``g2l[g] == -1``
    # for globals not in this rank's (owned ∪ halo) set.
    rows_g = self.row_indices.to(device=device, dtype=torch.int64)
    cols_g = self.col_indices.to(device=device, dtype=torch.int64)
    rows_l = g2l[rows_g]
    cols_l = g2l[cols_g]

    # Keep entries whose row is owned AND col is local (the
    # owned-row slice of A_local, with cols mapped into the
    # owned+halo local frame).
    mask = (rows_l >= 0) & (rows_l < num_owned) & (cols_l >= 0)
    local_rows = rows_l[mask]
    local_cols = cols_l[mask]
    local_vals = self.values[mask]

    return SparseTensor(local_vals, local_rows, local_cols,
                         (num_local, num_local))

def save_distributed(self, directory, num_partitions: int,
                     partition_method: str = "simple",
                     coords: Optional[torch.Tensor] = None,
                     verbose: bool = False) -> None:
    """Partition + write all shards to ``directory`` (single-process).
    Output is identical to a collective ``DSparseTensor.save``."""
    from ..io import save_sparse_sharded
    save_sparse_sharded(self, directory, num_partitions=num_partitions,
                        partition_method=partition_method, coords=coords,
                        verbose=verbose)

def partition_for_rank(self, rank: int, world_size: int,
                       coords: Optional[torch.Tensor] = None,
                       partition_method: str = "simple",
                       verbose: bool = False) -> "DSparseTensor":
    """Build this rank's :class:`DSparseTensor` shard. Collective:
    every rank must call with the same global matrix."""
    from ..distributed import DSparseTensor
    if self.is_batched:
        raise ValueError("partition_for_rank does not support batched SparseTensor")
    return DSparseTensor.from_global_distributed(
        self.values,
        self.row_indices,
        self.col_indices,
        self.sparse_shape,
        rank=rank,
        world_size=world_size,
        coords=coords,
        partition_method=partition_method,
        verbose=verbose,
    )

def detect_matrix_type(self) -> str:
    """Detect the most specialised cuDSS matrix-type label for ``self``.

    Returns one of ``"general"``, ``"symmetric"``, ``"spd"``,
    ``"hermitian"``, ``"hpd"``. Used by ``solve(..., matrix_type="auto")``
    on the cuDSS backend to pick the cheapest factorisation
    (Cholesky / LDL^H) the matrix supports.

    The underlying test is conservative -- the positive-definiteness
    check is Gershgorin-based, which is sufficient but not
    necessary. A truly SPD/HPD matrix that is not strictly
    diagonally dominant may report as plain symmetric/hermitian;
    this is safe (cuDSS will simply use the slightly more
    expensive indefinite factorisation).

    Returns
    -------
    str
        ``"general"`` | ``"symmetric"`` | ``"spd"`` |
        ``"hermitian"`` | ``"hpd"``.

    Raises
    ------
    ValueError
        If called on a batched or block-sparse tensor; matrix-type
        classification is only defined for a single 2-D matrix.
    """
    if self.is_batched or len(self.block_shape) > 0:
        raise ValueError(
            "detect_matrix_type() only supports a non-batched, "
            "non-block-sparse 2-D matrix; got "
            f"is_batched={self.is_batched}, "
            f"block_shape={self.block_shape}."
        )
    if self.values.numel() == 0:
        return "general"

    if self.values.is_complex():
        if bool(self.is_hermitian().item()):
            return "hpd" if bool(self.is_positive_definite().item()) else "hermitian"
        if bool(self.is_symmetric().item()):
            return "symmetric"  # complex symmetric, cuDSS LDL^T
        return "general"
    # Real path
    if bool(self.is_symmetric().item()):
        return "spd" if bool(self.is_positive_definite().item()) else "symmetric"
    return "general"

def T(self) -> "SparseTensor":
    """
    Transpose the sparse dimensions.
    
    Returns
    -------
    SparseTensor
        Transposed tensor with row/col indices swapped.
    """
    new_shape = list(self._shape)
    dim_m, dim_n = self._sparse_dim
    new_shape[dim_m], new_shape[dim_n] = new_shape[dim_n], new_shape[dim_m]
    
    result = SparseTensor(
        self.values,
        self.col_indices,  # Swap row and col
        self.row_indices,
        tuple(new_shape),
        sparse_dim=self._sparse_dim
    )
    return result

def conj(self) -> "SparseTensor":
    """
    Element-wise complex conjugate (same sparsity pattern).

    For real tensors this is a no-op view-equivalent (``values`` are
    returned unchanged); for complex tensors every stored value is
    conjugated. Gradients flow through ``torch.conj``.
    """
    return SparseTensor(
        self.values.conj(),
        self.row_indices,
        self.col_indices,
        tuple(self._shape),
        sparse_dim=self._sparse_dim,
    )

def H(self) -> "SparseTensor":
    """
    Conjugate (Hermitian) transpose ``A^H = conj(A)^T``.

    For real matrices this equals :meth:`T`. For complex matrices it is
    the proper adjoint, e.g. ``A.H()`` of a Hermitian matrix returns ``A``.
    Use this (not :meth:`T`) for inner products / normal equations on
    complex systems.
    """
    new_shape = list(self._shape)
    dim_m, dim_n = self._sparse_dim
    new_shape[dim_m], new_shape[dim_n] = new_shape[dim_n], new_shape[dim_m]

    return SparseTensor(
        self.values.conj(),       # conjugate the stored values
        self.col_indices,         # swap row and col (transpose)
        self.row_indices,
        tuple(new_shape),
        sparse_dim=self._sparse_dim,
    )

def flatten_blocks(self) -> "SparseTensor":
    """
    Flatten block dimensions into the sparse (M, N) dimensions.
    
    For a block-sparse tensor with shape [...batch, M, N, *block_shape],
    this creates a new tensor with shape [...batch, M*block_M, N*block_N]
    where each block entry becomes multiple scalar entries.
    
    Returns
    -------
    SparseTensor
        Flattened tensor without block dimensions.
        
    Example
    -------
    >>> # Block sparse: shape (10, 10, 2, 2), block_shape=(2, 2)
    >>> A = SparseTensor(val, row, col, (10, 10, 2, 2))
    >>> A_flat = A.flatten_blocks()
    >>> print(A_flat.shape)  # (20, 20)
    >>> print(A_flat.nnz)    # nnz * 4 (each block has 4 elements)
    
    Notes
    -----
    - Only works for 2D block shapes (block_M, block_N).
    - Use `unflatten_blocks(block_shape)` to reverse this operation.
    - The flattened tensor's sparsity pattern may have duplicates that
      need to be coalesced.
    """
    if not self.is_block:
        return self  # No blocks, return as is
    
    block_shape = self.block_shape
    if len(block_shape) != 2:
        raise ValueError(f"flatten_blocks only supports 2D blocks, got {block_shape}")
    
    block_M, block_N = block_shape
    M, N = self.sparse_shape
    batch_shape = self.batch_shape
    nnz = self.nnz
    
    # New sparse shape
    new_M = M * block_M
    new_N = N * block_N
    
    # Expand block entries into individual entries
    # Original: values shape [...batch, nnz, block_M, block_N]
    # New: values shape [...batch, nnz * block_M * block_N]
    
    # Create new row/col indices
    # For each (row, col) block at position (i, j), create indices:
    # (i*block_M + bi, j*block_N + bj) for bi in [0, block_M), bj in [0, block_N)
    
    row = self.row_indices  # [nnz]
    col = self.col_indices  # [nnz]
    
    # Create block offsets
    block_offsets = torch.arange(block_M * block_N, device=self.device)
    bi = block_offsets // block_N  # [block_M * block_N]
    bj = block_offsets % block_N   # [block_M * block_N]
    
    # Expand row/col to new indices
    # new_row[k * block_M * block_N + offset] = row[k] * block_M + bi[offset]
    new_row = (row.unsqueeze(-1) * block_M + bi.unsqueeze(0)).reshape(-1)  # [nnz * block_size]
    new_col = (col.unsqueeze(-1) * block_N + bj.unsqueeze(0)).reshape(-1)  # [nnz * block_size]
    
    # Flatten values
    if len(batch_shape) > 0:
        # [...batch, nnz, block_M, block_N] -> [...batch, nnz * block_M * block_N]
        vals = self.values.reshape(*batch_shape, nnz * block_M * block_N)
    else:
        # [nnz, block_M, block_N] -> [nnz * block_M * block_N]
        vals = self.values.reshape(nnz * block_M * block_N)
    
    new_shape = batch_shape + (new_M, new_N)
    
    return SparseTensor(
        vals, new_row, new_col, new_shape,
        sparse_dim=self._sparse_dim
    )

def unflatten_blocks(self, block_shape: Tuple[int, int]) -> "SparseTensor":
    """
    Restore block structure from a flattened tensor.
    
    This is the inverse of `flatten_blocks()`. It groups scalar entries
    back into block entries.
    
    Parameters
    ----------
    block_shape : Tuple[int, int]
        The (block_M, block_N) dimensions to create.
        M and N must be divisible by block_M and block_N respectively.
    
    Returns
    -------
    SparseTensor
        Block-sparse tensor with the specified block shape.
        
    Example
    -------
    >>> A_flat = SparseTensor(val, row, col, (20, 20))
    >>> A_block = A_flat.unflatten_blocks((2, 2))
    >>> print(A_block.shape)  # (10, 10, 2, 2)
    >>> print(A_block.block_shape)  # (2, 2)
    
    Notes
    -----
    - Requires that the sparsity pattern is block-aligned.
    - All block entries must be present (dense within each block).
    - For sparse blocks, use `to_block_sparse()` instead.
    """
    if self.is_block:
        raise ValueError("Tensor already has block structure. Use flatten_blocks first.")
    
    if len(block_shape) != 2:
        raise ValueError(f"block_shape must be 2D, got {block_shape}")
    
    block_M, block_N = block_shape
    M, N = self.sparse_shape
    batch_shape = self.batch_shape
    
    if M % block_M != 0 or N % block_N != 0:
        raise ValueError(
            f"Sparse shape ({M}, {N}) not divisible by block_shape ({block_M}, {block_N})"
        )
    
    new_M = M // block_M
    new_N = N // block_N
    block_size = block_M * block_N
    
    row = self.row_indices
    col = self.col_indices
    nnz = self.nnz
    
    if nnz % block_size != 0:
        raise ValueError(
            f"Number of non-zeros ({nnz}) not divisible by block size ({block_size}). "
            "The sparsity pattern may not be block-aligned."
        )
    
    # Compute block indices
    block_row = row // block_M  # Which block row
    block_col = col // block_N  # Which block col
    local_row = row % block_M   # Position within block
    local_col = col % block_N   # Position within block
    
    # Group entries by (block_row, block_col)
    # Create a unique block ID for sorting
    block_id = block_row * new_N + block_col
    
    # Sort by block_id, then by local position
    local_offset = local_row * block_N + local_col
    sort_key = block_id * block_size + local_offset
    sort_idx = torch.argsort(sort_key)
    
    sorted_block_id = block_id[sort_idx]
    sorted_local_offset = local_offset[sort_idx]
    
    # Extract unique blocks
    unique_blocks, counts = torch.unique_consecutive(sorted_block_id, return_counts=True)
    
    if not torch.all(counts == block_size):
        raise ValueError(
            "Not all blocks are complete. Each block must have exactly "
            f"{block_size} entries."
        )
    
    num_blocks = unique_blocks.size(0)
    new_row_indices = unique_blocks // new_N
    new_col_indices = unique_blocks % new_N
    
    # Reshape values to include block dimensions
    if len(batch_shape) > 0:
        # Sort values: [...batch, nnz] -> [...batch, num_blocks * block_size]
        sorted_vals = self.values[..., sort_idx]
        new_vals = sorted_vals.reshape(*batch_shape, num_blocks, block_M, block_N)
    else:
        sorted_vals = self.values[sort_idx]
        new_vals = sorted_vals.reshape(num_blocks, block_M, block_N)
    
    new_shape = batch_shape + (new_M, new_N, block_M, block_N)
    
    return SparseTensor(
        new_vals, new_row_indices, new_col_indices, new_shape,
        sparse_dim=self._sparse_dim
    )
