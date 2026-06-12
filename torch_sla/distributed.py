"""
Distributed sparse matrix for large-scale CFD / FEM / GNN computations.

:class:`DSparseTensor` mirrors :class:`torch.distributed.tensor.DTensor`:
each rank holds a local :class:`~torch_sla.sparse_tensor.SparseTensor`
chunk plus a :class:`~torch_sla.partition.Partition` map (owned rows +
halo) and the matvec stays entirely in the ``Shard(0)`` space via halo
exchange + local SpMV.

::

    from torch_sla import SparseTensor, DSparseTensor, solve, SolverConfig
    from torch.distributed.device_mesh import init_device_mesh

    A    = SparseTensor(val, row, col, shape)
    mesh = init_device_mesh("cpu", (world_size,))
    D    = DSparseTensor.partition(A, mesh, partition_method="metis")
    b_dt = D.scatter(b_global)

    with SolverConfig(method="cg", atol=1e-10, rtol=1e-10, maxiter=2000):
        x_dt = solve(D, b_dt)

The Krylov methods (CG / BiCGStab / GMRES / FGMRES / MINRES) and
preconditioners (Jacobi / block-Jacobi / SSOR / polynomial) live in
:mod:`torch_sla.distributed_solve`; partitioning lives in
:mod:`torch_sla.partition`.
"""

import os
import torch
from typing import Any, Tuple, List, Dict, Optional, Union, Literal
from dataclasses import dataclass
import warnings

from .backends import (
    is_scipy_available,
    is_eigen_available,
    is_cupy_available,
    is_cudss_available,
    select_backend,
    select_method,
    BackendType,
    MethodType,
)

try:
    import torch.distributed as dist
    DIST_AVAILABLE = True
except ImportError:
    DIST_AVAILABLE = False

# DTensor support (PyTorch 2.0+). On torch >=2.2 these live under
# ``torch.distributed.tensor``; torch 2.0-2.1 still keeps them under the
# private ``_tensor`` namespace. Centralise the version check here so
# every runtime import below can use the same names.
try:
    from torch.distributed.tensor import DTensor
    from torch.distributed.tensor.placement_types import Shard, Replicate
    DTENSOR_AVAILABLE = True
    _dtensor_module = "torch.distributed.tensor"
except ImportError:
    try:
        from torch.distributed._tensor import DTensor
        from torch.distributed._tensor.placement_types import Shard, Replicate
        DTENSOR_AVAILABLE = True
        _dtensor_module = "torch.distributed._tensor"
    except ImportError:
        DTENSOR_AVAILABLE = False
        DTensor = None
        Shard = None
        Replicate = None
        _dtensor_module = None


def _is_dtensor(x) -> bool:
    """Check if x is a DTensor instance."""
    if not DTENSOR_AVAILABLE or DTensor is None:
        return False
    return isinstance(x, DTensor)


# Partition struct + partitioning algorithms (METIS / simple / RCB /
# slicing / Hilbert) + halo discovery live in :mod:`torch_sla.partition`
# now. Re-exported here so existing ``from torch_sla.distributed import
# Partition, partition_simple, ...`` call sites keep working.
from .partition import (
    Partition,
    partition_graph_metis,
    partition_simple,
    partition_coordinates,
    _hilbert_curve_indices,
    _hilbert_sort_indices,
    _rcb_partition,
    find_halo_nodes,
    build_partition,
    resolve_partition_ids,
)


# ====================================================================== #
# DTensor-mirror placement vocabulary for sparse tensors.
#
# Adapted from ``torch.distributed.tensor.placement_types``:
#
# * :class:`Replicated`        --  every rank holds the full matrix
#                                  (analogous to ``Replicate()``).
# * :class:`RowPartitioned`    --  rows are split across ranks via an
#                                  irregular METIS / RCB / simple map
#                                  (the sparse analog of ``Shard(0)``;
#                                  *not* the uniform DTensor shard).
#
# A :class:`DSparseSpec` bundles a placement with the device mesh and
# global shape so the rest of the API can mirror DTensor's
# ``DTensor._spec``.
# ====================================================================== #
@dataclass(frozen=True)
class Replicated:
    """DSparseTensor placement: every rank holds the entire matrix."""
    pass


@dataclass(frozen=True)
class SparseShard:
    """DSparseTensor placement: shard one of the **sparse** axes of the
    underlying :class:`SparseTensor` across the mesh.

    Mirrors ``torch.distributed.tensor.Shard(dim)`` -- one generic
    placement parameterised by axis number, not separate classes per
    direction. For a 2-D sparse matrix ``(M, N)`` the canonical axes
    are 0 (rows) and 1 (cols); for a batched / block-sparse tensor
    ``(*batch, M, N, *block)`` the row axis is ``axis=len(batch_shape)``
    and the col axis is ``axis=len(batch_shape) + 1``.

    Unlike DTensor's ``Shard(dim)`` -- which assumes uniform chunks of
    size ``N/world_size`` -- a sparse shard can carry an irregular
    partition map (METIS / hypergraph / RCB) via the optional
    ``partition`` field.

    The ``partition`` field carries the per-rank ``owned_nodes`` /
    ``halo_nodes`` / ``neighbor_partitions`` so the placement is the
    single source of truth for the irregular shard map. It is set by
    :meth:`DSparseTensor.from_sparse_local` / :meth:`partition` /
    :meth:`from_global_distributed`; ``None`` only when an empty
    placement object is being passed around as a type tag.
    """
    axis: int = 0
    partition: Optional["Partition"] = None

    def __eq__(self, other: object) -> bool:
        # Frozen dataclass eq would call torch.Tensor.__eq__ inside
        # Partition and explode; restrict equality to the axis +
        # partition_id pair, which is what placement dispatch needs.
        if not isinstance(other, SparseShard):
            return False
        if self.axis != other.axis:
            return False
        my_pid = self.partition.partition_id if self.partition else None
        ot_pid = other.partition.partition_id if other.partition else None
        return my_pid == ot_pid

    def __hash__(self) -> int:
        pid = self.partition.partition_id if self.partition else -1
        return hash((type(self).__name__, self.axis, pid))


def row_shard(axis_offset: int = 0) -> SparseShard:
    """Convenience: row sharding for a ``(*batch, M, N, *block)`` tensor
    where there are ``axis_offset`` leading batch dims. Defaults to a
    plain 2-D matrix (``axis_offset=0``)."""
    return SparseShard(axis=axis_offset)


def col_shard(axis_offset: int = 0) -> SparseShard:
    """Convenience: col sharding for a ``(*batch, M, N, *block)`` tensor.
    Defaults to a plain 2-D matrix (``axis_offset=0``)."""
    return SparseShard(axis=axis_offset + 1)


# ---------------------------------------------------------------------- #
# Backward-compatibility aliases.
# Pre-rename: ``RowPartitioned()`` -- empty marker for "row-sharded".
# Post-rename: ``SparseShard(axis=0)`` is the canonical name.
# Kept as a deprecated alias so external callers keep working.
# ---------------------------------------------------------------------- #
def RowPartitioned() -> SparseShard:  # noqa: N802 -- legacy capital
    """Deprecated alias for :func:`row_shard()` / ``SparseShard(axis=0)``."""
    return SparseShard(axis=0)


SparsePlacement = Union[Replicated, SparseShard]


@dataclass(frozen=True)
class DSparseSpec:
    """The sparse analog of :class:`torch.distributed.tensor.DTensorSpec`.

    Bundles the placement, the device mesh, and the global shape so we
    can dispatch operations purely off the spec and treat
    :class:`DSparseTensor` like any other distributed tensor.
    """
    placement: SparsePlacement
    mesh: Any                            # torch.distributed.DeviceMesh
    global_shape: Tuple[int, int]


class DSparseTensor:
    """
    Distributed Sparse Tensor with automatic partitioning and halo exchange.

    A Pythonic wrapper that provides a unified interface for distributed
    sparse matrix operations. Supports indexing to access individual partitions.
    
    Parameters
    ----------
    values : torch.Tensor
        Non-zero values [nnz]
    row_indices : torch.Tensor
        Row indices [nnz]
    col_indices : torch.Tensor
        Column indices [nnz]
    shape : Tuple[int, int]
        Matrix shape (m, n)
    num_partitions : int
        Number of partitions to create
    coords : torch.Tensor, optional
        Node coordinates for geometric partitioning [num_nodes, dim]
    partition_method : str
        Partitioning method: 'metis', 'rcb', 'slicing', 'simple'
    device : str or torch.device
        Device for the matrix data
    verbose : bool
        Whether to print partition info
    
    Example
    -------
    >>> import torch
    >>> from torch_sla import DSparseTensor
    >>> 
    >>> # Create distributed tensor with 4 partitions
    >>> A = DSparseTensor(val, row, col, shape, num_partitions=4)
    >>> 
    >>> # Access individual partitions
    >>> A0 = A[0]  # First partition
    >>> A1 = A[1]  # Second partition
    >>> 
    >>> # Iterate over partitions
    >>> for partition in A:
    >>>     x = partition.solve(b_local)
    >>> 
    >>> # Properties
    >>> print(A.num_partitions)  # 4
    >>> print(A.shape)           # Global shape
    >>> print(len(A))            # 4
    >>> 
    >>> # Move to CUDA
    >>> A_cuda = A.cuda()
    >>> 
    >>> # Local halo exchange (for testing)
    >>> x_list = [torch.zeros(A[i].num_local) for i in range(4)]
    >>> A.halo_exchange_local(x_list)
    """
    
    def __init__(self) -> None:
        """Direct instantiation isn't supported -- use one of the
        classmethod constructors:

        * :meth:`partition` -- global :class:`SparseTensor` + mesh →
          row-sharded :class:`DSparseTensor`.
        * :meth:`from_global_distributed` -- global COO + rank/world
          → row-sharded :class:`DSparseTensor` (broadcasts partition
          ids from rank 0 for determinism).
        * :meth:`from_sparse_local` -- per-rank ``(SparseTensor,
          Partition)`` → :class:`DSparseTensor`.

        Each populates ``_local_tensor`` (the per-rank SparseTensor
        backing) and ``_spec`` (the placement + mesh + global shape).
        """
        raise TypeError(
            "DSparseTensor() does not support direct instantiation. Use "
            "DSparseTensor.partition(A, mesh) / "
            "DSparseTensor.from_global_distributed(...) / "
            "DSparseTensor.from_sparse_local(...)."
        )


    
    
    # ====================================================================== #
    # DTensor-mirror API: from_local / to_local / full_tensor / redistribute.
    #
    # These methods give DSparseTensor the same shape of API as
    # ``torch.distributed.tensor.DTensor``: every call resolves through
    # a private :class:`DSparseSpec` that bundles the placement, the
    # device mesh, and the global shape. Vectors crossing the API stay
    # as ``DTensor[Shard(0)]`` so the rest of the PyTorch distributed
    # ecosystem (FSDP, TP, DCP) composes for free.
    # ====================================================================== #

    @classmethod
    def from_sparse_local(
        cls,
        local_tensor: "SparseTensor",
        mesh: Any,
        partition: "Partition",
        *,
        axis: int = 0,
        global_shape: Optional[Tuple[int, int]] = None,
    ) -> "DSparseTensor":
        """Wrap a per-rank :class:`SparseTensor` chunk (already in
        local coords) plus its :class:`Partition` as a DSparseTensor.

        Use together with :meth:`SparseTensor.extract_partition`:

        .. code-block:: python

            partition = compute_partition(...)
            local_tensor = A_global.extract_partition(partition)
            D = DSparseTensor.from_sparse_local(
                local_tensor, mesh, partition,
                global_shape=A_global.shape,
            )
            y_dt = D @ x_dt              # halo exchange + local SpMV

        The partition is stamped onto ``_spec.placement.partition`` so
        the placement is the single source of truth for the irregular
        shard map.

        Parameters
        ----------
        local_tensor : SparseTensor
            This rank's local subdomain (size ``(num_local, num_local)``,
            COO in local coordinates). Usually built by
            :meth:`SparseTensor.extract_partition`.
        mesh : DeviceMesh
            The PyTorch device mesh.
        partition : Partition
            Irregular partition map for this rank
            (``owned_nodes`` / ``halo_nodes`` / ``neighbor_partitions``
            etc).
        axis : int
            Sparse axis being sharded (default 0 = rows).
        global_shape : Tuple[int, int], optional
            Global matrix shape. If omitted, inferred from
            ``partition.local_to_global.numel()`` -- only valid for
            square matrices.
        """
        if global_shape is None:
            n = int(partition.local_to_global.numel() +
                    0)  # placeholder; caller should pass it explicitly
            global_shape = (n, n)
        self = cls.__new__(cls)
        self._values = None
        self._row_indices = None
        self._col_indices = None
        self._shape = global_shape
        self._num_partitions = mesh.size() if mesh is not None else 1
        self._coords = None
        self._partition_method = None
        self._verbose = False
        self._device = local_tensor.values.device
        self._local_tensor = local_tensor
        self._halo_send_buffers = {}
        self._halo_recv_buffers = {}
        placement = SparseShard(axis=axis, partition=partition)
        self._spec = DSparseSpec(placement=placement, mesh=mesh,
                                 global_shape=global_shape)
        return self

    @classmethod
    def partition(
        cls,
        A: "SparseTensor",
        mesh: Any,
        *,
        partition_method: str = "simple",
        coords: Optional[torch.Tensor] = None,
        verbose: bool = False,
    ) -> "DSparseTensor":
        """One-shot constructor: take a global :class:`SparseTensor` +
        :class:`DeviceMesh`, partition rows across the mesh, return a
        ready-to-use distributed tensor with :class:`RowPartitioned`
        placement.

        Equivalent to::

            local = A.partition_for_rank(rank, world_size,
                                          partition_method=partition_method,
                                          coords=coords)
            D = DSparseTensor.from_local(local, mesh,
                                          placement=RowPartitioned())

        but in one line. This is the recommended way to build a
        distributed sparse tensor from a global :class:`SparseTensor`
        for both unit tests and small-to-medium production runs (where
        every rank can afford to hold the global ``A`` briefly).

        For memory-tight scenarios where only rank 0 should ever
        materialise the global matrix, use
        :meth:`from_global_distributed` (which broadcasts only the
        partition IDs from rank 0) and chain :meth:`from_local`
        manually.

        Parameters
        ----------
        A : SparseTensor
            Global sparse matrix; every rank should hold an identical
            copy at the time of the call.
        mesh : DeviceMesh
            Target device mesh. ``mesh.size()`` becomes the world size
            and ``dist.get_rank()`` picks this rank's chunk.
        partition_method : str
            Partitioning algorithm passed through to
            :meth:`SparseTensor.partition_for_rank`: ``"simple"`` /
            ``"metis"`` / ``"rcb"`` / ``"slicing"``.
        coords : torch.Tensor, optional
            Node coordinates for geometric partitioning (RCB/slicing).
        verbose : bool
            Print partition info on each rank.
        """
        if DIST_AVAILABLE and dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0
        world_size = mesh.size() if mesh is not None else 1

        # Compute partition ids, build the Partition struct, extract
        # this rank's local SparseTensor, wrap.
        partition_ids = resolve_partition_ids(
            A.row_indices, A.col_indices,
            int(A.shape[0]), world_size,
            method=partition_method, coords=coords,
        )
        partition = build_partition(
            A.row_indices, A.col_indices,
            int(A.shape[0]), partition_ids, rank,
        )
        local_st = A.extract_partition(partition)
        return cls.from_sparse_local(
            local_st, mesh, partition,
            global_shape=tuple(A.shape),
        )

    @property
    def spec(self) -> Optional[DSparseSpec]:
        """The :class:`DSparseSpec` for this tensor (placement + mesh +
        global shape), or ``None`` if this instance was built via the
        legacy single-process simulator constructor."""
        return self._spec

    def scatter(self, global_vec: torch.Tensor) -> "DTensor":
        """Convenience: extract this rank's owned slice from a global
        vector and wrap as a ``DTensor[Shard(0)]``.

        Common usage::

            b_dt = D.scatter(b_global)        # build distributed RHS
            x_dt = solve(D, b_dt)             # distributed solve
            r_dt = b_dt - D @ x_dt            # distributed residual

        ``global_vec`` is a 1-D ``torch.Tensor`` of size
        ``global_shape[0]``. Every rank should hold the same copy
        (typical in tests; in production the caller loads on rank 0
        and broadcasts).
        """
        partition = self._partition_for_dispatch()
        if partition is None:
            raise RuntimeError(
                "scatter() requires a partition map -- build this "
                "DSparseTensor via .partition(...) or .from_local(...)")
        owned = partition.owned_nodes.to(device=global_vec.device,
                                          dtype=torch.int64)
        local_slice = global_vec[owned].contiguous()
        _DTensor = DTensor  # use module-level import with fallback
        return _DTensor.from_local(local_slice, self._spec.mesh,
                                    [Shard(0)])

    def _partition_for_dispatch(self) -> Optional["Partition"]:
        """Return the active :class:`Partition` from the spec, or
        ``None`` if no spec is set."""
        if self._spec is not None and isinstance(
                self._spec.placement, SparseShard) \
                and self._spec.placement.partition is not None:
            return self._spec.placement.partition
        return None

    def full_tensor(self) -> "SparseTensor":
        """Materialise the full global matrix on every rank.

        Mirrors :meth:`DTensor.full_tensor`: every rank ends up with
        the same :class:`SparseTensor`. For :class:`SparseShard(axis=0)`
        we drop the halo rows from the local SparseTensor (only owned
        rows count), translate local indices back to global, and
        all-gather the (val, row, col) triples across the mesh.
        """
        from .sparse_tensor import SparseTensor

        if self._spec is None:
            raise RuntimeError(
                "DSparseTensor.full_tensor() requires a spec -- build "
                "via .partition(...) / .from_sparse_local(...).")

        partition = self._partition_for_dispatch()
        if partition is None:
            raise RuntimeError(
                "DSparseTensor.full_tensor() requires a SparseShard "
                "placement with a Partition.")

        st = self._local_tensor
        if st is None:
            raise RuntimeError(
                "DSparseTensor.full_tensor() requires a SparseTensor "
                "backing.")

        # Drop halo rows -- only owned rows participate in the global
        # matrix. Local row indices < num_owned are the owned ones.
        device = st.values.device
        num_owned = int(partition.owned_nodes.numel())
        owned_mask = st.row_indices < num_owned
        local_rows = st.row_indices[owned_mask]
        local_cols = st.col_indices[owned_mask]
        local_vals = st.values[owned_mask]

        # Translate local row / col → global indices.
        l2g = partition.local_to_global.to(device=device,
                                            dtype=torch.int64)
        global_rows = l2g[local_rows]
        global_cols = l2g[local_cols]

        if not (DIST_AVAILABLE and dist.is_initialized()):
            return SparseTensor(local_vals, global_rows, global_cols,
                                 tuple(self._spec.global_shape))

        # All-gather the per-rank triples across the mesh.
        world_size = dist.get_world_size()
        nnz_t = torch.tensor([int(global_rows.numel())], device=device,
                              dtype=torch.int64)
        all_nnz = [torch.zeros(1, device=device, dtype=torch.int64)
                    for _ in range(world_size)]
        dist.all_gather(all_nnz, nnz_t)
        sizes = [int(t.item()) for t in all_nnz]
        max_nnz = max(sizes)

        def _padded(t, dtype):
            out = torch.zeros(max_nnz, device=device, dtype=dtype)
            out[:t.numel()] = t.to(dtype=dtype)
            return out

        val_pad = _padded(local_vals, local_vals.dtype)
        row_pad = _padded(global_rows, torch.int64)
        col_pad = _padded(global_cols, torch.int64)

        all_vals = [torch.zeros_like(val_pad) for _ in range(world_size)]
        all_rows = [torch.zeros_like(row_pad) for _ in range(world_size)]
        all_cols = [torch.zeros_like(col_pad) for _ in range(world_size)]
        dist.all_gather(all_vals, val_pad)
        dist.all_gather(all_rows, row_pad)
        dist.all_gather(all_cols, col_pad)

        out_vals = torch.cat([all_vals[r][:sizes[r]] for r in range(world_size)])
        out_rows = torch.cat([all_rows[r][:sizes[r]] for r in range(world_size)])
        out_cols = torch.cat([all_cols[r][:sizes[r]] for r in range(world_size)])
        return SparseTensor(out_vals, out_rows, out_cols,
                             tuple(self._spec.global_shape))



    
    
    @classmethod
    def from_global_distributed(
        cls,
        values: torch.Tensor,
        row_indices: torch.Tensor,
        col_indices: torch.Tensor,
        shape: Tuple[int, int],
        rank: int,
        world_size: int,
        mesh: Any = None,
        coords: Optional[torch.Tensor] = None,
        partition_method: str = 'auto',
        device: Optional[Union[str, torch.device]] = None,
        verbose: bool = True
    ) -> "DSparseTensor":
        """
        Create local partition in a distributed-safe manner.
        
        This method ensures that all ranks compute the same partition assignment
        by having rank 0 compute the partition IDs and broadcasting to all ranks.
        
        Parameters
        ----------
        values : torch.Tensor
            Global non-zero values [nnz]
        row_indices : torch.Tensor
            Global row indices [nnz]
        col_indices : torch.Tensor
            Global column indices [nnz]
        shape : Tuple[int, int]
            Global matrix shape (M, N)
        rank : int
            Current process rank
        world_size : int
            Total number of processes
        coords : torch.Tensor, optional
            Node coordinates for geometric partitioning [num_nodes, dim]
        partition_method : str
            Partitioning method: 'metis', 'rcb', 'slicing', 'simple'
        device : str or torch.device, optional
            Target device
        verbose : bool
            Whether to print partition info
            
        Returns
        -------
        DSparseTensor
            This rank's row-sharded distributed tensor.
            
        Example
        -------
        >>> import torch.distributed as dist
        >>> 
        >>> # In each process:
        >>> rank = dist.get_rank()
        >>> world_size = dist.get_world_size()
        >>> 
        >>> local_matrix = DSparseTensor.from_global_distributed(
        ...     val, row, col, shape, 
        ...     rank=rank, world_size=world_size
        ... )
        """
        import torch.distributed as dist

        if device is None:
            device = values.device
        if isinstance(device, str):
            device = torch.device(device)

        # Compute partition IDs on rank 0 and broadcast for determinism.
        if rank == 0:
            partition_ids = resolve_partition_ids(
                row_indices, col_indices, int(shape[0]),
                world_size, method=partition_method, coords=coords,
            ).to(device)
        else:
            partition_ids = torch.zeros(shape[0], dtype=torch.int64,
                                         device=device)
        dist.broadcast(partition_ids, src=0)

        # Build Partition struct + extract local SparseTensor on this rank.
        partition = build_partition(
            row_indices, col_indices, int(shape[0]),
            partition_ids.cpu(), rank,
        )
        from .sparse_tensor import SparseTensor
        A = SparseTensor(values, row_indices, col_indices, shape)
        local_st = A.extract_partition(partition)

        # If no mesh was passed, build a 1-D mesh from the process
        # group so the result is still a real ``DSparseTensor[Shard(0)]``.
        if mesh is None:
            try:
                from torch.distributed.device_mesh import init_device_mesh
            except ImportError:
                from torch.distributed._tensor.device_mesh import init_device_mesh
            mesh = init_device_mesh(str(device.type), (world_size,))

        return cls.from_sparse_local(
            local_st, mesh, partition, global_shape=tuple(shape),
        )
    
    
    # =========================================================================
    # Properties
    # =========================================================================
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Global matrix shape."""
        return self._shape
    
    @property
    def num_partitions(self) -> int:
        """Number of partitions."""
        return self._num_partitions
    
    @property
    def device(self) -> torch.device:
        """Device of the matrix data."""
        return self._device
    
    @property
    def dtype(self) -> torch.dtype:
        """Data type of matrix values."""
        return self._local_tensor.values.dtype

    @property
    def nnz(self) -> int:
        """Number of non-zeros in this rank's local chunk (owned + halo
        rows). For the global nnz call :meth:`full_tensor` first."""
        return int(self._local_tensor.values.numel())
    
    
    
    # =========================================================================
    # Indexing and Iteration
    # =========================================================================
    
    
    
    
    # =========================================================================
    # Device Management
    # =========================================================================
    
    
    
    
    # =========================================================================
    # Distributed Operations
    # =========================================================================
    
    
    
    
    
    
    
    
    
    
    
    # =========================================================================
    # DTensor Utilities
    # =========================================================================
    
    
    
    
    
    # =========================================================================
    # Distributed Algorithms (True Distributed, No Gather)
    # =========================================================================
    
    
    
    
    
    
    
    
    
    
    
    # =========================================================================
    # Methods that require data gather (with warnings)
    # =========================================================================
    
    
    
    
    
    
    
    # =========================================================================
    # Matrix Operations
    # =========================================================================
    
    def __matmul__(self, x: Union[torch.Tensor, "DTensor"]) -> Union[torch.Tensor, "DTensor"]:
        """
        Distributed matrix-vector multiplication: y = D @ x
        
        Automatically handles scatter, distributed matvec, and gather.
        Supports gradient computation when values have requires_grad=True.
        
        Parameters
        ----------
        x : torch.Tensor or DTensor
            Global vector of shape (N,) where N = shape[1].
            - If torch.Tensor: treated as global vector (same on all ranks or single-node)
            - If DTensor: automatically handles distributed input/output
            
        Returns
        -------
        torch.Tensor or DTensor
            Global result vector of shape (M,) where M = shape[0].
            Returns DTensor if input is DTensor, otherwise torch.Tensor.
            
        Example
        -------
        >>> D = A.partition(num_partitions=4)
        >>> y = D @ x  # Equivalent to A @ x
        
        >>> # With DTensor input
        >>> from torch.distributed.tensor import DTensor, Replicate
        >>> x_dt = DTensor.from_local(x_local, mesh, [Replicate()])
        >>> y_dt = D @ x_dt  # Returns DTensor
        
        Notes
        -----
        **Gradient Support:**
        
        For single-node simulation with gradient support, uses global COO matvec.
        For true MPI distributed execution without gradients, uses partition-based matvec.
        
        **DTensor Support:**
        
        When input is a DTensor:
        - Replicated DTensor: extracts local tensor and computes as global
        - Sharded DTensor: redistributes to Replicate, computes, then reshards
        """
        # The matvec stays entirely in the ``Shard(0)`` space:
        # halo exchange in, local matvec, wrap result as
        # ``DTensor[Shard(0)]`` with the same mesh. No gather.
        if self._spec is not None and isinstance(self._spec.placement,
                                                  SparseShard):
            return self._matmul_spec(x)

        raise RuntimeError(
            "DSparseTensor.__matmul__ requires a SparseShard placement; "
            "build via DSparseTensor.partition(A, mesh) or "
            "DSparseTensor.from_sparse_local(...).")

    def _matmul_spec(self, x: Any) -> Any:
        """DTensor-mirror matvec dispatcher.

        Routes on the spec's ``SparseShard.axis`` (the placement
        carries which sparse axis is partitioned):

        * ``axis == 0`` -- row-partitioned: halo exchange + local SpMV
          → ``Shard(0)`` result (the original RowPartitioned path).
        * ``axis == 1`` -- col-partitioned: each rank has a column
          slice; local partial SpMV + ``dist.all_reduce(SUM)`` →
          ``Replicate()`` result (each rank ends up with the full y).

        Other axis values raise -- block-axis sharding isn't covered
        by SpMV in the same code path.
        """
        if self._local_tensor is None:
            raise RuntimeError(
                "DSparseTensor matvec requires a SparseTensor backing. "
                "Build via .partition(...) / .from_sparse_local(...).")

        placement = self._spec.placement
        if not isinstance(placement, SparseShard):
            raise RuntimeError(
                "_matmul_spec expects a SparseShard placement; got "
                f"{type(placement).__name__}")
        axis = placement.axis

        if axis == 0:
            return self._matmul_row_shard_via_sparse_tensor(x)
        if axis == 1:
            return self._matmul_col_shard(x)
        raise NotImplementedError(
            f"SparseShard(axis={axis}) matvec dispatch not implemented; "
            "axis must be 0 (row) or 1 (col) for SpMV.")


    def _matmul_row_shard_via_sparse_tensor(self, x: Any) -> Any:
        """Row-sharded matvec: pad owned→local, halo exchange, local
        SpMV via ``_local_tensor @ x``, slice back to the owned-row range.
        """
        partition = self._spec.placement.partition
        if partition is None:
            raise RuntimeError(
                "row-shard matvec requires spec.placement.partition; "
                "rebuild this DSparseTensor via from_sparse_local(...).")
        num_owned = int(partition.owned_nodes.numel())
        num_local = int(partition.local_to_global.numel())

        if _is_dtensor(x):
            x_local = x.to_local()
        else:
            x_local = x

        # Pad num_owned → num_local; halo will be filled by exchange.
        if x_local.shape[0] == num_owned:
            x_padded = torch.zeros(num_local, dtype=x_local.dtype,
                                    device=x_local.device)
            x_padded[:num_owned] = x_local
        elif x_local.shape[0] == num_local:
            x_padded = x_local
        else:
            raise ValueError(
                f"x has shape[0]={x_local.shape[0]}, expected num_owned "
                f"({num_owned}) or num_local ({num_local}).")

        self._halo_exchange_via_spec(x_padded, partition)
        # Local SpMV via SparseTensor's own __matmul__ (CSR-backed).
        y_full = self._local_tensor @ x_padded
        y_owned = y_full[:num_owned]

        if _is_dtensor(x):
            _DTensor = DTensor  # use module-level import with fallback
            return _DTensor.from_local(
                y_owned, self._spec.mesh, [Shard(0)])
        return y_owned

    def _halo_exchange_via_spec(self,
                                  x: torch.Tensor,
                                  partition: "Partition") -> torch.Tensor:
        """Halo exchange: read ``partition`` (from spec), exchange the
        ghost-node values with neighbour ranks in-place on ``x``.

        Per-rank send/recv buffers are cached on
        ``self._halo_send_buffers`` / ``self._halo_recv_buffers`` keyed
        by ``(neighbor_id, dtype)``.
        """
        if not DIST_AVAILABLE or not dist.is_initialized():
            return x

        device = x.device
        dtype = x.dtype

        send_bufs = {}
        recv_bufs = {}
        for nid in partition.neighbor_partitions:
            key = (nid, dtype)
            sb = self._halo_send_buffers.get(key)
            if sb is None:
                idx = partition.send_indices[nid].to(device=device,
                                                       dtype=torch.int64)
                sb = torch.empty(int(idx.numel()),
                                  dtype=dtype, device=device)
                self._halo_send_buffers[key] = sb
            send_bufs[nid] = sb
            rb = self._halo_recv_buffers.get(key)
            if rb is None:
                ridx = partition.recv_indices[nid].to(device=device,
                                                       dtype=torch.int64)
                rb = torch.empty(int(ridx.numel()),
                                  dtype=dtype, device=device)
                self._halo_recv_buffers[key] = rb
            recv_bufs[nid] = rb

        # Fill send buffers (gather from x at send_indices).
        for nid in partition.neighbor_partitions:
            send_idx = partition.send_indices[nid].to(device=device,
                                                       dtype=torch.int64)
            send_bufs[nid].copy_(x[send_idx])

        # Exchange. NCCL serialises P2P -- if every rank calls ``isend``
        # before any ``irecv``, both peers deadlock. Order the pairs by
        # comparing partition ids: lower id sends-then-recvs, higher id
        # recvs-then-sends. Gloo doesn't need it but it doesn't hurt.
        my_id = int(partition.partition_id)
        for nid in sorted(partition.neighbor_partitions, key=int):
            nid_i = int(nid)
            if my_id < nid_i:
                dist.send(send_bufs[nid], dst=nid_i)
                dist.recv(recv_bufs[nid], src=nid_i)
            else:
                dist.recv(recv_bufs[nid], src=nid_i)
                dist.send(send_bufs[nid], dst=nid_i)

        # Scatter recv buffers into x at recv_indices.
        for nid in partition.neighbor_partitions:
            recv_idx = partition.recv_indices[nid].to(device=device,
                                                       dtype=torch.int64)
            x[recv_idx] = recv_bufs[nid]
        return x

    def _matmul_col_shard(self, x: Any) -> Any:
        """Col-partitioned SpMV (``SparseShard(axis=1)``).

        **Status**: scaffolded but not yet implemented. Column
        partitioning needs a new ``SparseTensor.extract_column_partition``
        that slices columns into a local ``(M, num_owned_cols)``
        submatrix plus a 2-D mesh sharding shim so the halo *and* the
        ``all_reduce(SUM)`` semantics line up. Until then this dispatch
        raises so callers don't silently get wrong answers.
        """
        raise NotImplementedError(
            "SparseShard(axis=1) col-partitioned matvec is not yet "
            "implemented; use SparseShard(axis=0) (row-partitioned)."
        )


    # ====================================================================== #
    # Shard(0)-space distributed solve dispatcher.
    #
    # Every Krylov method below keeps every vector local (size
    # ``num_owned``)
    # and uses ``dist.all_reduce`` for the inner products that CG
    # needs. Matvec routes through ``_pad_owned_to_local`` so halo
    # entries are filled by ``halo_exchange`` per iteration.
    # ====================================================================== #
    def solve_distributed_shard(
        self,
        b: Any,
        *,
        method: Any = None,
        preconditioner: Any = None,
        atol: Any = None,
        rtol: Any = None,
        maxiter: Any = None,
        restart: Any = None,
        verbose: Any = None,
    ) -> Any:
        """Distributed Krylov solve entirely in Shard(0) space.

        Requires this :class:`DSparseTensor` to carry a real spec
        (built via :meth:`from_local`). The right-hand side ``b`` may
        be a ``DTensor[Shard(0)]`` (most common) or a raw
        ``torch.Tensor`` sized ``num_owned`` for the calling rank;
        the return value mirrors the input's wrapper.

        Methods (all live in Shard(0) space):

        * ``"cg"``       Saad §6.7 conjugate gradient -- SPD systems
        * ``"bicgstab"`` Saad §7.4 BiCGStab -- non-symmetric, no restart
        * ``"gmres"``    Saad §6.5 restarted GMRES(m) -- general
        * ``"fgmres"``   Saad §9.4 flexible GMRES(m) -- variable preconditioner
        * ``"minres"``   Paige-Saunders MINRES -- symmetric indefinite

        Inner products go through ``dist.all_reduce`` (sum), matvec
        through ``halo_exchange`` -- no rank ever sees a global vector.

        SolverConfig integration
        ------------------------
        Every kwarg defaults to ``None``, meaning "look at the active
        :class:`SolverConfig` scope on this thread, then fall back to
        the hard-coded default". The precedence chain matches
        :func:`solve` -- explicit kwarg → innermost scope → outer
        scopes (LIFO) → hard-coded default.

        >>> with SolverConfig(method="bicgstab", atol=1e-8):
        ...     x = D.solve_distributed_shard(b)        # picks BiCGStab + 1e-8
        ...     y = D.solve_distributed_shard(b, atol=1e-12)  # kwarg wins
        """
        if self._spec is None or not isinstance(
                self._spec.placement, SparseShard):
            raise RuntimeError(
                "solve_distributed_shard() requires a DSparseTensor "
                "with SparseShard placement -- build one via "
                "DSparseTensor.from_local(local, mesh, ...) or "
                "DSparseTensor.partition(A, mesh, ...)."
            )
        if not (DIST_AVAILABLE and dist.is_initialized()):
            raise RuntimeError(
                "solve_distributed_shard() requires torch.distributed "
                "to be initialised."
            )

        # Merge with active SolverConfig scope. Explicit kwargs (non-
        # ``None``) win; otherwise we walk the scope stack via
        # ``solve._active_defaults`` and fall back to hard-coded.
        from .solve import _active_defaults
        defaults = _active_defaults()
        def _pick(value, name, hardcoded):
            if value is not None:
                return value
            if name in defaults:
                return defaults[name]
            return hardcoded
        method  = _pick(method,  "method",  "cg")
        atol    = _pick(atol,    "atol",    1e-10)
        rtol    = _pick(rtol,    "rtol",    0.0)
        maxiter = _pick(maxiter, "maxiter", 1000)
        restart = restart if restart is not None else 30  # not in SolverConfig
        verbose = _pick(verbose, "verbose", False)
        # ``preconditioner`` is special-cased in SolverConfig because
        # ``None`` is a legitimate "no preconditioning" choice (the
        # _UNSET sentinel distinguishes that from "not set"). Mirror
        # that here: explicit ``None`` means identity precond.
        if preconditioner is None and "preconditioner" in defaults:
            preconditioner = defaults["preconditioner"]
        from . import distributed_solve as _ds
        M_apply = _ds.make_preconditioner(self, preconditioner)

        if _is_dtensor(b):
            b_owned = b.to_local()
            wrap_output = True
        else:
            b_owned = b
            wrap_output = False

        method_l = method.lower()
        common = dict(M_apply=M_apply, atol=atol, rtol=rtol,
                       maxiter=maxiter, verbose=verbose)
        if method_l in ("cg", "pcg"):
            x_owned = _ds.cg_shard(self, b_owned, **common)
        elif method_l == "bicgstab":
            x_owned = _ds.bicgstab_shard(self, b_owned, **common)
        elif method_l in ("gmres", "fgmres"):
            x_owned = _ds.gmres_shard(
                self, b_owned, restart=restart,
                flexible=(method_l == "fgmres"), **common)
        elif method_l == "minres":
            x_owned = _ds.minres_shard(self, b_owned, **common)
        else:
            raise ValueError(
                f"Unknown distributed solve method {method!r}; expected "
                "one of cg, bicgstab, gmres, fgmres, minres."
            )

        if wrap_output:
            _DTensor = DTensor  # use module-level import with fallback
            return _DTensor.from_local(
                x_owned, self._spec.mesh, [Shard(0)])
        return x_owned

    # ------------------------------------------------------------------ #
    # Shard(0)-space primitives reused by every Krylov method.
    # ------------------------------------------------------------------ #
    def _shard_dot(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Global inner product over Shard(0) vectors: local ``torch.dot``
        then ``dist.all_reduce(SUM)`` across the mesh."""
        d = torch.dot(u, v)
        dist.all_reduce(d, op=dist.ReduceOp.SUM)
        return d

    def _shard_norm(self, u: torch.Tensor) -> torch.Tensor:
        return self._shard_dot(u, u).sqrt()

    def _num_owned(self) -> int:
        """Owned-row count for the Shard(0) layout."""
        return int(self._spec.placement.partition.owned_nodes.numel())

    def _shard_matvec(self, x_owned: torch.Tensor) -> torch.Tensor:
        """Apply ``A`` to a Shard(0) vector: pad owned→local, halo
        exchange, local SpMV via a cached CSR multiply, slice back to
        the owned-row range.

        Used by every Shard(0)-space Krylov method (CG / BiCGStab /
        GMRES / FGMRES / MINRES) and the polynomial preconditioner.
        ``SparseTensor.__matmul__`` would fall back to a COO
        gather+scatter_add SpMV (~2× slower than torch's CSR kernel
        on CPU at 64k DOF), so we cache a single CSR per partition.
        """
        partition = self._spec.placement.partition
        num_owned = int(partition.owned_nodes.numel())
        num_local = int(partition.local_to_global.numel())
        if x_owned.shape[0] == num_owned:
            x_padded = torch.zeros(num_local, dtype=x_owned.dtype,
                                    device=x_owned.device)
            x_padded[:num_owned] = x_owned
        elif x_owned.shape[0] == num_local:
            x_padded = x_owned
        else:
            raise ValueError(
                f"x shape[0]={x_owned.shape[0]}, expected "
                f"num_owned={num_owned} or num_local={num_local}")
        self._halo_exchange_via_spec(x_padded, partition)

        # Cache the local CSR on first matvec; subsequent iters re-use it.
        csr = getattr(self, "_local_csr_cache", None)
        if csr is None:
            st = self._local_tensor
            indices = torch.stack([st.row_indices.to(torch.int64),
                                    st.col_indices.to(torch.int64)])
            coo = torch.sparse_coo_tensor(indices, st.values,
                                           tuple(st.shape)).coalesce()
            csr = coo.to_sparse_csr()
            self._local_csr_cache = csr

        # ``torch.mv(csr, x)`` is the fused CSR · 1-D dense kernel and
        # is materially faster than ``csr @ x.unsqueeze(-1)`` on CPU.
        y_full = torch.mv(csr, x_padded)
        return y_full[:num_owned]

    # ------------------------------------------------------------------ #
    # The preconditioner factory + four Krylov methods (CG / BiCGStab /
    # GMRES / FGMRES / MINRES) live in :mod:`torch_sla.distributed_solve`
    # as free functions taking ``self`` as ``D``. ``solve_distributed_shard``
    # above dispatches to them.
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        return (f"DSparseTensor(shape={tuple(self._shape)}, "
                f"num_partitions={self._num_partitions}, "
                f"local_nnz={self.nnz}, device={self._device})")

