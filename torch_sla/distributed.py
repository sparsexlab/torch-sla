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


    def save(
        self,
        directory: Any,
        rank: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        """Persist this rank's shard to ``directory``. Convenience for
        :func:`torch_sla.io.save_dsparse(self, directory)`."""
        from .io import save_dsparse
        save_dsparse(self, directory, rank=rank, verbose=verbose)

    @classmethod
    def load(
        cls,
        directory: Any,
        mesh: Any = None,
        rank: Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
    ) -> "DSparseTensor":
        """Reconstruct this rank's :class:`DSparseTensor` from a
        directory previously written by :meth:`save` /
        :func:`save_dsparse`. Convenience for
        :func:`torch_sla.io.load_dsparse(directory, mesh, rank, device)`.
        """
        from .io import load_dsparse
        return load_dsparse(directory, mesh=mesh, rank=rank, device=device)


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
        rows). For the global nnz across all ranks call
        :meth:`global_nnz`."""
        return int(self._local_tensor.values.numel())

    def global_nnz(self) -> int:
        """Number of non-zeros summed across every rank.

        Triggers a single ``dist.all_reduce(SUM)`` (cached after the
        first call). For a single-process ``DSparseTensor`` returns the
        same value as :attr:`nnz`.
        """
        cached = getattr(self, "_global_nnz_cache", None)
        if cached is not None:
            return cached
        local = torch.tensor([self.nnz], dtype=torch.long, device=self._device)
        if DIST_AVAILABLE and dist.is_initialized():
            dist.all_reduce(local, op=dist.ReduceOp.SUM)
        total = int(local.item())
        self._global_nnz_cache = total
        return total

    # ---- Tensor-mirror properties (parity with SparseTensor) ----

    @property
    def ndim(self) -> int:
        """Number of dimensions. ``DSparseTensor`` is always 2-D."""
        return 2

    @property
    def sparse_shape(self) -> Tuple[int, int]:
        """The ``(M, N)`` sparse matrix dimensions -- same as
        :attr:`shape` for ``DSparseTensor`` (no batch / block axes)."""
        return self._shape

    @property
    def sparse_dim(self) -> Tuple[int, int]:
        """The dimensions that are sparse, ``(0, 1)``."""
        return (0, 1)

    @property
    def batch_shape(self) -> Tuple[int, ...]:
        """Batch dimensions. Always ``()`` for ``DSparseTensor``."""
        return ()

    @property
    def block_shape(self) -> Tuple[int, ...]:
        """Block dimensions. Always ``()`` for ``DSparseTensor``."""
        return ()

    @property
    def batch_size(self) -> int:
        """Total number of batch elements. Always ``1`` for
        ``DSparseTensor``."""
        return 1

    @property
    def is_batched(self) -> bool:
        """Always ``False`` for ``DSparseTensor``."""
        return False

    @property
    def is_block(self) -> bool:
        """Always ``False`` for ``DSparseTensor``."""
        return False

    @property
    def is_cuda(self) -> bool:
        """Whether this rank's local chunk lives on CUDA."""
        return self._device.type == "cuda"

    @property
    def is_square(self) -> bool:
        """Whether the global sparse dimensions are square (M == N)."""
        M, N = self._shape
        return M == N

    @property
    def values(self) -> torch.Tensor:
        """This rank's local non-zero values (owned + halo rows)."""
        return self._local_tensor.values

    @property
    def row_indices(self) -> torch.Tensor:
        """This rank's local row indices (in local coords)."""
        return self._local_tensor.row_indices

    @property
    def col_indices(self) -> torch.Tensor:
        """This rank's local col indices (in local coords)."""
        return self._local_tensor.col_indices

    # =========================================================================
    # Indexing and Iteration
    # =========================================================================
    
    
    
    
    # =========================================================================
    # Device Management
    # =========================================================================

    def to(
        self,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "DSparseTensor":
        """Return a new :class:`DSparseTensor` with this rank's local
        chunk (and its :class:`Partition` index tensors) moved to
        ``device`` and/or cast to ``dtype``. Parity with
        :meth:`SparseTensor.to`.

        The :class:`DeviceMesh` is left untouched -- it represents the
        process group, which is independent of per-rank tensor device.
        Halo send/recv buffer caches are reset since they live on the
        old device.
        """
        if device is None and dtype is None:
            return self
        if isinstance(device, str):
            device = torch.device(device)

        new_local = self._local_tensor.to(device=device, dtype=dtype)

        placement = self._spec.placement
        new_partition = (
            placement.partition.to(device)
            if device is not None and placement.partition is not None
            else placement.partition
        )

        out = type(self).__new__(type(self))
        out._values = None
        out._row_indices = None
        out._col_indices = None
        out._shape = self._shape
        out._num_partitions = self._num_partitions
        out._coords = self._coords
        out._partition_method = self._partition_method
        out._verbose = self._verbose
        out._device = new_local.values.device
        out._local_tensor = new_local
        out._halo_send_buffers = {}
        out._halo_recv_buffers = {}
        new_placement = SparseShard(
            axis=placement.axis, partition=new_partition,
        )
        out._spec = DSparseSpec(
            placement=new_placement, mesh=self._spec.mesh,
            global_shape=self._spec.global_shape,
        )
        return out

    def cuda(self, device: Optional[int] = None) -> "DSparseTensor":
        """Move this rank's local chunk to CUDA. ``device`` selects the
        CUDA device index (default: current)."""
        if device is None:
            return self.to("cuda")
        return self.to(f"cuda:{device}")

    def cpu(self) -> "DSparseTensor":
        """Move this rank's local chunk to CPU."""
        return self.to("cpu")

    def float(self) -> "DSparseTensor":
        """Cast local values to float32."""
        return self.to(dtype=torch.float32)

    def double(self) -> "DSparseTensor":
        """Cast local values to float64."""
        return self.to(dtype=torch.float64)

    def half(self) -> "DSparseTensor":
        """Cast local values to float16."""
        return self.to(dtype=torch.float16)

    # =========================================================================
    # Reductions (cross-rank via all_reduce)
    # =========================================================================
    #
    # Semantics mirror ``SparseTensor``: reductions cover only the
    # explicitly stored non-zero values -- implicit zeros are ignored.
    # Each rank's ``_local_tensor`` holds disjoint owned rows (no halo
    # rows in the COO), so summing local results across ranks gives the
    # global value with no double-counting.

    def _all_reduce_scalar(
        self,
        value: torch.Tensor,
        op: "dist.ReduceOp" = None,
    ) -> torch.Tensor:
        """In-place ``dist.all_reduce`` on a 0-d tensor (no-op when no
        process group is initialised)."""
        if op is None:
            op = dist.ReduceOp.SUM if DIST_AVAILABLE else None
        if DIST_AVAILABLE and dist.is_initialized():
            dist.all_reduce(value, op=op)
        return value

    def sum(
        self,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdim: bool = False,
    ) -> torch.Tensor:
        """Sum of stored values.

        * ``axis=None`` -- scalar, single ``all_reduce(SUM)`` across ranks.
        * ``axis=0`` -- length-N column sums (dense), per-rank
          scatter-add into global col bins + ``all_reduce(SUM)``.
        * ``axis=1`` -- length-M row sums (dense), same with row bins.

        Implicit zeros are ignored, matching :meth:`SparseTensor.sum`.
        """
        local = self._local_tensor
        if axis is None:
            total = local.values.detach().clone() \
                if not local.values.requires_grad else local.values.clone()
            total = total.sum()
            return self._all_reduce_scalar(total, dist.ReduceOp.SUM)

        if axis in (0, -2):
            # Column sums: scatter local values into the global N-vector
            # by their global column index.
            M, N = self._shape
            partition = self._spec.placement.partition
            local_col_global = partition.local_to_global[local.col_indices]
            out = torch.zeros(N, dtype=local.values.dtype,
                              device=local.values.device)
            out.scatter_add_(0, local_col_global, local.values)
            if DIST_AVAILABLE and dist.is_initialized():
                dist.all_reduce(out, op=dist.ReduceOp.SUM)
            return out.unsqueeze(0) if keepdim else out

        if axis in (1, -1):
            # Row sums: each rank's owned rows are disjoint, so the
            # all_reduce just fills in zeros from non-owning ranks.
            M, N = self._shape
            partition = self._spec.placement.partition
            local_row_global = partition.local_to_global[local.row_indices]
            out = torch.zeros(M, dtype=local.values.dtype,
                              device=local.values.device)
            out.scatter_add_(0, local_row_global, local.values)
            if DIST_AVAILABLE and dist.is_initialized():
                dist.all_reduce(out, op=dist.ReduceOp.SUM)
            return out.unsqueeze(1) if keepdim else out

        raise ValueError(f"DSparseTensor.sum: axis {axis} out of range "
                         f"(only None, 0, 1 supported on a 2-D tensor)")

    def mean(
        self,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
    ) -> torch.Tensor:
        """Mean of stored non-zero values (matches ``SparseTensor`` --
        implicit zeros excluded). For ``axis=None``, returns
        ``sum / global_nnz``."""
        total = self.sum(axis=axis)
        if axis is None:
            return total / max(1, self.global_nnz())
        # Element-wise: divide each col/row sum by the global count of
        # stored entries in that col/row. Requires a parallel reduction
        # over indices.
        local = self._local_tensor
        partition = self._spec.placement.partition
        M, N = self._shape
        counts = torch.zeros(
            N if axis in (0, -2) else M,
            dtype=torch.long, device=local.values.device,
        )
        idx_global = (partition.local_to_global[local.col_indices]
                      if axis in (0, -2)
                      else partition.local_to_global[local.row_indices])
        counts.scatter_add_(
            0, idx_global, torch.ones_like(idx_global),
        )
        if DIST_AVAILABLE and dist.is_initialized():
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        counts = counts.clamp_(min=1).to(total.dtype)
        return total / counts

    def prod(self) -> torch.Tensor:
        """Product of all stored values across every rank.

        ``ReduceOp.PROD`` is unsupported by the gloo backend, so we
        ``all_gather`` per-rank scalars and multiply locally instead --
        works on every backend.
        """
        local_p = self._local_tensor.values.prod()
        if not (DIST_AVAILABLE and dist.is_initialized()):
            return local_p
        world = dist.get_world_size()
        gathered = [torch.zeros_like(local_p) for _ in range(world)]
        dist.all_gather(gathered, local_p)
        return torch.stack(gathered).prod()

    def max(self) -> torch.Tensor:
        """Max of stored values (single ``all_reduce(MAX)``). Implicit
        zeros ignored."""
        local_m = self._local_tensor.values.max()
        return self._all_reduce_scalar(local_m, dist.ReduceOp.MAX)

    def min(self) -> torch.Tensor:
        """Min of stored values (single ``all_reduce(MIN)``). Implicit
        zeros ignored."""
        local_m = self._local_tensor.values.min()
        return self._all_reduce_scalar(local_m, dist.ReduceOp.MIN)

    def norm(self, ord: Any = "fro") -> torch.Tensor:
        """Matrix norm of the global matrix.

        Supported orders:

        * ``'fro'`` (default) -- Frobenius norm,
          ``sqrt(sum(|v|^2))``.  One ``all_reduce(SUM)``.
        * ``1`` -- max absolute column sum.  Two reductions: per-col
          ``all_reduce(SUM)`` followed by a global ``max`` on rank-local
          dense vector.
        * ``float('inf')`` -- max absolute row sum.

        ``ord=2`` (spectral norm) is not implemented for distributed
        matrices in this PR -- call :meth:`full_tensor().norm(2)` for
        a single-process fallback.
        """
        if ord == "fro":
            v = self._local_tensor.values
            if v.is_complex():
                local_sq = (v.real ** 2 + v.imag ** 2).sum()
            else:
                local_sq = (v.float() ** 2).sum() \
                    if v.dtype in (torch.float16, torch.bfloat16) \
                    else (v ** 2).sum()
            total = self._all_reduce_scalar(local_sq, dist.ReduceOp.SUM)
            return total.sqrt()

        if ord == 1:
            col_sums = self._abs_axis_sum(axis=0)
            return col_sums.max()

        if ord == float("inf"):
            row_sums = self._abs_axis_sum(axis=1)
            return row_sums.max()

        if ord == 2:
            raise NotImplementedError(
                "DSparseTensor.norm(2) (spectral norm) is not yet "
                "implemented; call .full_tensor().norm(2) for a "
                "single-process fallback."
            )

        raise ValueError(f"Unsupported norm order: {ord!r}")

    def _abs_axis_sum(self, axis: int) -> torch.Tensor:
        """Helper: ``|·|`` then ``sum(axis=axis)``. Returns dense
        vector after global all_reduce."""
        local = self._local_tensor
        partition = self._spec.placement.partition
        M, N = self._shape
        abs_v = local.values.abs()
        if axis in (0, -2):
            out = torch.zeros(N, dtype=abs_v.dtype, device=abs_v.device)
            idx = partition.local_to_global[local.col_indices]
        elif axis in (1, -1):
            out = torch.zeros(M, dtype=abs_v.dtype, device=abs_v.device)
            idx = partition.local_to_global[local.row_indices]
        else:
            raise ValueError(f"axis must be 0 or 1, got {axis}")
        out.scatter_add_(0, idx, abs_v)
        if DIST_AVAILABLE and dist.is_initialized():
            dist.all_reduce(out, op=dist.ReduceOp.SUM)
        return out


    
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

        Buffers + send/recv index tensors are cached on the instance so
        the hot path does nothing but ``torch.index_select`` (gather),
        ``batch_isend_irecv`` (async NCCL), wait, ``index_copy_``
        (scatter). Per-rank caches are keyed by ``(neighbor_id, dtype)``.
        """
        if not DIST_AVAILABLE or not dist.is_initialized():
            return x

        device = x.device
        dtype = x.dtype

        # Cache buffers + idx tensors on first call. After that, every
        # matvec iter hits the cache.
        send_bufs, recv_bufs = {}, {}
        send_idxs, recv_idxs = {}, {}
        for nid in partition.neighbor_partitions:
            key = (nid, dtype)
            entry = self._halo_send_buffers.get(key)
            if entry is None:
                idx = partition.send_indices[nid].to(device=device,
                                                       dtype=torch.int64)
                buf = torch.empty(int(idx.numel()),
                                   dtype=dtype, device=device)
                entry = (buf, idx)
                self._halo_send_buffers[key] = entry
            send_bufs[nid], send_idxs[nid] = entry

            entry = self._halo_recv_buffers.get(key)
            if entry is None:
                ridx = partition.recv_indices[nid].to(device=device,
                                                       dtype=torch.int64)
                buf = torch.empty(int(ridx.numel()),
                                   dtype=dtype, device=device)
                entry = (buf, ridx)
                self._halo_recv_buffers[key] = entry
            recv_bufs[nid], recv_idxs[nid] = entry

        # Gather: send_buf <- x[send_idx]. Fused on GPU.
        for nid in partition.neighbor_partitions:
            torch.index_select(x, 0, send_idxs[nid], out=send_bufs[nid])

        # ``batch_isend_irecv`` schedules every send/recv in one go.
        # NCCL serialises P2P under the hood but accepts a batched
        # plan without deadlocking, and lets the CPU return immediately
        # so subsequent kernel launches can overlap with the halo
        # comm on the same device. Falls back to the legacy synchronous
        # ordering on gloo (gloo + NCCL P2P don't share the batched API).
        backend = dist.get_backend() if dist.is_initialized() else "gloo"
        my_id = int(partition.partition_id)
        ordered = sorted(partition.neighbor_partitions, key=int)
        if backend == "nccl":
            ops = []
            for nid in ordered:
                ops.append(dist.P2POp(dist.isend, send_bufs[nid], int(nid)))
                ops.append(dist.P2POp(dist.irecv, recv_bufs[nid], int(nid)))
            for req in dist.batch_isend_irecv(ops):
                req.wait()
        else:
            for nid in ordered:
                nid_i = int(nid)
                if my_id < nid_i:
                    dist.send(send_bufs[nid], dst=nid_i)
                    dist.recv(recv_bufs[nid], src=nid_i)
                else:
                    dist.recv(recv_bufs[nid], src=nid_i)
                    dist.send(send_bufs[nid], dst=nid_i)

        # Scatter: x[recv_idx] <- recv_buf. ``index_copy_`` is in-place
        # and faster than fancy-indexed assignment on CUDA.
        for nid in partition.neighbor_partitions:
            x.index_copy_(0, recv_idxs[nid], recv_bufs[nid])
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

        Buffers cached on the instance:
        * ``_local_csr_cache`` -- the local CSR.
        * ``_x_padded_cache``  -- the (num_local,) input scratch.

        Per-iter work: ``index_copy_`` (small) + halo exchange +
        ``torch.mv`` + slice -- no large allocations.

        We tried an interior/boundary split with overlapped halo to
        hide NCCL latency behind the interior SpMV; in practice it
        degraded perf 5-15% (two SpMV launches + sparse pattern
        duplication + PyTorch cross-stream sync ate the comm savings)
        and was reverted. The CSR-cache + cached pad + async batched
        NCCL P2P is the local optimum.
        """
        partition = self._spec.placement.partition
        num_owned = int(partition.owned_nodes.numel())
        num_local = int(partition.local_to_global.numel())
        dtype = x_owned.dtype
        device = x_owned.device

        if x_owned.shape[0] == num_owned:
            xp = getattr(self, "_x_padded_cache", None)
            if (xp is None or xp.shape[0] != num_local
                    or xp.dtype != dtype or xp.device != device):
                xp = torch.zeros(num_local, dtype=dtype, device=device)
                self._x_padded_cache = xp
            xp[:num_owned].copy_(x_owned)
            x_padded = xp
        elif x_owned.shape[0] == num_local:
            x_padded = x_owned
        else:
            raise ValueError(
                f"x shape[0]={x_owned.shape[0]}, expected "
                f"num_owned={num_owned} or num_local={num_local}")
        self._halo_exchange_via_spec(x_padded, partition)

        # Cache the local CSR on first matvec; subsequent iters re-use it.
        # int32 indices halve the col_indices storage (and improve cuSPARSE
        # L1 cache hit rate) when num_local < 2^31. Fall back to int64 only
        # if the matrix is genuinely too large to address with int32.
        csr = getattr(self, "_local_csr_cache", None)
        if csr is None:
            st = self._local_tensor
            indices = torch.stack([st.row_indices.to(torch.int64),
                                    st.col_indices.to(torch.int64)])
            coo = torch.sparse_coo_tensor(indices, st.values,
                                           tuple(st.shape)).coalesce()
            csr64 = coo.to_sparse_csr()
            idx_dtype = (torch.int32
                          if num_local < 2_147_483_647
                          else torch.int64)
            if idx_dtype is torch.int32:
                csr = torch.sparse_csr_tensor(
                    csr64.crow_indices().to(idx_dtype),
                    csr64.col_indices().to(idx_dtype),
                    csr64.values(),
                    csr64.size(),
                )
            else:
                csr = csr64
            self._local_csr_cache = csr

        # Pre-allocate output buffer; required for CUDA Graphs capture
        # (no allocations allowed inside graph). torch.mv on sparse CSR
        # supports ``out=`` and is slightly faster than the no-out
        # variant on torch 2.1.
        yf = getattr(self, "_y_full_cache", None)
        if (yf is None or yf.shape[0] != num_local
                or yf.dtype != dtype or yf.device != device):
            yf = torch.empty(num_local, dtype=dtype, device=device)
            self._y_full_cache = yf
        torch.mv(csr, x_padded, out=yf)
        return yf[:num_owned]

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

