"""
Distributed Sparse Matrix for large-scale CFD/FEM computations.

Provides domain decomposition with halo exchange, following the standard
approach used in Ansys, OpenFOAM, and other industrial CFD/FEM solvers.

Key Features:
- Graph-based partitioning (METIS or simple geometric methods)
- Halo/ghost node exchange for parallel computations
- Support for both CPU and CUDA devices
- Same API as SparseTensor for easy migration

Example
-------
>>> from torch_sla import DSparseMatrix
>>> 
>>> # Create from global matrix
>>> A_global = SparseTensor(val, row, col, shape)
>>> A_dist = DSparseMatrix.from_global(A_global, num_partitions=4)
>>> 
>>> # Distributed solve
>>> x_dist = A_dist.solve(b_dist)
>>> 
>>> # Halo exchange for iterative methods
>>> A_dist.halo_exchange(local_x)
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

# DTensor support (PyTorch 2.0+)
try:
    from torch.distributed.tensor import DTensor
    from torch.distributed.tensor.placement_types import Shard, Replicate
    DTENSOR_AVAILABLE = True
except ImportError:
    try:
        # Older import path (PyTorch 2.0-2.1)
        from torch.distributed._tensor import DTensor
        from torch.distributed._tensor.placement_types import Shard, Replicate
        DTENSOR_AVAILABLE = True
    except ImportError:
        DTENSOR_AVAILABLE = False
        DTensor = None
        Shard = None
        Replicate = None


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

    The ``partition`` field is the **target** state of the DSparseMatrix
    dissolution: it eventually carries ``owned_nodes`` /
    ``halo_nodes`` / ``neighbor_partitions`` so the placement is the
    single source of truth for the irregular shard map. Today it
    defaults to ``None`` and the per-rank ``DSparseMatrix.partition``
    still owns that data; A future release will swap the two so
    ``_local_tensor`` becomes a plain :class:`SparseTensor` and the
    placement carries the map.
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
    
    def __init__(
        self,
        values: torch.Tensor,
        row_indices: torch.Tensor,
        col_indices: torch.Tensor,
        shape: Tuple[int, int],
        num_partitions: int,
        coords: Optional[torch.Tensor] = None,
        partition_method: str = 'auto',
        device: Optional[Union[str, torch.device]] = None,
        verbose: bool = True
    ):
        self._values = values
        self._row_indices = row_indices
        self._col_indices = col_indices
        self._shape = shape
        self._num_partitions = num_partitions
        self._coords = coords
        self._partition_method = partition_method
        self._verbose = verbose
        
        # Infer device from input tensor if not explicitly specified
        if device is None:
            device = values.device
        if isinstance(device, str):
            device = torch.device(device)
        self._device = device
        
        # Compute partition IDs
        # NOTE: In distributed mode, this should be computed on rank 0 and broadcast
        # to ensure consistency. See _compute_partitions_distributed() for distributed-safe version.
        self._partition_ids = self._compute_partitions(partition_method, coords)
        
        # Create all partitions
        self._partitions: List[DSparseMatrix] = []
        self._create_partitions()

        # DTensor-mirror state. Populated by :meth:`from_local` and
        # related classmethods that wrap an already-distributed local
        # matrix; ``None`` for the legacy single-process simulator
        # constructor so existing call sites keep behaving identically.
        self._local_matrix: Optional[DSparseMatrix] = None
        # ``_local_tensor`` is the per-rank chunk -- a
        # plain :class:`SparseTensor` in local coordinates. When set,
        # matvec uses this + ``_spec.placement.partition`` and bypasses
        # ``DSparseMatrix`` entirely. Built via
        # :meth:`from_sparse_local`.
        self._local_tensor: Optional["SparseTensor"] = None
        # Per-rank caches for halo-exchange (populated lazily by
        # ``_halo_exchange_via_spec``).
        self._halo_send_buffers: Dict[int, torch.Tensor] = {}
        self._halo_recv_buffers: Dict[int, torch.Tensor] = {}
        self._spec: Optional[DSparseSpec] = None
    
    def _compute_partitions(
        self, 
        method: str, 
        coords: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Compute partition assignments for each node."""
        if method == 'auto':
            if coords is not None:
                method = 'rcb'
            else:
                method = 'metis'
        
        if method == 'metis':
            return partition_graph_metis(
                self._row_indices, self._col_indices, 
                self._shape[0], self._num_partitions
            )
        elif method in ['rcb', 'slicing']:
            if coords is None:
                raise ValueError(f"Partition method '{method}' requires coords")
            return partition_coordinates(coords, self._num_partitions, method=method)
        elif method == 'simple':
            return partition_simple(self._shape[0], self._num_partitions)
        else:
            raise ValueError(f"Unknown partition method: {method}")
    
    def _create_partitions(self):
        """Create all partition matrices."""
        for i in range(self._num_partitions):
            mat = DSparseMatrix._from_global_impl(
                self._values, self._row_indices, self._col_indices,
                self._shape, self._num_partitions, i,
                partition_ids=self._partition_ids,
                device=self._device,
                verbose=self._verbose
            )
            self._partitions.append(mat)
        
        # Store reference to all partitions for local halo exchange
        for mat in self._partitions:
            mat._all_partitions = [m.partition for m in self._partitions]
    
    # ====================================================================== #
    # DTensor-mirror API: from_local / to_local / full_tensor / redistribute.
    #
    # These methods give DSparseTensor the same shape of API as
    # ``torch.distributed.tensor.DTensor``: every call resolves through
    # a private :class:`DSparseSpec` that bundles the placement, the
    # device mesh, and the global shape. Vectors crossing the API stay
    # as ``DTensor[Shard(0)]`` so the rest of the PyTorch distributed
    # ecosystem (FSDP, TP, DCP) composes for free.
    #
    # The legacy single-process simulator entry points
    # (``DSparseTensor(values, row, col, shape, num_partitions=...)``,
    # ``from_sparse_tensor``) leave ``_local_matrix`` / ``_spec`` as
    # ``None`` -- callers that don't want the DTensor mirror keep
    # working byte-for-byte.
    # ====================================================================== #
    @classmethod
    def from_local(
        cls,
        local_matrix: "DSparseMatrix",
        mesh: Any,
        *,
        placement: SparsePlacement = None,
        global_shape: Optional[Tuple[int, int]] = None,
    ) -> "DSparseTensor":
        """Wrap an already-distributed :class:`DSparseMatrix` chunk as a
        DTensor-mirror :class:`DSparseTensor`.

        Mirrors :meth:`torch.distributed.tensor.DTensor.from_local`. Each
        rank passes the local subdomain it received from
        :meth:`DSparseTensor.from_global_distributed` /
        :meth:`from_device_mesh`; the placement / mesh metadata is
        carried in :class:`DSparseSpec` for downstream dispatch.

        Parameters
        ----------
        local_matrix : DSparseMatrix
            This rank's owned + halo subdomain.
        mesh : DeviceMesh
            The PyTorch device mesh the matrix is distributed over.
        placement : SparsePlacement, optional
            Defaults to :class:`RowPartitioned` (rows split across the
            mesh, which is what ``from_global_distributed`` produces).
            Pass :class:`Replicated` for the full-matrix-on-every-rank
            layout.
        global_shape : Tuple[int, int], optional
            Override the deduced global shape. Defaults to
            ``local_matrix.global_shape``.
        """
        if placement is None:
            placement = RowPartitioned()
        if global_shape is None:
            global_shape = tuple(local_matrix.global_shape)

        # We deliberately bypass __init__ -- the legacy constructor
        # builds an in-process Python list of every partition, which is
        # the opposite of what we want here. Build a minimal instance
        # and stamp the DTensor-mirror state on directly.
        self = cls.__new__(cls)
        self._values = None
        self._row_indices = None
        self._col_indices = None
        self._shape = global_shape
        self._num_partitions = mesh.size() if mesh is not None else 1
        self._coords = None
        self._partition_method = None
        self._verbose = False
        self._device = local_matrix.device
        self._partition_ids = None
        self._partitions = []
        self._local_matrix = local_matrix
        self._local_tensor = None
        self._halo_send_buffers = {}
        self._halo_recv_buffers = {}
        self._spec = DSparseSpec(placement=placement, mesh=mesh,
                                 global_shape=global_shape)
        return self

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
        """Constructor for the SparseTensor-backed path: wrap a plain
        :class:`SparseTensor` (per-rank local chunk in local coords)
        plus its :class:`Partition` as a DSparseTensor whose
        ``_local_tensor`` field is set instead of ``_local_matrix``.

        Use this together with :meth:`SparseTensor.extract_partition` to
        build a distributed tensor without going through
        :class:`DSparseMatrix`:

        .. code-block:: python

            partition = compute_partition(...)
            local_tensor = A_global.extract_partition(partition)
            D = DSparseTensor.from_sparse_local(
                local_tensor, mesh, partition,
                global_shape=A_global.shape,
            )
            y_dt = D @ x_dt              # routes through SparseTensor path

        Compared to :meth:`from_local`, this path leaves
        ``_local_matrix=None`` and stamps the partition onto
        ``_spec.placement.partition`` so the placement is the single
        source of truth for the irregular shard map. The matvec
        dispatch picks this path automatically when
        ``_local_tensor is not None``.

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
        self._partition_ids = None
        self._partitions = []
        self._local_matrix = None
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

        # Skip DSparseMatrix: compute partition ids, build the Partition
        # struct, extract the local SparseTensor, wrap. No legacy code.
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
        from torch.distributed.tensor import DTensor as _DTensor, Shard
        return _DTensor.from_local(local_slice, self._spec.mesh,
                                    [Shard(0)])

    def _partition_for_dispatch(self) -> Optional["Partition"]:
        """Return the active :class:`Partition` regardless of which
        local backing is set. ``None`` for the legacy simulator
        constructor (no real spec)."""
        if self._spec is not None and isinstance(
                self._spec.placement, SparseShard) \
                and self._spec.placement.partition is not None:
            return self._spec.placement.partition
        if self._local_matrix is not None:
            return self._local_matrix.partition
        return None

    def to_local(self) -> "DSparseMatrix":
        """Return this rank's local :class:`DSparseMatrix` chunk.

        Mirrors :meth:`DTensor.to_local`. For the legacy single-process
        simulator instance the call falls back to the gathered global
        matrix (logically equivalent: the "local" tensor IS the global
        tensor when there's only one rank)."""
        if self._local_matrix is not None:
            return self._local_matrix
        if not self._partitions:
            raise RuntimeError(
                "DSparseTensor has neither a DTensor-mirror local "
                "matrix nor a simulated partition list -- nothing to "
                "hand back as 'local'."
            )
        # Single-process simulator path: rank 0 owns everything.
        return self._partitions[0]

    def full_tensor(self) -> "SparseTensor":
        """Materialise the full global matrix on every rank.

        Mirrors :meth:`DTensor.full_tensor` (the symmetric ``Allgather``
        return -- every rank ends up with the same result). Returns a
        plain :class:`SparseTensor` so callers can pipe it into the
        non-distributed code path.

        Cheap when the placement is already :class:`Replicated` (just a
        format conversion); for :class:`RowPartitioned` we Allgather
        the global COO triples across the mesh.
        """
        from .sparse_tensor import SparseTensor

        if self._spec is None:
            # Single-process simulator: the original COO triple still
            # lives on the instance; convert directly.
            return self.to_sparse_tensor()

        if isinstance(self._spec.placement, Replicated):
            # Every rank already has the full matrix; rebuild a global
            # SparseTensor from the local one's COO state.
            local = self._local_matrix
            return SparseTensor(local.local_values, local.local_row,
                                local.local_col, self._spec.global_shape)

        if not (DIST_AVAILABLE and dist.is_initialized()):
            raise RuntimeError(
                "DSparseTensor.full_tensor() with RowPartitioned "
                "placement requires torch.distributed to be initialised."
            )
        return self._all_gather_global_matrix()

    def redistribute(
        self,
        placement: SparsePlacement,
        *,
        mesh: Any = None,
    ) -> "DSparseTensor":
        """Convert this :class:`DSparseTensor` to a new placement.

        Mirrors :meth:`DTensor.redistribute`. Only two transitions are
        currently meaningful for sparse matrices: ``RowPartitioned ->
        Replicated`` (Allgather every rank's COO triples and rebuild)
        and ``Replicated -> RowPartitioned`` (drop the rows this rank
        doesn't own).
        """
        if self._spec is None:
            raise RuntimeError(
                "redistribute() requires a DTensor-mirror DSparseTensor;"
                " build one via DSparseTensor.from_local(...) first."
            )
        target_mesh = mesh or self._spec.mesh

        if type(self._spec.placement) is type(placement):
            # Same placement: no-op besides optionally re-stamping mesh.
            if target_mesh is self._spec.mesh:
                return self
            return DSparseTensor.from_local(
                self._local_matrix, target_mesh,
                placement=placement,
                global_shape=self._spec.global_shape,
            )

        if isinstance(self._spec.placement, SparseShard) \
                and isinstance(placement, Replicated):
            full = self._all_gather_global_matrix()
            # ``DSparseMatrix.from_global`` is a static helper that
            # builds a single-partition local chunk -- pass ``num_part=1``
            # so the whole matrix is owned by this rank.
            full_local = DSparseMatrix._from_global_impl(
                SparseTensor(full.val, full.row, full.col,
                             self._spec.global_shape),
                num_partitions=1,
                partition_id=0,
                device=self._local_matrix.device,
                verbose=False,
            )
            return DSparseTensor.from_local(
                full_local, target_mesh,
                placement=Replicated(),
                global_shape=self._spec.global_shape,
            )

        raise NotImplementedError(
            f"redistribute {type(self._spec.placement).__name__} -> "
            f"{type(placement).__name__} not yet supported"
        )

    def _all_gather_global_matrix(self) -> "SparseTensor":
        """Allgather the per-rank COO triples and rebuild a global
        :class:`SparseTensor`. Used by :meth:`full_tensor` and
        :meth:`redistribute`."""
        from .sparse_tensor import SparseTensor

        local = self._local_matrix
        device = local.device
        world_size = self._spec.mesh.size()

        # Local row/col indices live in the LOCAL coordinate system; we
        # need them in GLOBAL coordinates before stitching. Use the
        # partition's ``local_nodes`` map (local idx -> global idx).
        l2g = local.partition.local_nodes.to(device)
        # Only owned-row entries -- duplicates across ranks would
        # otherwise show up after the all_gather.
        owned_mask = local.local_row < local.num_owned
        rows_g = l2g[local.local_row[owned_mask]].contiguous()
        cols_g = l2g[local.local_col[owned_mask]].contiguous()
        vals   = local.local_values[owned_mask].contiguous()

        # Allgather variable-length COO triples. Same shape pattern as
        # gather_global: exchange sizes, allocate per-rank buffers,
        # all_gather each field.
        owned_nnz = torch.tensor([int(vals.numel())],
                                 dtype=torch.int64, device=device)
        sizes = [torch.zeros(1, dtype=torch.int64, device=device)
                 for _ in range(world_size)]
        dist.all_gather(sizes, owned_nnz)
        ns = [int(s.item()) for s in sizes]

        rows_list = [torch.zeros(n, dtype=torch.int64, device=device)
                     for n in ns]
        cols_list = [torch.zeros(n, dtype=torch.int64, device=device)
                     for n in ns]
        vals_list = [torch.zeros(n, dtype=vals.dtype, device=device)
                     for n in ns]
        dist.all_gather(rows_list, rows_g)
        dist.all_gather(cols_list, cols_g)
        dist.all_gather(vals_list, vals)

        return SparseTensor(
            torch.cat(vals_list),
            torch.cat(rows_list),
            torch.cat(cols_list),
            self._spec.global_shape,
        )

    @classmethod
    def from_sparse_tensor(
        cls,
        sparse_tensor: "SparseTensor",
        num_partitions: int,
        coords: Optional[torch.Tensor] = None,
        partition_method: str = 'auto',
        device: Optional[Union[str, torch.device]] = None,
        verbose: bool = True
    ) -> "DSparseTensor":
        """
        Create DSparseTensor from a SparseTensor.
        
        Parameters
        ----------
        sparse_tensor : SparseTensor
            Input sparse tensor (must be 2D, not batched)
        num_partitions : int
            Number of partitions
        coords : torch.Tensor, optional
            Node coordinates for geometric partitioning
        partition_method : str
            Partitioning method
        device : str or torch.device, optional
            Target device (defaults to sparse_tensor's device)
        verbose : bool
            Whether to print partition info
            
        Returns
        -------
        DSparseTensor
            Distributed sparse tensor
        """
        # Avoid circular import
        from .sparse_tensor import SparseTensor
        
        if sparse_tensor.is_batched:
            raise ValueError("DSparseTensor does not support batched SparseTensor. "
                           "Use a 2D SparseTensor.")
        
        if device is None:
            device = sparse_tensor.device
        
        # Use sparse_shape for the matrix dimensions
        sparse_shape = sparse_tensor.sparse_shape
        
        return cls(
            sparse_tensor.values,
            sparse_tensor.row_indices,
            sparse_tensor.col_indices,
            sparse_shape,
            num_partitions=num_partitions,
            coords=coords,
            partition_method=partition_method,
            device=device,
            verbose=verbose
        )
    
    @classmethod
    def from_torch_sparse(
        cls,
        A: torch.Tensor,
        num_partitions: int,
        **kwargs
    ) -> "DSparseTensor":
        """Create DSparseTensor from PyTorch sparse tensor."""
        if A.layout == torch.sparse_csr:
            A = A.to_sparse_coo()
        
        indices = A._indices()
        values = A._values()
        
        return cls(
            values, indices[0], indices[1], tuple(A.shape),
            num_partitions=num_partitions, **kwargs
        )
    
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
        DSparseMatrix
            Local partition matrix for this rank
            
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
            from torch.distributed.device_mesh import init_device_mesh
            mesh = init_device_mesh(str(device.type), (world_size,))

        return cls.from_sparse_local(
            local_st, mesh, partition, global_shape=tuple(shape),
        )
    
    @classmethod
    def from_device_mesh(
        cls,
        values: torch.Tensor,
        row_indices: torch.Tensor,
        col_indices: torch.Tensor,
        shape: Tuple[int, int],
        device_mesh: "DeviceMesh",
        coords: Optional[torch.Tensor] = None,
        partition_method: str = 'simple',
        placement: str = 'shard_rows',
        verbose: bool = False
    ) -> "DSparseMatrix":
        """
        Create local partition using PyTorch DeviceMesh.
        
        This is the recommended method for distributed training with PyTorch's
        DTensor ecosystem. Each rank receives only its local partition.
        
        Parameters
        ----------
        values : torch.Tensor
            Global non-zero values [nnz] (same on all ranks)
        row_indices : torch.Tensor
            Global row indices [nnz]
        col_indices : torch.Tensor
            Global column indices [nnz]
        shape : Tuple[int, int]
            Global matrix shape (M, N)
        device_mesh : DeviceMesh
            PyTorch DeviceMesh specifying device topology
        coords : torch.Tensor, optional
            Node coordinates for geometric partitioning
        partition_method : str
            Partitioning method: 'metis', 'rcb', 'simple'
            Default is 'simple' for determinism in distributed setting
        placement : str
            How to distribute: 'shard_rows', 'shard_cols', 'replicate'
        verbose : bool
            Whether to print partition info
            
        Returns
        -------
        DSparseMatrix
            Local partition for this rank
            
        Example
        -------
        >>> from torch.distributed.device_mesh import init_device_mesh
        >>> from torch_sla import DSparseTensor
        >>> 
        >>> # Initialize 4-GPU device mesh
        >>> mesh = init_device_mesh("cuda", (4,), mesh_dim_names=("dp",))
        >>> 
        >>> # Create distributed sparse tensor (each rank gets its partition)
        >>> local_matrix = DSparseTensor.from_device_mesh(
        ...     val, row, col, shape,
        ...     device_mesh=mesh,
        ...     partition_method='simple'
        ... )
        >>> 
        >>> # Local operations
        >>> y_local = local_matrix.matvec(x_local)
        >>> x_local = local_matrix.solve(b_local)
        """
        try:
            from torch.distributed.device_mesh import DeviceMesh
        except ImportError:
            raise ImportError("DeviceMesh requires PyTorch 2.0+. "
                            "Use from_global_distributed() instead.")
        
        if not DIST_AVAILABLE or not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized. "
                             "Call dist.init_process_group() first.")
        
        # Get rank info from device mesh
        rank = device_mesh.get_local_rank()
        world_size = device_mesh.size()
        device_type = device_mesh.device_type
        
        # Determine target device
        if device_type == "cuda":
            device = torch.device(f"cuda:{rank}")
        else:
            device = torch.device(device_type)
        
        # Use the distributed-safe factory method
        return cls.from_global_distributed(
            values, row_indices, col_indices, shape,
            rank=rank, world_size=world_size,
            coords=coords,
            partition_method=partition_method,
            device=device,
            verbose=verbose
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
        return self._values.dtype
    
    @property
    def nnz(self) -> int:
        """Total number of non-zeros."""
        return self._values.size(0)
    
    @property
    def partition_ids(self) -> torch.Tensor:
        """Partition assignment for each node."""
        return self._partition_ids
    
    @property
    def is_cuda(self) -> bool:
        """Check if matrix is on CUDA."""
        return self._device.type == 'cuda'
    
    # =========================================================================
    # Indexing and Iteration
    # =========================================================================
    
    def __len__(self) -> int:
        """Number of partitions."""
        return self._num_partitions
    
    def __getitem__(self, idx: int) -> DSparseMatrix:
        """Get a specific partition."""
        if idx < 0:
            idx = self._num_partitions + idx
        if idx < 0 or idx >= self._num_partitions:
            raise IndexError(f"Partition index {idx} out of range [0, {self._num_partitions})")
        return self._partitions[idx]
    
    def __iter__(self):
        """Iterate over partitions."""
        return iter(self._partitions)
    
    # =========================================================================
    # Device Management
    # =========================================================================
    
    def to(self, device: Union[str, torch.device]) -> "DSparseTensor":
        """
        Move all partitions to a different device.
        
        Parameters
        ----------
        device : str or torch.device
            Target device
            
        Returns
        -------
        DSparseTensor
            New distributed tensor on target device
        """
        if isinstance(device, str):
            device = torch.device(device)
        
        new_tensor = DSparseTensor.__new__(DSparseTensor)
        new_tensor._values = self._values.to(device)
        new_tensor._row_indices = self._row_indices.to(device)
        new_tensor._col_indices = self._col_indices.to(device)
        new_tensor._shape = self._shape
        new_tensor._num_partitions = self._num_partitions
        new_tensor._coords = self._coords
        new_tensor._partition_method = self._partition_method
        new_tensor._verbose = False  # Don't print again
        new_tensor._device = device
        new_tensor._partition_ids = self._partition_ids
        
        # Move partitions
        new_tensor._partitions = [p.to(device) for p in self._partitions]
        
        # Update references
        for mat in new_tensor._partitions:
            mat._all_partitions = [m.partition for m in new_tensor._partitions]
        
        return new_tensor
    
    def cuda(self, device: Optional[int] = None) -> "DSparseTensor":
        """Move to CUDA device."""
        if device is not None:
            return self.to(f'cuda:{device}')
        return self.to('cuda')
    
    def cpu(self) -> "DSparseTensor":
        """Move to CPU."""
        return self.to('cpu')
    
    # =========================================================================
    # Distributed Operations
    # =========================================================================
    
    def halo_exchange_local(self, x_list: List[torch.Tensor]) -> None:
        """
        Local halo exchange for single-process simulation.
        
        Exchanges halo values between all partitions locally.
        Useful for testing without actual distributed setup.
        
        Parameters
        ----------
        x_list : List[torch.Tensor]
            List of local vectors, one per partition. Each vector is
            modified in-place to update halo values.
        """
        if len(x_list) != self._num_partitions:
            raise ValueError(f"Expected {self._num_partitions} vectors, got {len(x_list)}")
        
        for part_id in range(self._num_partitions):
            partition = self._partitions[part_id].partition
            x = x_list[part_id]
            
            halo_offset = len(partition.owned_nodes)
            
            for halo_idx, global_node in enumerate(partition.halo_nodes.tolist()):
                local_halo_idx = halo_offset + halo_idx
                
                for neighbor_id in partition.neighbor_partitions:
                    neighbor_partition = self._partitions[neighbor_id].partition
                    neighbor_g2l = neighbor_partition.global_to_local
                    
                    if global_node < len(neighbor_g2l):
                        local_idx_in_neighbor = neighbor_g2l[global_node].item()
                        if local_idx_in_neighbor >= 0 and local_idx_in_neighbor < len(neighbor_partition.owned_nodes):
                            x[local_halo_idx] = x_list[neighbor_id][local_idx_in_neighbor]
                            break
    
    def matvec_all(
        self,
        x_list: List[torch.Tensor],
        exchange_halo: bool = True
    ) -> List[torch.Tensor]:
        """
        Matrix-vector multiply on all partitions.
        
        Performs y = A @ x for each partition, with optional halo exchange.
        
        Parameters
        ----------
        x_list : List[torch.Tensor]
            List of local vectors, one per partition. Each vector should have
            size = num_owned + num_halo for that partition.
        exchange_halo : bool
            Whether to perform halo exchange before multiplication.
            Default True.
            
        Returns
        -------
        List[torch.Tensor]
            List of result vectors, one per partition. Each result has
            size = num_owned (only owned nodes have valid results).
            
        Example
        -------
        >>> D = SparseTensor(val, row, col, shape).partition(4)
        >>> x_local = D.scatter_local(x_global)
        >>> y_local = D.matvec_all(x_local)
        >>> y_global = D.gather_global(y_local)
        """
        return [self._partitions[i].matvec(x_list[i], exchange_halo=exchange_halo)
                for i in range(self._num_partitions)]
    
    def solve_all(
        self,
        b_list: List[torch.Tensor],
        **kwargs
    ) -> List[torch.Tensor]:
        """
        Solve on all partitions (subdomain solves).
        
        NOTE: This performs LOCAL subdomain solves, NOT a global distributed solve.
        Each partition solves its own local system independently.
        For a true distributed solve, use `solve_distributed()`.
        
        Parameters
        ----------
        b_list : List[torch.Tensor]
            List of local RHS vectors, one per partition
        **kwargs
            Additional arguments passed to each partition's solve method
            
        Returns
        -------
        List[torch.Tensor]
            List of solution vectors, one per partition
        """
        return [self._partitions[i].solve(b_list[i], **kwargs) 
                for i in range(self._num_partitions)]
    
    def solve_distributed(
        self,
        b_global: Union[torch.Tensor, "DTensor"],
        method: str = 'cg',
        atol: float = 1e-10,
        maxiter: int = 1000,
        verbose: bool = False
    ) -> Union[torch.Tensor, "DTensor"]:
        """
        Distributed solve: find x such that A @ x = b using all partitions.
        
        This performs a TRUE distributed solve where all partitions collaborate
        to solve the global system. Uses distributed CG with global reductions.
        
        Parameters
        ----------
        b_global : torch.Tensor or DTensor
            Global RHS vector [N].
            - If torch.Tensor: treated as global vector
            - If DTensor: automatically handles distributed input/output
        method : str
            Solver method: 'cg' (Conjugate Gradient)
        atol : float
            Absolute tolerance for convergence
        maxiter : int
            Maximum iterations
        verbose : bool
            Print convergence info
            
        Returns
        -------
        torch.Tensor or DTensor
            Global solution vector [N].
            Returns DTensor if input is DTensor, otherwise torch.Tensor.
            
        Example
        -------
        >>> D = A.partition(num_partitions=4)
        >>> x = D.solve_distributed(b)  # Distributed CG solve
        >>> residual = torch.norm(A @ x - b)
        
        >>> # With DTensor input
        >>> from torch.distributed.tensor import DTensor, Replicate
        >>> b_dt = DTensor.from_local(b_local, mesh, [Replicate()])
        >>> x_dt = D.solve_distributed(b_dt)  # Returns DTensor
        """
        # Check for DTensor input
        if _is_dtensor(b_global):
            return self._solve_distributed_dtensor(b_global, method, atol, maxiter, verbose)
        
        N = self._shape[0]
        dtype = b_global.dtype
        device = self._device
        
        # Initialize x = 0
        x_global = torch.zeros(N, dtype=dtype, device=device)
        
        # Scatter b to local
        b_local = self.scatter_local(b_global)
        
        # Distributed CG
        if method == 'cg':
            x_global = self._distributed_cg(x_global, b_global, atol, maxiter, verbose)
        else:
            raise ValueError(f"Unknown method: {method}. Supported: 'cg'")
        
        return x_global
    
    def _solve_distributed_dtensor(
        self,
        b_dtensor: "DTensor",
        method: str,
        atol: float,
        maxiter: int,
        verbose: bool
    ) -> "DTensor":
        """
        Distributed solve with DTensor input.
        
        Handles DTensor layout conversion and result wrapping.
        
        Parameters
        ----------
        b_dtensor : DTensor
            Right-hand side as DTensor
        method : str
            Solver method
        atol : float
            Absolute tolerance
        maxiter : int
            Maximum iterations
        verbose : bool
            Print convergence info
            
        Returns
        -------
        DTensor
            Solution as DTensor with same placement as input
        """
        if not DTENSOR_AVAILABLE:
            raise RuntimeError("DTensor support requires PyTorch 2.0+")
        
        # Get DTensor metadata
        device_mesh = b_dtensor.device_mesh
        placements = b_dtensor.placements
        original_placements = tuple(placements)
        
        # Check if input is replicated
        is_replicated = all(isinstance(p, Replicate) for p in placements)
        
        if is_replicated:
            # Input is replicated - extract and solve
            b_local = b_dtensor.to_local()
            x_local = self._solve_distributed_tensor(b_local, method, atol, maxiter, verbose)
            # Wrap result as replicated DTensor
            return DTensor.from_local(x_local, device_mesh, [Replicate()])
        
        # Input is sharded - redistribute to replicated for solve
        replicate_placements = [Replicate() for _ in placements]
        b_replicated = b_dtensor.redistribute(device_mesh, replicate_placements)
        b_full = b_replicated.to_local()
        
        # Solve with full vector
        x_full = self._solve_distributed_tensor(b_full, method, atol, maxiter, verbose)
        
        # Wrap as replicated DTensor
        x_replicated = DTensor.from_local(x_full, device_mesh, [Replicate()])
        
        # Redistribute back to original placement if it was sharded
        if not is_replicated:
            output_placements = []
            for p in original_placements:
                if isinstance(p, Shard):
                    output_placements.append(Shard(p.dim))
                else:
                    output_placements.append(Replicate())
            
            return x_replicated.redistribute(device_mesh, output_placements)
        
        return x_replicated
    
    def _solve_distributed_tensor(
        self,
        b_global: torch.Tensor,
        method: str,
        atol: float,
        maxiter: int,
        verbose: bool
    ) -> torch.Tensor:
        """
        Internal solve implementation for torch.Tensor input.
        
        Separated from solve_distributed to allow DTensor wrapper to call it.
        """
        N = self._shape[0]
        dtype = b_global.dtype
        device = self._device
        
        # Initialize x = 0
        x_global = torch.zeros(N, dtype=dtype, device=device)
        
        # Scatter b to local
        b_local = self.scatter_local(b_global)
        
        # Distributed CG
        if method == 'cg':
            x_global = self._distributed_cg(x_global, b_global, atol, maxiter, verbose)
        else:
            raise ValueError(f"Unknown method: {method}. Supported: 'cg'")
        
        return x_global
    
    def _distributed_cg(
        self,
        x: torch.Tensor,
        b: torch.Tensor,
        atol: float,
        maxiter: int,
        verbose: bool
    ) -> torch.Tensor:
        """
        Distributed Conjugate Gradient.
        
        All partitions work together, with global reductions for inner products.
        """
        N = self._shape[0]
        dtype = b.dtype
        device = self._device
        
        # r = b - A @ x
        Ax = self @ x  # Uses __matmul__ which does scatter -> matvec_all -> gather
        r = b - Ax
        
        # p = r
        p = r.clone()
        
        # rs_old = r^T @ r (global)
        rs_old = torch.dot(r, r)
        
        for i in range(maxiter):
            # Ap = A @ p
            Ap = self @ p
            
            # pAp = p^T @ A @ p (global)
            pAp = torch.dot(p, Ap)
            
            if pAp.abs() < 1e-30:
                if verbose:
                    print(f"  Distributed CG: pAp too small at iter {i}")
                break
            
            # alpha = rs_old / pAp
            alpha = rs_old / pAp
            
            # x = x + alpha * p
            x = x + alpha * p
            
            # r = r - alpha * Ap
            r = r - alpha * Ap
            
            # rs_new = r^T @ r (global)
            rs_new = torch.dot(r, r)
            
            residual = rs_new.sqrt()
            
            if verbose and i % 100 == 0:
                print(f"  Distributed CG iter {i}: residual = {residual:.2e}")
            
            if residual < atol:
                if verbose:
                    print(f"  Distributed CG converged at iter {i}, residual = {residual:.2e}")
                break
            
            if rs_old.abs() < 1e-30:
                break
            
            # beta = rs_new / rs_old
            beta = rs_new / rs_old
            
            # p = r + beta * p
            p = r + beta * p
            
            rs_old = rs_new
        
        return x
    
    def gather_global(self, x_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Gather local vectors to global vector.
        
        Parameters
        ----------
        x_list : List[torch.Tensor]
            List of local vectors, one per partition
            
        Returns
        -------
        torch.Tensor
            Global vector
        """
        x_global = torch.zeros(self._shape[0], dtype=x_list[0].dtype, device=self._device)
        
        for i in range(self._num_partitions):
            partition = self._partitions[i].partition
            owned_nodes = partition.owned_nodes
            num_owned = len(owned_nodes)
            x_global[owned_nodes] = x_list[i][:num_owned].to(self._device)
        
        return x_global
    
    def scatter_local(self, x_global: torch.Tensor) -> List[torch.Tensor]:
        """
        Scatter global vector to local vectors.
        
        Parameters
        ----------
        x_global : torch.Tensor
            Global vector
            
        Returns
        -------
        List[torch.Tensor]
            List of local vectors (with halo values filled)
        """
        x_list = []
        
        for i in range(self._num_partitions):
            partition = self._partitions[i].partition
            local_nodes = partition.local_nodes
            x_local = x_global[local_nodes].to(self._partitions[i].device)
            x_list.append(x_local)
        
        return x_list
    
    def to_sparse_tensor(self) -> "SparseTensor":
        """
        Gather all partitions into a single SparseTensor.
        
        This creates a global SparseTensor from the distributed data.
        Useful for verification, debugging, or when you need to perform
        operations that require the full matrix.
        
        Returns
        -------
        SparseTensor
            Global sparse tensor containing all data
            
        Example
        -------
        >>> D = DSparseTensor(val, row, col, shape, num_partitions=4)
        >>> A = D.to_sparse_tensor()  # Gather to global SparseTensor
        >>> x = A.solve(b)  # Solve on the full matrix
        """
        from .sparse_tensor import SparseTensor
        
        # Return the original global data as SparseTensor
        return SparseTensor(
            self._values.to(self._device),
            self._row_indices.to(self._device),
            self._col_indices.to(self._device),
            self._shape
        )
    
    # Alias for convenience
    gather = to_sparse_tensor
    
    # =========================================================================
    # DTensor Utilities
    # =========================================================================
    
    def scatter_to_dtensor(
        self,
        x_global: torch.Tensor,
        device_mesh: "DeviceMesh",
        shard_dim: int = 0
    ) -> "DTensor":
        """
        Convert a global tensor to a sharded DTensor aligned with matrix partitioning.
        
        This creates a DTensor where each rank holds the portion of the vector
        corresponding to its owned nodes in the matrix partitioning.
        
        Parameters
        ----------
        x_global : torch.Tensor
            Global vector of shape [N]
        device_mesh : DeviceMesh
            PyTorch DeviceMesh for distribution
        shard_dim : int
            Dimension to shard (default 0 for vectors)
            
        Returns
        -------
        DTensor
            Sharded DTensor with local data for this rank
            
        Example
        -------
        >>> mesh = init_device_mesh("cuda", (4,))
        >>> x_global = torch.randn(N)
        >>> x_dt = D.scatter_to_dtensor(x_global, mesh)
        """
        if not DTENSOR_AVAILABLE:
            raise RuntimeError("DTensor support requires PyTorch 2.0+")
        
        # Create sharded DTensor
        # Each rank gets the portion corresponding to its partition
        placements = [Shard(shard_dim)]
        return DTensor.from_local(
            x_global,  # Will be redistributed by DTensor
            device_mesh,
            placements
        )
    
    def gather_from_dtensor(
        self,
        x_dtensor: "DTensor"
    ) -> torch.Tensor:
        """
        Convert a DTensor to a global tensor.
        
        Parameters
        ----------
        x_dtensor : DTensor
            Distributed tensor
            
        Returns
        -------
        torch.Tensor
            Full global tensor
            
        Example
        -------
        >>> x_global = D.gather_from_dtensor(x_dt)
        """
        if not DTENSOR_AVAILABLE:
            raise RuntimeError("DTensor support requires PyTorch 2.0+")
        
        return x_dtensor.full_tensor()
    
    def to_dtensor(
        self,
        x: torch.Tensor,
        device_mesh: "DeviceMesh",
        replicate: bool = True
    ) -> "DTensor":
        """
        Convert a tensor to DTensor with specified placement.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        device_mesh : DeviceMesh
            PyTorch DeviceMesh
        replicate : bool
            If True, create a replicated DTensor (same data on all ranks).
            If False, create a sharded DTensor (data is split).
            
        Returns
        -------
        DTensor
            Resulting DTensor
            
        Example
        -------
        >>> mesh = init_device_mesh("cuda", (4,))
        >>> x_dt = D.to_dtensor(x, mesh, replicate=True)
        """
        if not DTENSOR_AVAILABLE:
            raise RuntimeError("DTensor support requires PyTorch 2.0+")
        
        if replicate:
            placements = [Replicate()]
        else:
            placements = [Shard(0)]
        
        return DTensor.from_local(x, device_mesh, placements)
    
    @property
    def supports_dtensor(self) -> bool:
        """Check if DTensor operations are available."""
        return DTENSOR_AVAILABLE
    
    # =========================================================================
    # Distributed Algorithms (True Distributed, No Gather)
    # =========================================================================
    
    def _global_matvec_with_grad(self, x: torch.Tensor) -> torch.Tensor:
        """
        Global matrix-vector multiplication that preserves gradients.
        
        Uses the original COO data to maintain gradient flow.
        For true distributed MPI execution, use _distributed_matvec instead.
        
        This method is used for gradient-enabled operations like eigsh, solve.
        """
        # Use original global COO data for gradient support
        # y[i] = sum_j A[i,j] * x[j]
        # y = scatter_add(values * x[col], row)
        y = torch.zeros(self._shape[0], dtype=x.dtype, device=x.device)
        vals = self._values.to(x.device)
        rows = self._row_indices.to(x.device)
        cols = self._col_indices.to(x.device)
        
        # y[row] += values * x[col]
        contributions = vals * x[cols]
        y = y.scatter_add(0, rows, contributions)
        return y
    
    def _distributed_matvec(self, x: torch.Tensor) -> torch.Tensor:
        """
        Distributed matrix-vector multiplication with gradient support.
        
        For single-node simulation with gradient support, uses _global_matvec_with_grad.
        For true distributed MPI execution, uses scatter -> local matvec -> gather.
        """
        # Check if we need gradients
        if self._values.requires_grad or x.requires_grad:
            # Use global matvec that preserves gradients
            return self._global_matvec_with_grad(x)
        
        # Otherwise use true distributed pattern
        x_local = self.scatter_local(x)
        y_local = self.matvec_all(x_local)
        return self.gather_global(y_local)
    
    def _distributed_lobpcg(
        self,
        k: int,
        largest: bool = True,
        maxiter: int = 1000,
        tol: float = 1e-8
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Distributed LOBPCG eigenvalue solver.
        
        Uses distributed matvec with global QR and Rayleigh-Ritz.
        No data gather required - only needs global reductions.
        """
        N = self._shape[0]
        dtype = self._values.dtype
        device = self._device
        
        # Initialize random subspace (global vectors)
        m = min(2 * k, N)
        X = torch.randn(N, m, dtype=dtype, device=device)
        
        # Global QR decomposition
        X, _ = torch.linalg.qr(X)
        
        eigenvalues_prev = None
        
        for iteration in range(maxiter):
            # Distributed matvec: AX = D @ X (column by column or batched)
            AX = torch.zeros_like(X)
            for j in range(X.shape[1]):
                AX[:, j] = self._distributed_matvec(X[:, j])
            
            # Rayleigh-Ritz: project onto subspace
            # H = X^T @ AX (global reduction)
            H = X.T @ AX
            
            # Solve small eigenvalue problem
            eigenvalues, eigenvectors = torch.linalg.eigh(H)
            
            # Sort eigenvalues
            if largest:
                idx = eigenvalues.argsort(descending=True)
            else:
                idx = eigenvalues.argsort()
            
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # Update X = X @ V
            X = X @ eigenvectors
            
            # Check convergence
            if eigenvalues_prev is not None:
                diff = (eigenvalues[:k] - eigenvalues_prev[:k]).abs()
                if (diff < tol * eigenvalues[:k].abs().clamp(min=1e-10)).all():
                    break
            eigenvalues_prev = eigenvalues.clone()
            
            # Expand subspace with residual
            if iteration < maxiter - 1:
                # Compute residual: R = AX - X @ diag(eigenvalues)
                AX_new = torch.zeros_like(X)
                for j in range(X.shape[1]):
                    AX_new[:, j] = self._distributed_matvec(X[:, j])
                
                residual = AX_new - X * eigenvalues.unsqueeze(0)
                
                # Orthogonalize and expand
                combined = torch.cat([X[:, :k], residual[:, :k]], dim=1)
                X, _ = torch.linalg.qr(combined)
                
                # Pad if needed
                if X.size(1) < m:
                    extra = torch.randn(N, m - X.size(1), dtype=dtype, device=device)
                    X = torch.cat([X, extra], dim=1)
                    X, _ = torch.linalg.qr(X)
        
        return eigenvalues[:k], X[:, :k]
    
    def eigsh(
        self,
        k: int = 6,
        which: str = "LM",
        sigma: Optional[float] = None,
        return_eigenvectors: bool = True,
        maxiter: int = 1000,
        tol: float = 1e-8
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute k eigenvalues for symmetric matrices using distributed LOBPCG.
        
        This is a TRUE distributed algorithm - no data gather required.
        Uses distributed matvec with global QR decomposition.
        
        Parameters
        ----------
        k : int, optional
            Number of eigenvalues to compute. Default: 6.
        which : {"LM", "SM", "LA", "SA"}, optional
            Which eigenvalues to find:
            - "LM"/"LA": Largest (default)
            - "SM"/"SA": Smallest
        sigma : float, optional
            Find eigenvalues near sigma (not yet supported).
        return_eigenvectors : bool, optional
            Whether to return eigenvectors. Default: True.
        maxiter : int, optional
            Maximum LOBPCG iterations. Default: 1000.
        tol : float, optional
            Convergence tolerance. Default: 1e-8.
            
        Returns
        -------
        eigenvalues : torch.Tensor
            Shape [k].
        eigenvectors : torch.Tensor or None
            Shape [N, k] if return_eigenvectors is True.
        
        Notes
        -----
        **Distributed Algorithm:**
        
        - Uses distributed LOBPCG (Locally Optimal Block PCG)
        - Only requires distributed matvec + global reductions
        - Memory: O(N * k) per node for eigenvectors
        - Communication: O(k^2) per iteration for Rayleigh-Ritz
        
        **Gradient Support:**
        
        - Gradients flow through the distributed matvec operations
        - O(iterations) graph nodes (not O(1) like adjoint)
        """
        if sigma is not None:
            warnings.warn("sigma (shift-invert) not yet supported for distributed eigsh. Ignoring.")
        
        largest = which in ('LM', 'LA')
        eigenvalues, eigenvectors = self._distributed_lobpcg(k, largest=largest, maxiter=maxiter, tol=tol)
        
        if return_eigenvectors:
            return eigenvalues, eigenvectors
        return eigenvalues, None
    
    def eigs(
        self,
        k: int = 6,
        which: str = "LM",
        sigma: Optional[float] = None,
        return_eigenvectors: bool = True,
        maxiter: int = 1000,
        tol: float = 1e-8
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute k eigenvalues using distributed LOBPCG.
        
        For symmetric matrices, equivalent to eigsh().
        For non-symmetric, currently falls back to eigsh() (symmetric assumption).
        
        Parameters
        ----------
        k : int, optional
            Number of eigenvalues to compute. Default: 6.
        which : str, optional
            Which eigenvalues to find.
        sigma : float, optional
            Find eigenvalues near sigma.
        return_eigenvectors : bool, optional
            Whether to return eigenvectors. Default: True.
        maxiter : int, optional
            Maximum iterations. Default: 1000.
        tol : float, optional
            Convergence tolerance. Default: 1e-8.
            
        Returns
        -------
        eigenvalues : torch.Tensor
            Shape [k].
        eigenvectors : torch.Tensor or None
            Shape [N, k] if return_eigenvectors is True.
        """
        # For now, use eigsh (assumes symmetric)
        # TODO: Implement Arnoldi for non-symmetric
        return self.eigsh(k=k, which=which, sigma=sigma, 
                         return_eigenvectors=return_eigenvectors,
                         maxiter=maxiter, tol=tol)
    
    def svd(
        self, 
        k: int = 6,
        maxiter: int = 1000,
        tol: float = 1e-8
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute truncated SVD using distributed power iteration.
        
        Uses A^T @ A for eigenvalues, then recovers U from A @ V.
        
        Parameters
        ----------
        k : int, optional
            Number of singular values to compute. Default: 6.
        maxiter : int, optional
            Maximum iterations. Default: 1000.
        tol : float, optional
            Convergence tolerance. Default: 1e-8.
            
        Returns
        -------
        U : torch.Tensor
            Left singular vectors. Shape [M, k].
        S : torch.Tensor
            Singular values. Shape [k].
        Vt : torch.Tensor
            Right singular vectors. Shape [k, N].
        
        Notes
        -----
        **Distributed Algorithm:**
        
        - Computes eigenvalues of A^T @ A using distributed LOBPCG
        - No data gather required
        """
        M, N = self._shape
        dtype = self._values.dtype
        device = self._device
        
        # For SVD, we need A^T @ A which requires transpose
        # Create A^T as a DSparseTensor
        A_T = self.T()
        
        # Power iteration for A^T @ A
        # Initialize random vectors
        V = torch.randn(N, k, dtype=dtype, device=device)
        V, _ = torch.linalg.qr(V)
        
        for iteration in range(maxiter):
            # AV = A @ V
            AV = torch.zeros(M, k, dtype=dtype, device=device)
            for j in range(k):
                AV[:, j] = self._distributed_matvec(V[:, j])
            
            # AtAV = A^T @ (A @ V)
            AtAV = torch.zeros(N, k, dtype=dtype, device=device)
            for j in range(k):
                AtAV[:, j] = A_T._distributed_matvec(AV[:, j])
            
            # QR decomposition
            V_new, R = torch.linalg.qr(AtAV)
            
            # Check convergence
            diff = (V_new - V).norm()
            V = V_new
            
            if diff < tol:
                break
        
        # Compute singular values and U
        # AV = A @ V, then normalize to get U
        AV = torch.zeros(M, k, dtype=dtype, device=device)
        for j in range(k):
            AV[:, j] = self._distributed_matvec(V[:, j])
        
        # S = ||AV[:, j]||
        S = AV.norm(dim=0)
        
        # U = AV / S
        U = AV / S.unsqueeze(0).clamp(min=1e-10)
        
        return U, S, V.T
    
    def norm(self, ord: Literal['fro', 1, 2] = 'fro') -> torch.Tensor:
        """
        Compute matrix norm (distributed).
        
        For Frobenius norm, computed locally and aggregated.
        For spectral norm, uses distributed SVD.
        
        Parameters
        ----------
        ord : {'fro', 1, 2}
            Type of norm:
            - 'fro': Frobenius norm (distributed sum)
            - 1: Maximum column sum
            - 2: Spectral norm (largest singular value via distributed SVD)
            
        Returns
        -------
        torch.Tensor
            Scalar tensor containing the norm value.
        """
        if ord == 'fro':
            # Frobenius norm: sqrt(sum(values^2))
            # This is truly distributed - each partition has its own values
            return torch.sqrt((self._values ** 2).sum())
        elif ord == 2:
            # Spectral norm: largest singular value
            _, S, _ = self.svd(k=1, maxiter=100)
            return S[0]
        elif ord == 1:
            # Maximum column sum - need to gather
            warnings.warn("1-norm requires data gather. Using to_sparse_tensor().")
            return self.to_sparse_tensor().norm(ord=1)
        else:
            raise ValueError(f"Unknown norm order: {ord}")
    
    def condition_number(self, ord: int = 2) -> torch.Tensor:
        """
        Estimate condition number using distributed SVD.
        
        Parameters
        ----------
        ord : int, optional
            Norm order. Default: 2 (spectral).
            
        Returns
        -------
        torch.Tensor
            Condition number estimate (σ_max / σ_min).
        """
        if ord == 2:
            # Need largest and smallest singular values
            # Compute k=6 singular values
            _, S, _ = self.svd(k=6, maxiter=200)
            return S[0] / S[-1].clamp(min=1e-10)
        else:
            warnings.warn(f"ord={ord} requires data gather. Using to_sparse_tensor().")
            return self.to_sparse_tensor().condition_number(ord=ord)
    
    def det(self) -> torch.Tensor:
        """
        Compute determinant of the distributed sparse matrix.
        
        WARNING: This operation requires gathering the full matrix to compute
        the determinant, as determinant is a global property that cannot be
        computed in a truly distributed manner without full matrix information.
        
        The determinant is computed by:
        1. Gathering all partitions into a global SparseTensor
        2. Computing the determinant using LU decomposition (CPU) or 
           torch.linalg.det (CUDA)
        
        Returns
        -------
        torch.Tensor
            Determinant value (scalar tensor).
            
        Raises
        ------
        ValueError
            If matrix is not square
            
        Notes
        -----
        - Only square matrices have determinants
        - This method gathers all data, so use with caution for large matrices
        - Supports gradient computation via autograd
        - For very large matrices, consider using log-determinant or other
          approximations instead
        
        Examples
        --------
        >>> import torch
        >>> from torch_sla import DSparseTensor
        >>> 
        >>> # Create distributed sparse matrix
        >>> val = torch.tensor([4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0])
        >>> row = torch.tensor([0, 0, 1, 1, 1, 2, 2])
        >>> col = torch.tensor([0, 1, 0, 1, 2, 1, 2])
        >>> D = DSparseTensor(val, row, col, (3, 3), num_partitions=2)
        >>> 
        >>> # Compute determinant (gathers to single node)
        >>> det = D.det()
        >>> print(det)
        >>>
        >>> # With gradient support
        >>> val = val.requires_grad_(True)
        >>> D = DSparseTensor(val, row, col, (3, 3), num_partitions=2)
        >>> det = D.det()
        >>> det.backward()
        >>> print(val.grad)  # Gradient w.r.t. matrix values
        """
        M, N = self._shape
        
        if M != N:
            raise ValueError(f"Matrix must be square for determinant, got shape ({M}, {N})")
        
        # Warn user about data gather
        warnings.warn(
            "det() requires gathering all partitions to compute the determinant. "
            "This is a global operation that cannot be computed in a truly distributed manner. "
            "For large matrices, this may be memory-intensive."
        )
        
        # Gather to global SparseTensor and compute determinant
        A_global = self.to_sparse_tensor()
        return A_global.det()
    
    def T(self) -> "DSparseTensor":
        """
        Transpose the distributed sparse tensor.
        
        Returns a new DSparseTensor with swapped row/column indices.
        
        Returns
        -------
        DSparseTensor
            Transposed matrix.
        """
        # Swap row and column indices
        return DSparseTensor(
            self._values,
            self._col_indices,  # swap
            self._row_indices,  # swap
            (self._shape[1], self._shape[0]),
            num_partitions=self._num_partitions,
            coords=self._coords,
            partition_method=self._partition_method,
            device=self._device,
            verbose=False
        )
    
    # =========================================================================
    # Methods that require data gather (with warnings)
    # =========================================================================
    
    def to_dense(self) -> torch.Tensor:
        """
        Convert to dense tensor.
        
        WARNING: This gathers all data to a single node.
        Only use for small matrices or debugging.
        
        Returns
        -------
        torch.Tensor
            Dense matrix of shape (M, N).
        """
        warnings.warn("to_dense() gathers all data to a single node. "
                     "Only use for debugging or small matrices.")
        return self.to_sparse_tensor().to_dense()
    
    def is_symmetric(self, atol: float = 1e-8, rtol: float = 1e-5) -> torch.Tensor:
        """
        Check if matrix is symmetric.
        
        Can be done distributedly by comparing values with transpose.
        
        Parameters
        ----------
        atol : float
            Absolute tolerance for symmetry check.
        rtol : float
            Relative tolerance for symmetry check.
            
        Returns
        -------
        torch.Tensor
            Boolean scalar tensor.
        """
        # This can be done without gather by checking local values
        # For now, use simple implementation
        return self.to_sparse_tensor().is_symmetric(atol=atol, rtol=rtol)
    
    def is_positive_definite(self) -> torch.Tensor:
        """
        Check if matrix is positive definite.
        
        Uses distributed eigenvalue computation.
        
        Returns
        -------
        torch.Tensor
            Boolean scalar tensor.
        """
        # Check smallest eigenvalue > 0
        eigenvalues, _ = self.eigsh(k=1, which='SA', return_eigenvectors=False, maxiter=200)
        return eigenvalues[0] > 0
    
    def lu(self):
        """
        Compute LU decomposition.
        
        WARNING: LU is inherently not distributed-friendly.
        This gathers data to a single node.
        
        For distributed solves, use solve_distributed() with iterative methods.
        
        Returns
        -------
        LUFactorization
            Factorization object with solve() method.
        """
        warnings.warn("LU decomposition is not distributed. "
                     "Use solve_distributed() for distributed solves.")
        return self.to_sparse_tensor().lu()
    
    def spy(self, **kwargs):
        """
        Visualize sparsity pattern.
        
        Gathers data for visualization.
        
        Parameters
        ----------
        **kwargs
            Arguments passed to SparseTensor.spy().
        """
        return self.to_sparse_tensor().spy(**kwargs)
    
    def nonlinear_solve(
        self,
        residual_fn,
        u0: torch.Tensor,
        *params,
        method: str = 'newton',
        tol: float = 1e-6,
        atol: float = 1e-10,
        max_iter: int = 50,
        line_search: bool = True,
        verbose: bool = False,
    ) -> torch.Tensor:
        """
        Solve nonlinear equation F(u, D, *params) = 0 using distributed Newton-Krylov.
        
        Uses Jacobian-free Newton-Krylov with distributed CG for linear solves.
        
        Parameters
        ----------
        residual_fn : callable
            Function F(u, D, *params) -> residual tensor.
            D is this DSparseTensor.
        u0 : torch.Tensor
            Initial guess (global vector).
        *params : torch.Tensor
            Additional parameters.
        method : str
            'newton': Newton-Krylov with distributed CG
            'picard': Fixed-point iteration
        tol : float
            Relative tolerance.
        atol : float
            Absolute tolerance.
        max_iter : int
            Maximum outer iterations.
        line_search : bool
            Use Armijo line search.
        verbose : bool
            Print convergence info.
            
        Returns
        -------
        torch.Tensor
            Solution u such that F(u, D, *params) ≈ 0.
        
        Notes
        -----
        **Distributed Algorithm:**
        
        - Uses Jacobian-free Newton-Krylov (JFNK)
        - Linear solves use distributed CG
        - Jacobian-vector products computed via finite differences
        """
        u = u0.clone()
        N = u.shape[0]
        dtype = u.dtype
        device = u.device
        
        for outer_iter in range(max_iter):
            # Compute residual
            F = residual_fn(u, self, *params)
            F_norm = F.norm()
            
            if verbose:
                print(f"  Newton iter {outer_iter}: ||F|| = {F_norm:.2e}")
            
            if F_norm < atol:
                if verbose:
                    print(f"  Converged (atol) at iteration {outer_iter}")
                break
            
            if outer_iter > 0 and F_norm < tol * F_norm_init:
                if verbose:
                    print(f"  Converged (rtol) at iteration {outer_iter}")
                break
            
            if outer_iter == 0:
                F_norm_init = F_norm
            
            if method == 'picard':
                # Simple fixed-point: u = u - F (assuming F = Au - b form)
                u = u - F
            else:
                # Newton-Krylov: solve J @ du = -F using CG with Jacobian-vector products
                # J @ v ≈ (F(u + eps*v) - F(u)) / eps
                eps = 1e-7 * max(u.norm(), 1.0)
                
                def matvec(v):
                    """Jacobian-vector product via finite differences."""
                    F_plus = residual_fn(u + eps * v, self, *params)
                    return (F_plus - F) / eps
                
                # Distributed CG for J @ du = -F
                du = torch.zeros_like(u)
                r = -F - matvec(du)  # r = -F - J @ 0 = -F
                p = r.clone()
                rs_old = torch.dot(r, r)
                
                for cg_iter in range(min(100, N)):
                    Jp = matvec(p)
                    pJp = torch.dot(p, Jp)
                    
                    if pJp.abs() < 1e-30:
                        break
                    
                    alpha = rs_old / pJp
                    du = du + alpha * p
                    r = r - alpha * Jp
                    rs_new = torch.dot(r, r)
                    
                    if rs_new.sqrt() < 1e-10:
                        break
                    
                    beta = rs_new / rs_old
                    p = r + beta * p
                    rs_old = rs_new
                
                # Line search
                if line_search:
                    alpha = 1.0
                    F_new_norm = residual_fn(u + alpha * du, self, *params).norm()
                    while F_new_norm > F_norm and alpha > 1e-8:
                        alpha *= 0.5
                        F_new_norm = residual_fn(u + alpha * du, self, *params).norm()
                    u = u + alpha * du
                else:
                    u = u + du
        
        return u
    
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
        # DTensor-mirror fast path: when this DSparseTensor carries a
        # real distributed spec (built via ``from_local`` on a multi-
        # rank job), the matvec stays entirely in the ``Shard(0)``
        # space -- halo exchange in, local matvec, wrap result as
        # DTensor with the same mesh + Shard(0) placement. No gather.
        if self._spec is not None and isinstance(self._spec.placement,
                                                  SparseShard):
            return self._matmul_spec(x)

        # Check for DTensor input (legacy "replicate-and-densify" path
        # for the single-process simulator).
        if _is_dtensor(x):
            return self._matmul_dtensor(x)

        return self._distributed_matvec(x)

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
        # Spec mode requires *some* local backing -- either the legacy
        # DSparseMatrix or the SparseTensor backing.
        assert (self._local_matrix is not None
                or self._local_tensor is not None), \
            "spec-mode requires _local_matrix or _local_tensor"

        placement = self._spec.placement
        if not isinstance(placement, SparseShard):
            raise RuntimeError(
                "_matmul_spec expects a SparseShard placement; got "
                f"{type(placement).__name__}")
        axis = placement.axis

        if axis == 0:
            return self._matmul_row_shard(x)
        if axis == 1:
            return self._matmul_col_shard(x)
        raise NotImplementedError(
            f"SparseShard(axis={axis}) matvec dispatch not implemented; "
            "axis must be 0 (row) or 1 (col) for SpMV.")

    def _matmul_row_shard(self, x: Any) -> Any:
        """Row-partitioned SpMV (``SparseShard(axis=0)``): halo exchange
        + local SpMV → ``DTensor[Shard(0)]``.

        Dispatches on which backing is set:

        * ``_local_tensor`` (plain :class:`SparseTensor`, SparseTensor-backed path)
          → ``_matmul_row_shard_via_sparse_tensor``
        * ``_local_matrix`` (legacy :class:`DSparseMatrix`) → original
          path.
        """
        if self._local_tensor is not None:
            return self._matmul_row_shard_via_sparse_tensor(x)

        local_matrix = self._local_matrix
        if _is_dtensor(x):
            x_local = x.to_local()
        else:
            x_local = x
        x_padded = self._pad_owned_to_local(x_local)
        y_full = local_matrix.matvec(x_padded, exchange_halo=True)
        y_owned = y_full[:local_matrix.num_owned]
        if _is_dtensor(x):
            from torch.distributed.tensor import DTensor as _DTensor, Shard
            return _DTensor.from_local(
                y_owned, self._spec.mesh, [Shard(0)])
        return y_owned

    def _matmul_row_shard_via_sparse_tensor(self, x: Any) -> Any:
        """SparseTensor-backed matvec: backed by ``_local_tensor`` (SparseTensor) +
        ``_spec.placement.partition``. No DSparseMatrix involvement.
        """
        partition = self._spec.placement.partition
        if partition is None:
            raise RuntimeError(
                "SparseTensor-backed matvec requires spec.placement.partition; "
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
            from torch.distributed.tensor import DTensor as _DTensor, Shard
            return _DTensor.from_local(
                y_owned, self._spec.mesh, [Shard(0)])
        return y_owned

    def _halo_exchange_via_spec(self,
                                  x: torch.Tensor,
                                  partition: "Partition") -> torch.Tensor:
        """Halo-exchange port that reads ``partition`` (from spec) and
        operates on ``x`` in-place. Functionally identical to
        :meth:`DSparseMatrix.halo_exchange` but lives on DSparseTensor
        so the dissolution doesn't require keeping a DSparseMatrix
        around.

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

        # Exchange (gloo-friendly: isend/irecv pairs).
        requests = []
        for nid in partition.neighbor_partitions:
            requests.append(dist.isend(send_bufs[nid], dst=int(nid)))
            requests.append(dist.irecv(recv_bufs[nid], src=int(nid)))
        for r in requests:
            r.wait()

        # Scatter recv buffers into x at recv_indices.
        for nid in partition.neighbor_partitions:
            recv_idx = partition.recv_indices[nid].to(device=device,
                                                       dtype=torch.int64)
            x[recv_idx] = recv_bufs[nid]
        return x

    def _matmul_col_shard(self, x: Any) -> Any:
        """Col-partitioned SpMV (``SparseShard(axis=1)``).

        **Status**: scaffolded but not wired to a real col-partition data
        layout. Today's ``DSparseMatrix`` (built by
        ``partition_for_rank`` / ``DSparseTensor.partition``) is
        **row-partitioned**; each rank owns a row slice plus a halo
        of columns it reads from. Column partitioning requires:

        1. A new ``SparseTensor.extract_column_partition(partition)``
           that slices columns into a local ``(M, num_owned_cols)``
           submatrix.
        2. A 2-D mesh sharding shim so the halo *and* the
           ``all_reduce(SUM)`` semantics line up.

        Both land in the column-partition follow-up -- step adds
        the column extraction; a follow-up wires it through this method.
        Until then this dispatch raises so callers don't silently get
        wrong answers.
        """
        raise NotImplementedError(
            "SparseShard(axis=1) col-partitioned matvec is scaffolded "
            "but requires the column-partition data path that ships "
            "with the column-partition follow-up. Today.s "
            "DSparseMatrix is row-partitioned -- use SparseShard(axis=0)."
        )

    def _pad_owned_to_local(self, x: torch.Tensor) -> torch.Tensor:
        """Convert a Shard(0)-sized ``[num_owned]`` tensor into a full
        ``[num_local]`` tensor with halo entries zero-padded (they get
        overwritten by the next ``halo_exchange``). Pass-through when
        ``x`` is already num_local-sized."""
        local_matrix = self._local_matrix
        nl = local_matrix.num_local
        no = local_matrix.num_owned
        if x.shape[0] == nl:
            return x
        if x.shape[0] == no:
            padded = torch.zeros(nl, dtype=x.dtype, device=x.device)
            padded[:no] = x
            return padded
        raise ValueError(
            f"x has shape[0]={x.shape[0]}, expected num_owned={no} or "
            f"num_local={nl}."
        )

    # ====================================================================== #
    # Shard(0)-space distributed CG.
    #
    # The classic ``_distributed_cg`` (single-process simulator path)
    # operates on global N-sized vectors. That's only correct in the
    # legacy single-process simulator -- on multiple ranks every CG
    # iteration would Allgather the whole vector. The Shard(0)
    # implementation below keeps every vector local (size ``num_owned``)
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
            from torch.distributed.tensor import DTensor as _DTensor, Shard
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
        """Owned-row count regardless of which local backing is set."""
        if self._local_tensor is not None:
            return int(self._spec.placement.partition.owned_nodes.numel())
        return int(self._local_matrix.num_owned)

    def _shard_matvec(self, x_owned: torch.Tensor) -> torch.Tensor:
        """Apply ``A`` to a Shard(0) vector.

        Routes through whichever backing is set on the spec'd
        DSparseTensor:

        * ``_local_tensor`` (plain :class:`SparseTensor`)
          → pad → ``_halo_exchange_via_spec`` →
          ``_local_tensor @ x`` → slice.
        * ``_local_matrix`` (legacy :class:`DSparseMatrix`)
          → ``local_matrix.matvec(..., exchange_halo=True)`` → slice.

        Used by every Shard(0)-space Krylov method (CG / BiCGStab /
        GMRES / FGMRES / MINRES) and the polynomial preconditioner.
        """
        if self._local_tensor is not None:
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
            y_full = self._local_tensor @ x_padded
            return y_full[:num_owned]

        local_matrix = self._local_matrix
        x_padded = self._pad_owned_to_local(x_owned)
        y_full = local_matrix.matvec(x_padded, exchange_halo=True)
        return y_full[:local_matrix.num_owned]

    # ------------------------------------------------------------------ #
    # The preconditioner factory + four Krylov methods (CG / BiCGStab /
    # GMRES / FGMRES / MINRES) live in :mod:`torch_sla.distributed_solve`
    # as free functions taking ``self`` as ``D``. ``solve_distributed_shard``
    # above dispatches to them.
    # ------------------------------------------------------------------ #

    def _matmul_dtensor(self, x: "DTensor") -> "DTensor":
        """
        Matrix-vector multiplication with DTensor input.
        
        Handles DTensor layout conversion and result wrapping.
        
        Parameters
        ----------
        x : DTensor
            Distributed tensor input
            
        Returns
        -------
        DTensor
            Result as DTensor with same placement as input
        """
        if not DTENSOR_AVAILABLE:
            raise RuntimeError("DTensor support requires PyTorch 2.0+")
        
        # Get DTensor metadata
        device_mesh = x.device_mesh
        placements = x.placements
        
        # Store original placement for output
        original_placements = tuple(placements)
        
        # Check if input is replicated (easiest case)
        is_replicated = all(isinstance(p, Replicate) for p in placements)
        
        if is_replicated:
            # Input is replicated on all ranks - just extract and compute
            x_local = x.to_local()
            y_local = self._distributed_matvec(x_local)
            # Wrap result as replicated DTensor
            return DTensor.from_local(y_local, device_mesh, [Replicate()])
        
        # Input is sharded - need to handle redistribution
        # For sparse matvec, we typically need the full vector on each rank
        # (because sparse matrix rows may reference any column)
        
        # Redistribute to replicated
        replicate_placements = [Replicate() for _ in placements]
        x_replicated = x.redistribute(device_mesh, replicate_placements)
        x_full = x_replicated.to_local()
        
        # Compute matvec with full vector
        y_full = self._distributed_matvec(x_full)
        
        # Wrap as replicated DTensor first
        y_replicated = DTensor.from_local(y_full, device_mesh, [Replicate()])
        
        # Redistribute back to original placement if it was sharded
        if not is_replicated:
            # For output, we shard along the row dimension (dim 0)
            # which corresponds to the matrix row partitioning
            output_placements = []
            for p in original_placements:
                if isinstance(p, Shard):
                    # Preserve shard dimension for output
                    output_placements.append(Shard(p.dim))
                else:
                    output_placements.append(Replicate())
            
            return y_replicated.redistribute(device_mesh, output_placements)
        
        return y_replicated
    
    # =========================================================================
    # Representation
    # =========================================================================
    
    def __repr__(self) -> str:
        return (f"DSparseTensor(shape={self._shape}, num_partitions={self._num_partitions}, "
                f"nnz={self.nnz}, device={self._device})")

