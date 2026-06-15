"""Distributed sparse tensor stack.

Re-exports the public surface so ``from torch_sla.distributed import X``
keeps working after the package split. Implementation lives in:

* :mod:`.core`    -- :class:`DSparseTensor` + :class:`DSparseSpec` +
  placement vocabulary (:class:`SparseShard`, :class:`Replicated`).
* :mod:`.matvec`  -- ``D @ x`` / hot-path ``_shard_matvec`` / halo exchange.
* :mod:`.solve`   -- Shard(0) Krylov methods + preconditioners.
* :mod:`.eigsh`   -- distributed LOBPCG.
* :mod:`.collectives` -- owned-slice <-> global-vector helpers.
"""
from .core import (
    DSparseTensor,
    DSparseSpec,
    Replicated,
    VertexShard,
    VertexShardReplicated,
    BatchShard,
    SparseShard,  # deprecated alias for VertexShard / VertexShardReplicated
    # Re-exported from torch_sla.partition for back-compat.
    Partition,
    partition_graph_metis,
    partition_coordinates,
    partition_simple,
    _hilbert_sort_indices,
    _hilbert_curve_indices,
)
from .collectives import gather_owned_to_global

__all__ = [
    "DSparseTensor",
    "DSparseSpec",
    "Replicated",
    "VertexShard",
    "VertexShardReplicated",
    "BatchShard",
    "SparseShard",
    "Partition",
    "partition_graph_metis",
    "partition_coordinates",
    "partition_simple",
    "gather_owned_to_global",
]
