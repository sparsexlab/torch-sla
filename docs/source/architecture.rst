Architecture
============

torch-sla's class hierarchy and distributed model mirror PyTorch's own
``torch.Tensor`` / ``torch.distributed.tensor.DTensor`` split: a single
sparse "local data" class, and a thin distributed wrapper that adds
placement + mesh metadata on top. This page is the source of truth for
the design contracts every new feature should respect.

----

.. _arch-class-hierarchy:

Class hierarchy
---------------

.. list-table::
   :widths: 30 35 35
   :header-rows: 1

   * - Role
     - PyTorch
     - torch-sla
   * - **Local data**
     - ``torch.Tensor``
     - :class:`~torch_sla.SparseTensor`
   * - **Distributed wrapper** (local data + spec)
     - ``torch.distributed.tensor.DTensor``
     - :class:`~torch_sla.DSparseTensor`
   * - **Per-rank local chunk**
     - ``DTensor._local_tensor`` (a ``torch.Tensor``)
     - ``DSparseTensor._local_tensor`` (a ``SparseTensor``)
   * - **Distributed metadata**
     - ``DTensor._spec`` (``DTensorSpec``)
     - ``DSparseTensor._spec`` (``DSparseSpec``)
   * - **Sharding placement**
     - ``Shard(dim)``, ``Replicate()``, ``Partial(op)``
     - :class:`~torch_sla.SparseShard(axis)`, :class:`~torch_sla.Replicated`

Key invariant: ``SparseTensor`` is **always** local data. ``DSparseTensor``
is **always** a distributed wrapper holding one rank's ``SparseTensor`` plus
a spec. No "hybrid" class.

.. note::

   ``DSparseMatrix`` is the historical per-rank chunk class from before
   the DTensor-mirror refactor. It currently coexists with
   ``SparseTensor`` inside ``DSparseTensor._local_tensor``; see
   :ref:`arch-dsparse-dissolution` for the migration plan.

----

.. _arch-shape-contract:

Shape contract: ``(*batch, M, N, *block)``
-------------------------------------------

A :class:`~torch_sla.SparseTensor` always has the canonical shape

.. code-block:: text

   shape = (*batch_shape, M, N, *block_shape)
           └──dense───┘   └sparse└──dense──┘
            leading        2 dims  trailing

The two sparse dimensions are **always** the matrix axes ``M`` and
``N`` -- they cannot move. Dense axes flank them:

* ``batch_shape`` (left) -- dense batch dims for batched SpMV / solve
* ``block_shape`` (right) -- dense block dims for block-sparse formats
  (BSR / BCSC)

If a user has a tensor where the sparse axes aren't in this slot,
``SparseTensor.permute(...)`` reorders to the canonical layout. The
contract is positional, not by sparse-dim metadata, so every algorithm
(matvec, solve, eigsh, ...) knows where to look.

----

.. _arch-placement-vocab:

Placement vocabulary
--------------------

A ``DSparseSpec`` carries:

* ``placement``: how the data is sharded
* ``mesh``: which devices it's sharded over
* ``global_shape``: the original full-tensor shape

``placement`` is either a single placement (1-D mesh) or a list, one
element per mesh dimension (multi-D mesh -- same convention as
DTensor).

.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - Class
     - Axis kind
     - Use case
   * - :class:`~torch_sla.Replicated`
     - --
     - Full matrix on every rank
   * - ``torch.distributed.tensor.Shard(dim)``
     - dense axis (batch or block)
     - Per-rank gets a slice of batches; no cross-rank communication for SpMV
   * - :class:`~torch_sla.SparseShard(axis)`
     - sparse axis (``axis=len(batch_shape)`` for rows, ``+1`` for cols)
     - Irregular row/col partition of the matrix; needs halo exchange or all-reduce
   * - :class:`~torch_sla.SparseShard(axis)` with hypergraph-derived partition
     - sparse axis
     - Minimal-communication SpMV via PaToH / Mondriaan hypergraph cut

Convenience constructors :func:`~torch_sla.row_shard` and
:func:`~torch_sla.col_shard` cover the common 2-D-matrix case:

.. code-block:: python

   from torch_sla import row_shard, col_shard, SparseShard

   row_shard()              # SparseShard(axis=0), plain (M, N)
   col_shard()              # SparseShard(axis=1)
   row_shard(batch_ndim=2)  # SparseShard(axis=2), for (B1, B2, M, N) tensor

Multi-axis sharding on a 2-D mesh: pass a list, exactly like DTensor.

.. code-block:: python

   from torch.distributed.tensor import Shard
   from torch_sla import SparseShard

   # 2-D mesh: 4 batch shards × 8 row shards
   mesh = init_device_mesh("cuda", (4, 8))
   placement = [Shard(0),              # mesh dim 0: dense batch dim
                SparseShard(axis=2)]   # mesh dim 1: sparse row axis (batch_ndim=2)

----

.. _arch-matvec-dispatch:

Matvec dispatch by placement
----------------------------

``DSparseTensor.__matmul__`` dispatches on placement to pick the right
communication pattern. Each row of this table is a separate code path:

.. list-table::
   :widths: 30 35 35
   :header-rows: 1

   * - Placement
     - matvec algorithm
     - Cross-rank communication
   * - ``Replicated``
     - local ``A @ x`` (no comm)
     - none
   * - ``Shard(batch_dim)``
     - per-batch independent SpMV
     - none (embarrassingly parallel)
   * - ``SparseShard(row_axis)``
     - halo exchange + local SpMV
     - O(halo nnz) point-to-point
   * - ``SparseShard(col_axis)``
     - local partial SpMV + ``all_reduce(SUM)``
     - O(M) all-reduce
   * - 2-D placement list ``[SparseShard(M), SparseShard(N)]``
     - 2-D Cannon / SUMMA
     - O(sqrt) better than 1-D for large mesh

----

.. _arch-partition-algorithms:

Partition algorithms
--------------------

Picking how rows / cols get distributed is its own subproblem. The
options below are scoring tradeoffs and Python-binding maturity for
each:

.. list-table::
   :widths: 18 22 22 38
   :header-rows: 1

   * - Algorithm
     - vs. METIS quality
     - Python binding
     - Best for
   * - ``simple`` / striped
     - much worse (no locality)
     - n/a (~10 LOC)
     - sanity tests, deterministic across ranks
   * - METIS (current default)
     - baseline
     - ``pymetis`` stable
     - graphs up to ~100M nodes
   * - Hilbert space-filling curve
     - worse but ~10-100x faster
     - pure Python or ``pyhilbert``
     - PDE meshes / geometric structure
   * - KaHIP
     - +20% quality
     - ``kahip-python`` finicky; subprocess shell-out also works
     - graphs up to ~1B nodes
   * - Mt-METIS
     - same quality, 4-16x faster
     - no Python binding; C call from ctypes
     - mid-size users with many CPU cores
   * - PaToH (hypergraph)
     - **minimal SpMV communication** -- theoretical optimum
     - ``pypatoh`` half-maintained
     - sparse matvec specifically
   * - Mondriaan
     - similar to PaToH, 2-D-specific
     - command-line wrapper
     - sparse matrices specifically
   * - ParMETIS
     - METIS-quality, distributed
     - no Python binding (MPI C only)
     - true HPC clusters
   * - GNN-based learned
     - research-grade
     - DIY implementation
     - very-large / streaming graphs (>1B edges)

torch-sla's :meth:`~torch_sla.SparseTensor.partition_for_rank` exposes
the partition through the ``partition_method`` kwarg. Today it
supports ``simple``, ``metis``, ``rcb``, ``slicing``. ``hilbert`` and
``patoh`` are tracked follow-ups.

----

.. _arch-dsparse-dissolution:

DSparseMatrix dissolution (Phase B)
-----------------------------------

Historical: ``DSparseMatrix`` predates the DTensor-mirror refactor and
holds both **local data** (values, row, col, partition map, CSR cache)
and **distributed-aware methods** (``matvec``, ``halo_exchange``,
``solve``). After the refactor, ``DSparseTensor`` wraps a
``DSparseMatrix`` plus a spec -- which means two classes carry
overlapping responsibilities for "the local rank's chunk".

The target architecture (Phase B):

* The local chunk is a plain ``SparseTensor`` in **local coordinates**
  (``num_local × num_local`` CSR, no awareness of being "distributed").
* All partition metadata (``owned_nodes``, ``halo_nodes``,
  ``neighbor_partitions``) moves into the ``SparseShard`` placement.
* All distributed-aware methods (``matvec``, ``halo_exchange``,
  ``solve``) move onto ``DSparseTensor`` and read from the spec.
* ``DSparseMatrix`` becomes a deprecated alias that constructs a
  ``DSparseTensor`` internally and delegates the old method names.

Migration steps (one PR each, non-breaking):

1. Lift ``Partition`` out of ``distributed.py``; let
   ``SparseShard(axis, partition=)`` carry the metadata.
2. Add ``SparseTensor.extract_partition(partition)`` -- builds the
   local subdomain as a plain ``SparseTensor`` in local coordinates.
3. Migrate ``DSparseTensor._matmul_spec`` to operate on ``SparseTensor``
   directly. Move ``halo_exchange`` onto ``DSparseTensor``.
4. Migrate the in-tree Krylov methods (``_distributed_*_shard``) to
   call the new ``DSparseTensor``-native matvec. No-op behaviourally
   if step 3 is correct.
5. ``DSparseMatrix`` becomes a ``DeprecationWarning``-emitting alias.
6. A future release removes ``DSparseMatrix`` entirely.

----

Why this matches DTensor
------------------------

Every design choice above maps 1:1 onto a corresponding DTensor
decision:

* ``SparseTensor`` ≅ ``torch.Tensor`` -- same "local data" role.
* ``DSparseTensor`` ≅ ``DTensor`` -- same "(local + spec)" structure.
* ``SparseShard(axis)`` ≅ ``Shard(dim)`` -- one parameterised placement
  per sharded direction, not separate classes.
* Placement *list* over mesh dims ≅ DTensor's multi-axis sharding.
* Spec's ``mesh`` + ``global_shape`` separation ≅ DTensorSpec.

By staying parallel to DTensor, torch-sla composes cleanly with
PyTorch's distributed ecosystem (FSDP, TP, DCP) -- a sparse vector
result from ``DSparseTensor.matvec`` is already a ``DTensor`` with the
right placement, ready to feed into a downstream FSDP module.
