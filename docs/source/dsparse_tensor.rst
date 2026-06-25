DSparseTensor
=============

:class:`~torch_sla.DSparseTensor` is the distributed counterpart of
:class:`~torch_sla.SparseTensor`. It partitions the rows of a sparse matrix
across the ranks of a ``DeviceMesh`` using domain decomposition: each rank owns
a contiguous block of rows and a halo of ghost columns referenced by its rows
but owned elsewhere. Vectors stay rank-local; the only communication is a
halo exchange before each matrix--vector product and an all-reduce for the
inner products inside Krylov iterations. The wrapper mirrors
``torch.distributed.tensor.DTensor`` and reuses the single-process operations
rank-locally, so the operation vocabulary matches :doc:`sparse_tensor`.

Partitioning
------------

A distributed tensor is built from a global :class:`~torch_sla.SparseTensor`
and a mesh. Three entry points cover the common layouts:

.. code-block:: python

   from torch.distributed.device_mesh import init_device_mesh
   from torch_sla import SparseTensor, DSparseTensor

   mesh = init_device_mesh("cpu", (world_size,))
   A = SparseTensor(val, row, col, (n, n))

   # Row-partition a single matrix across the mesh (RowPartitioned placement)
   D = DSparseTensor.partition(A, mesh, partition_method="metis")

   # Partition a batch of same-pattern matrices, one shard per rank (BatchShard)
   D = DSparseTensor.partition_batch(A_batched, mesh)

   # Re-wrap shards that were already produced per-rank (no re-partitioning)
   D = DSparseTensor.from_global_distributed(local_shard, spec, mesh)

``partition_method`` selects how rows are assigned: ``"simple"`` (contiguous
blocks), ``"metis"`` (graph partition, minimizes halo), or ``"coordinates"``
(geometric, needs ``coords``). The owned/halo bookkeeping is computed once and
cached on the placement.

Scatter / gather move a global vector in and out of the sharded layout:

.. code-block:: python

   d = D.scatter(global_vec)     # global torch.Tensor -> DTensor[Shard(0)]
   y = (D @ d).full_tensor()     # gather a sharded result back to global

Distributed solves and the sugar API
-------------------------------------

The ``*_shard`` methods are the primitives: they operate entirely in
Shard(0) space (each vector sized to the rank's owned rows) and drive the
Krylov iteration with halo-exchange SpMV plus all-reduce dot products. On top
of them sits a thin sugar layer whose names and signatures match
:class:`~torch_sla.SparseTensor`, so single-process code ports with minimal
change:

.. list-table::
   :widths: 30 35 35
   :header-rows: 1

   * - Sugar method
     - Delegates to
     - Mirror of
   * - :meth:`~torch_sla.DSparseTensor.solve`
     - ``solve_distributed_shard``
     - :meth:`SparseTensor.solve <torch_sla.SparseTensor.solve>`
   * - :meth:`~torch_sla.DSparseTensor.solve_batch`
     - ``solve_batch_shard``
     - :meth:`SparseTensor.solve_batch <torch_sla.SparseTensor.solve_batch>`
   * - :meth:`~torch_sla.DSparseTensor.nonlinear_solve`
     - ``nonlinear_solve_distributed_shard``
     - :meth:`SparseTensor.nonlinear_solve <torch_sla.SparseTensor.nonlinear_solve>`
   * - :meth:`~torch_sla.DSparseTensor.connected_components`
     - ``connected_components_shard``
     - :meth:`SparseTensor.connected_components <torch_sla.SparseTensor.connected_components>`
   * - :meth:`~torch_sla.DSparseTensor.lsqr` / :meth:`~torch_sla.DSparseTensor.lsmr`
     - ``lsqr_shard`` / ``lsmr_shard``
     - ``spsolve(method='lsqr'/'lsmr')``

Every distributed result is *rank-invariant*: the same global solution,
eigenvalue or component labelling regardless of world size or partition method.

.. code-block:: python

   b = D.scatter(global_b)
   x = D.solve(b)                       # distributed CG, DTensor in / DTensor out
   x_global = x.full_tensor()           # gather to a single rank

.. seealso::

   :doc:`distributed_scaling` -- how to benchmark weak / strong /
   throughput scaling of the distributed solve with ``torchrun``, what
   the metrics mean, and how to extend it. Script:
   ``benchmarks/distributed/scaling/distributed_solve_scaling.py``.

Halo-exchange SpMV
------------------

The matrix--vector product ``D @ x`` (and the matvec inside every solve) is the
one place ranks communicate during a kernel. Before multiplying, each rank
fills its halo entries with the owned values from neighboring ranks via a
point-to-point halo exchange, then runs a purely local SpMV over its owned
rows. This keeps memory and compute per rank proportional to its share of the
matrix; see :ref:`op-distributed-matvec`.

The diagram below shows two ranks. Each owns a contiguous block of rows
(solid nodes). A row owned by rank 0 may reference a column owned by rank 1
(and vice versa); those off-rank entries are the *halo* / *ghost* slots
(dashed). The dashed cross-rank arrows are the halo exchange: each rank sends
its owned boundary values to fill the neighbor's ghost slots, after which both
ranks run a fully local SpMV.

.. graphviz::
   :alt: Halo exchange between two ranks before a distributed SpMV
   :caption: Halo exchange: boundary values are sent to fill the neighbor's
             ghost slots, then each rank runs a local SpMV.

   digraph halo {
       rankdir=LR;
       node [shape=circle, fontsize=10, fixedsize=true, width=0.45];
       edge [fontsize=9];

       subgraph cluster_r0 {
           label="rank 0  (owns rows 0..k)";
           style=rounded; color="#3b6ea5"; fontcolor="#3b6ea5";
           o0a [label="0", style=filled, fillcolor="#cfe2f3"];
           o0b [label="1", style=filled, fillcolor="#cfe2f3"];
           o0bnd [label="k", style=filled, fillcolor="#9fc5e8"];
           g0 [label="ghost", style="dashed,filled", fillcolor="#f4f4f4"];
           o0a -> o0b -> o0bnd [style=invis];
       }

       subgraph cluster_r1 {
           label="rank 1  (owns rows k+1..n)";
           style=rounded; color="#a55b3b"; fontcolor="#a55b3b";
           o1bnd [label="k+1", style=filled, fillcolor="#f9cb9c"];
           o1a [label="...", style=filled, fillcolor="#fce5cd"];
           o1b [label="n", style=filled, fillcolor="#fce5cd"];
           g1 [label="ghost", style="dashed,filled", fillcolor="#f4f4f4"];
           o1bnd -> o1a -> o1b [style=invis];
       }

       // halo exchange: boundary owned value -> neighbor's ghost slot
       o0bnd -> g1 [label="send boundary", color="#3b6ea5",
                    style=dashed, constraint=false];
       o1bnd -> g0 [label="send boundary", color="#a55b3b",
                    style=dashed, constraint=false];

       // local SpMV consumes the now-filled ghost
       g0 -> o0bnd [label="local SpMV", color="#888888", style=dotted];
       g1 -> o1bnd [label="local SpMV", color="#888888", style=dotted];
   }

Operation catalog
-----------------

The distributed operations cross-reference their single-process equivalents on
:class:`~torch_sla.SparseTensor`. Operations not listed (``det``, ``logdet``,
``svd``, ``condition_number``) gather to one rank before computing and exist
for convenience rather than scale.

.. list-table::
   :widths: 26 26 48
   :header-rows: 1

   * - Operation
     - API
     - Single-process equivalent
   * - :ref:`partition <op-partition>`
     - :meth:`~torch_sla.DSparseTensor.partition`
     - construction from a global :class:`~torch_sla.SparseTensor`
   * - :ref:`solve <op-distributed-solve>`
     - :meth:`~torch_sla.DSparseTensor.solve`
     - :ref:`SparseTensor.solve <op-solve>`
   * - :ref:`solve_batch <op-solve-batch>`
     - :meth:`~torch_sla.DSparseTensor.solve_batch`
     - :ref:`SparseTensor.solve_batch <op-solve-batch>`
   * - :ref:`nonlinear_solve <op-nonlinear-solve>`
     - :meth:`~torch_sla.DSparseTensor.nonlinear_solve`
     - :ref:`SparseTensor.nonlinear_solve <op-nonlinear-solve>`
   * - :ref:`matvec / @ <op-distributed-matvec>`
     - :meth:`~torch_sla.DSparseTensor.__matmul__`
     - :ref:`SparseTensor.matvec <op-matvec>`
   * - :ref:`eigsh <op-distributed-eigsh>`
     - :meth:`~torch_sla.DSparseTensor.eigsh`
     - :ref:`SparseTensor.eigsh <op-eigsh>`
   * - :ref:`connected_components <op-distributed-cc>`
     - :meth:`~torch_sla.DSparseTensor.connected_components`
     - :ref:`SparseTensor.connected_components <op-connected-components>`
   * - :ref:`lsqr / lsmr <op-distributed-solve>`
     - :meth:`~torch_sla.DSparseTensor.lsqr`
     - ``spsolve(method='lsqr'/'lsmr')``

Save / load of a sharded tensor (one file per partition) is handled by
:meth:`~torch_sla.DSparseTensor.save` / :meth:`~torch_sla.DSparseTensor.load`
and the functional :func:`~torch_sla.save_distributed` /
:func:`~torch_sla.load_partition`.
