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

Halo-exchange SpMV
------------------

The matrix--vector product ``D @ x`` (and the matvec inside every solve) is the
one place ranks communicate during a kernel. Before multiplying, each rank
fills its halo entries with the owned values from neighboring ranks via a
point-to-point halo exchange, then runs a purely local SpMV over its owned
rows. This keeps memory and compute per rank proportional to its share of the
matrix; see :ref:`op-distributed-matvec`.

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
