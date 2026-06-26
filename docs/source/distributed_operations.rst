Distributed operations (multi-node) — hand-off guide
====================================================

This page is the **hand-off guide** for the runnable distributed
operation examples in ``examples/distributed/``. Each script row-shards a
:class:`~torch_sla.DSparseTensor` across a device mesh, runs one
distributed operation, and asserts a correctness gate against a
single-process reference so anyone can confirm the op actually works on a
real multi-node, multi-GPU cluster.

Every example is **device-aware**: it selects ``nccl`` + CUDA when
``torch.cuda.is_available()`` (pinning one GPU per ``LOCAL_RANK`` via
``torch.cuda.set_device(LOCAL_RANK)`` and building the mesh on
``"cuda"``) and otherwise falls back to ``gloo`` + CPU, so the identical
script runs on a laptop or a GPU cluster.

How to launch
-------------

``torchrun`` spawns one process (*rank*) per ``--nproc_per_node`` and
sets ``WORLD_SIZE`` / ``RANK`` / ``LOCAL_RANK`` in the environment. The
scripts read ``LOCAL_RANK`` to pin their GPU.

**Single node** (``--standalone``):

.. code-block:: bash

   cd examples/distributed
   torchrun --standalone --nproc_per_node=4 distributed_matvec.py

**Multiple nodes** (``c10d`` rendezvous; run the *same* command on
*every* node, with ``HEAD_NODE_IP:29500`` reachable from all):

.. code-block:: bash

   # Recommended NCCL settings on a real cluster:
   export NCCL_DEBUG=INFO          # print the transport NCCL selected
   export NCCL_SOCKET_IFNAME=eth0  # pin the NIC if auto-detect picks wrong
   export NCCL_IB_DISABLE=0        # keep InfiniBand on (1 forces TCP)

   # on EVERY node (e.g. 2 nodes × 4 GPUs = world size 8):
   torchrun \
       --nnodes=2 --nproc_per_node=4 \
       --rdzv-id=sla --rdzv-backend=c10d \
       --rdzv-endpoint=HEAD_NODE_IP:29500 \
       distributed_matvec.py

The ``examples/distributed/launch.sh`` helper wraps both forms: set
``RDZV_ENDPOINT`` (and optionally ``NNODES`` / ``RDZV_ID``) to switch
from ``--standalone`` to the multi-node ``--rdzv-*`` form, e.g.
``RDZV_ENDPOINT=HEAD_NODE_IP:29500 NNODES=2 ./launch.sh all 4``.

The five operations
-------------------

matvec
~~~~~~

Distributed sparse matrix-vector product ``y = A @ x`` with automatic
halo exchange.

* **Script**: ``examples/distributed/distributed_matvec.py``
* **Single node**:
  ``torchrun --standalone --nproc_per_node=4 distributed_matvec.py``
* **Multi node**:
  ``torchrun --nnodes=2 --nproc_per_node=4 --rdzv-id=sla --rdzv-backend=c10d --rdzv-endpoint=HEAD_NODE_IP:29500 distributed_matvec.py``
* **Gate**: ``max|y_dist − y_ref|`` vs the single-process ``A @ x``
  (asserts ``< 1e-12``).

solve
~~~~~

Distributed Krylov linear solve ``A x = b`` (CG with Jacobi
preconditioning) in Shard(0) space.

* **Script**: ``examples/distributed/distributed_solve.py``
* **Single node**:
  ``torchrun --standalone --nproc_per_node=4 distributed_solve.py``
* **Multi node**:
  ``torchrun --nnodes=2 --nproc_per_node=4 --rdzv-id=sla --rdzv-backend=c10d --rdzv-endpoint=HEAD_NODE_IP:29500 distributed_solve.py``
* **Gate**: relative residual ``‖b − A x‖ / ‖b‖`` (asserts ``< 1e-8``)
  **and** relative difference vs a single-process scipy CG
  (asserts ``< 1e-6``).

eigsh
~~~~~

Distributed symmetric eigensolver (LOBPCG) for the smallest eigenpairs.

* **Script**: ``examples/distributed/distributed_eigsh.py``
* **Single node**:
  ``torchrun --standalone --nproc_per_node=4 distributed_eigsh.py``
* **Multi node**:
  ``torchrun --nnodes=2 --nproc_per_node=4 --rdzv-id=sla --rdzv-backend=c10d --rdzv-endpoint=HEAD_NODE_IP:29500 distributed_eigsh.py``
* **Gate**: per-eigenvalue relative error vs ``scipy.sparse.linalg.eigsh``
  (asserts each ``< 1e-3``).

connected_components
~~~~~~~~~~~~~~~~~~~~~

Distributed connected components on a row-sharded adjacency via label
propagation with boundary-label halo exchange.

* **Script**: ``examples/distributed/distributed_connected_components.py``
* **Single node**:
  ``torchrun --standalone --nproc_per_node=4 distributed_connected_components.py``
* **Multi node**:
  ``torchrun --nnodes=2 --nproc_per_node=4 --rdzv-id=sla --rdzv-backend=c10d --rdzv-endpoint=HEAD_NODE_IP:29500 distributed_connected_components.py``
* **Gate**: global component count equals the expected count **and** the
  ``scipy.sparse.csgraph.connected_components`` count; the induced node
  partition matches scipy's (numbering-invariant comparison).

nonlinear_solve
~~~~~~~~~~~~~~~

Distributed Newton solve of a semilinear system ``F(u) = A u − λ exp(u) = 0``
(Bratu-type; diagonal Jacobian shift, Newton step via distributed GMRES).

* **Script**: ``examples/distributed/distributed_nonlinear_solve.py``
* **Single node**:
  ``torchrun --standalone --nproc_per_node=4 distributed_nonlinear_solve.py``
* **Multi node**:
  ``torchrun --nnodes=2 --nproc_per_node=4 --rdzv-id=sla --rdzv-backend=c10d --rdzv-endpoint=HEAD_NODE_IP:29500 distributed_nonlinear_solve.py``
* **Gate**: global residual ``‖F(u*)‖`` (asserts ``< 1e-8``) **and**
  relative difference vs a single-process Newton on the full system
  (asserts ``< 1e-6``).

Collecting results
------------------

Each script prints a ``[rank R] ...`` line on every rank and the final
verdict on rank 0. The distributed results live as **owned slices** (one
row-block per rank); to inspect or persist the full vector, assemble it
on every rank with
:func:`torch_sla.distributed.gather_owned_to_global`:

.. code-block:: python

   from torch_sla.distributed import gather_owned_to_global

   owned = D._spec.placement.partition.owned_nodes
   full = gather_owned_to_global(owned, owned_values, N_global)  # all ranks

The matvec / solve examples instead use ``DTensor`` wrappers
(``D.scatter(x_global)`` and ``.full_tensor()``) to move between the
global vector and the Shard(0) layout. When capturing logs from a
multi-node run, redirect each node's ``torchrun`` output to its own file
— rank 0's file carries the assertions and the final ``converged`` line.

.. seealso::

   * :doc:`distributed_scaling` — performance/scaling hand-off guide.
   * ``examples/distributed/README.md`` — the same launch matrix plus an
     API-at-a-glance snippet.
