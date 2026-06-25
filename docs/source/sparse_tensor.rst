SparseTensor
============

:class:`~torch_sla.SparseTensor` is the single-process sparse matrix. It stores
a matrix in COO form (``val``, ``row``, ``col`` plus a ``shape``), optionally
with a leading batch dimension, and carries the full operation vocabulary:
linear and nonlinear solves, eigendecomposition, SVD, matrix--vector products,
scalar/structural queries, graph analysis and visualization. Every solve is
differentiable through ``torch.autograd`` and dispatches to the backend best
suited to the device (SciPy / PyTorch-native on CPU, cuDSS / STRUMPACK on GPU).

For the matrix sharded across processes, see :doc:`dsparse_tensor`. For the
per-operation reference, see :doc:`operations`.

Construction
------------

A :class:`~torch_sla.SparseTensor` can be built directly from COO triplets or
through several convenience constructors:

.. code-block:: python

   import torch
   from torch_sla import SparseTensor

   # Direct COO: values, row indices, col indices, shape
   A = SparseTensor(val, row, col, (n, n))

   # From a dense matrix (drops the zeros)
   A = SparseTensor.from_dense(dense)

   # From an explicit list of (row, col, value) entries
   A = SparseTensor.from_coo_list(entries, shape=(n, n))

   # Structured constructors
   I  = SparseTensor.eye(n)                 # identity
   D  = SparseTensor.diag(values)           # diagonal
   T  = SparseTensor.tridiagonal(n, ...)    # tridiagonal band

Conversions back out:

.. code-block:: python

   dense = A.to_dense()          # torch.Tensor
   crow, col, val = A.to_csr()   # CSR triplet
   t = A.to_torch_sparse()       # torch.sparse_coo_tensor

A leading batch dimension (shape ``[B, n, n]``) makes every operation act on a
stack of same-pattern matrices in one call; see :ref:`op-solve-batch`. For
matrices with *different* patterns use
:class:`~torch_sla.SparseTensorList`.

Operation catalog
-----------------

Each operation links to its detailed entry in :doc:`operations` and to its API
object. Operations marked **(grad)** propagate gradients via the adjoint method.

Linear solves
~~~~~~~~~~~~~

.. list-table::
   :widths: 22 18 60
   :header-rows: 1

   * - Operation
     - API
     - Description
   * - :ref:`solve <op-solve>` **(grad)**
     - :meth:`~torch_sla.SparseTensor.solve`
     - Solve :math:`Ax = b`; auto-selects a direct or iterative backend.
   * - :ref:`solve_batch <op-solve-batch>`
     - :meth:`~torch_sla.SparseTensor.solve_batch`
     - Many right-hand sides / value sets sharing one sparsity pattern.
   * - :ref:`lu <op-lu>`
     - :meth:`~torch_sla.SparseTensor.lu`
     - Cache an LU factorization for repeated solves.

Nonlinear
~~~~~~~~~

.. list-table::
   :widths: 22 18 60
   :header-rows: 1

   * - Operation
     - API
     - Description
   * - :ref:`nonlinear_solve <op-nonlinear-solve>` **(grad)**
     - :meth:`~torch_sla.SparseTensor.nonlinear_solve`
     - Newton / Picard / Anderson solve of :math:`F(u, \theta) = 0`.

Eigen / spectral
~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 22 18 60
   :header-rows: 1

   * - Operation
     - API
     - Description
   * - :ref:`eigsh <op-eigsh>` **(grad)**
     - :meth:`~torch_sla.SparseTensor.eigsh`
     - Top-k eigenpairs of a symmetric/Hermitian matrix.
   * - :ref:`eigs <op-eigsh>`
     - :meth:`~torch_sla.SparseTensor.eigs`
     - Top-k eigenpairs of a general matrix.
   * - :ref:`svd <op-svd>` **(grad)**
     - :meth:`~torch_sla.SparseTensor.svd`
     - Truncated rank-k singular value decomposition.

Matrix--vector
~~~~~~~~~~~~~~

.. list-table::
   :widths: 22 18 60
   :header-rows: 1

   * - Operation
     - API
     - Description
   * - :ref:`matvec / @ <op-matvec>`
     - :meth:`~torch_sla.SparseTensor.__matmul__`
     - Sparse matrix--vector / matrix--matrix product (SpMV).

Scalar / structural
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 22 18 60
   :header-rows: 1

   * - Operation
     - API
     - Description
   * - :ref:`det <op-det>` **(grad)**
     - :meth:`~torch_sla.SparseTensor.det`
     - Determinant via sparse LU.
   * - :ref:`logdet <op-logdet>` **(grad)**
     - :meth:`~torch_sla.SparseTensor.logdet`
     - Log-determinant (numerically stable for large matrices).
   * - :ref:`norm <op-norm>` **(grad)**
     - :meth:`~torch_sla.SparseTensor.norm`
     - Frobenius / 1- / 2-norm.
   * - :ref:`condition_number <op-condition-number>`
     - :meth:`~torch_sla.SparseTensor.condition_number`
     - Ratio :math:`\sigma_{\max}/\sigma_{\min}`.
   * - :ref:`is_symmetric / is_positive_definite <op-predicates>`
     - :meth:`~torch_sla.SparseTensor.is_symmetric`
     - Structural predicates for solver selection.

Graph
~~~~~

.. list-table::
   :widths: 22 18 60
   :header-rows: 1

   * - Operation
     - API
     - Description
   * - :ref:`connected_components <op-connected-components>`
     - :meth:`~torch_sla.SparseTensor.connected_components`
     - Label connected components of the adjacency pattern.

Visualization
~~~~~~~~~~~~~

.. list-table::
   :widths: 22 18 60
   :header-rows: 1

   * - Operation
     - API
     - Description
   * - :ref:`spy <op-spy>`
     - :meth:`~torch_sla.SparseTensor.spy`
     - Plot the sparsity pattern as a matplotlib figure.

I/O and reductions
~~~~~~~~~~~~~~~~~~

Beyond the headline operations, :class:`~torch_sla.SparseTensor` also offers
save/load (safetensors and Matrix Market), element-wise math (``abs``,
``sqrt``, ``exp``, ``log``, ...) and reductions (``sum``, ``mean``, ``max``,
``min``). These are documented in the :doc:`API reference <torch_sla>`.
