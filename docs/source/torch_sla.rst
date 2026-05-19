API Reference
=============

This section provides the complete API documentation for torch-sla.

----

Core Classes
------------

SparseTensor
~~~~~~~~~~~~

The main class for working with sparse matrices. Supports batched operations, automatic differentiation, and multiple backends.

.. autoclass:: torch_sla.SparseTensor
   :members:
   :undoc-members:
   :show-inheritance:

SparseTensorList
~~~~~~~~~~~~~~~~

Container for multiple sparse matrices with different sparsity patterns. Useful for batched operations on heterogeneous graphs.

.. autoclass:: torch_sla.SparseTensorList
   :members:
   :undoc-members:
   :show-inheritance:

LUFactorization
~~~~~~~~~~~~~~~

LU factorization for efficient repeated solves with the same matrix.

.. autoclass:: torch_sla.LUFactorization
   :members:
   :undoc-members:
   :show-inheritance:

----

Distributed Classes
-------------------

DSparseTensor
~~~~~~~~~~~~~

Distributed sparse tensor with domain decomposition support. Uses halo exchange for communication between partitions.

.. autoclass:: torch_sla.DSparseTensor
   :members:
   :undoc-members:
   :show-inheritance:

DSparseMatrix
~~~~~~~~~~~~~

Distributed sparse matrix designed for large-scale CFD/FEM computations. Provides domain decomposition with halo exchange.

.. autoclass:: torch_sla.DSparseMatrix
   :members:
   :undoc-members:
   :show-inheritance:

Partition
~~~~~~~~~

Dataclass representing a single partition/subdomain for distributed computing.

.. autoclass:: torch_sla.Partition
   :members:
   :undoc-members:

----

Linear Solve Functions
----------------------

spsolve
~~~~~~~

.. autofunction:: torch_sla.spsolve

spsolve_coo
~~~~~~~~~~~

.. autofunction:: torch_sla.spsolve_coo

spsolve_csr
~~~~~~~~~~~

.. autofunction:: torch_sla.spsolve_csr

----

Batch Solve Functions
---------------------

spsolve_batch_same_layout
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: torch_sla.spsolve_batch_same_layout

spsolve_batch_different_layout
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: torch_sla.spsolve_batch_different_layout

ParallelBatchSolver
~~~~~~~~~~~~~~~~~~~

.. autoclass:: torch_sla.ParallelBatchSolver
   :members:
   :undoc-members:
   :show-inheritance:

----

Nonlinear Solve
---------------

nonlinear_solve
~~~~~~~~~~~~~~~

.. autofunction:: torch_sla.nonlinear_solve

adjoint_solve
~~~~~~~~~~~~~

.. autofunction:: torch_sla.adjoint_solve

NonlinearSolveAdjoint
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: torch_sla.NonlinearSolveAdjoint
   :members:
   :undoc-members:
   :show-inheritance:

----

Persistence (I/O)
-----------------

safetensors Format
~~~~~~~~~~~~~~~~~~

save_sparse
^^^^^^^^^^^

.. autofunction:: torch_sla.save_sparse

load_sparse
^^^^^^^^^^^

.. autofunction:: torch_sla.load_sparse

save_distributed
^^^^^^^^^^^^^^^^

.. autofunction:: torch_sla.save_distributed

load_partition
^^^^^^^^^^^^^^

.. autofunction:: torch_sla.load_partition

load_metadata
^^^^^^^^^^^^^

.. autofunction:: torch_sla.load_metadata

load_sparse_as_partition
^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: torch_sla.load_sparse_as_partition

load_distributed_as_sparse
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: torch_sla.load_distributed_as_sparse

save_dsparse
^^^^^^^^^^^^

.. autofunction:: torch_sla.save_dsparse

load_dsparse
^^^^^^^^^^^^

.. autofunction:: torch_sla.load_dsparse

Matrix Market Format
~~~~~~~~~~~~~~~~~~~~

save_mtx
^^^^^^^^

.. autofunction:: torch_sla.save_mtx

load_mtx
^^^^^^^^

.. autofunction:: torch_sla.load_mtx

load_mtx_info
^^^^^^^^^^^^^

.. autofunction:: torch_sla.load_mtx_info

----

Partitioning Functions
----------------------

partition_graph_metis
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: torch_sla.partition_graph_metis

partition_coordinates
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: torch_sla.partition_coordinates

partition_simple
~~~~~~~~~~~~~~~~

.. autofunction:: torch_sla.partition_simple

----

Backend Utilities
-----------------

get_available_backends
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: torch_sla.get_available_backends

show_backends
~~~~~~~~~~~~~

.. autofunction:: torch_sla.show_backends

get_backend_methods
~~~~~~~~~~~~~~~~~~~

.. autofunction:: torch_sla.get_backend_methods

get_default_method
~~~~~~~~~~~~~~~~~~

.. autofunction:: torch_sla.get_default_method

select_backend
~~~~~~~~~~~~~~

.. autofunction:: torch_sla.select_backend

select_method
~~~~~~~~~~~~~

.. autofunction:: torch_sla.select_method

Backend Availability Checks
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: torch_sla.is_scipy_available

.. autofunction:: torch_sla.is_eigen_available

.. autofunction:: torch_sla.is_cupy_available

.. autofunction:: torch_sla.is_cudss_available

----

Utility Functions
-----------------

auto_select_method
~~~~~~~~~~~~~~~~~~

.. autofunction:: torch_sla.auto_select_method

estimate_direct_solver_memory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: torch_sla.estimate_direct_solver_memory

get_available_gpu_memory
~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: torch_sla.get_available_gpu_memory

----

Constants
---------

BACKEND_METHODS
~~~~~~~~~~~~~~~

Dictionary mapping backend names to available solver methods.

.. code-block:: python

   BACKEND_METHODS = {
       'scipy': ['lu', 'umfpack', 'cg', 'bicgstab', 'gmres', 'minres'],
       'eigen': ['cg', 'bicgstab'],
       'pytorch': ['cg', 'bicgstab'],
       'cupy': ['lu', 'cg', 'cgs', 'gmres', 'minres', 'lsqr', 'lsmr'],
       'cudss': ['lu', 'cholesky', 'ldlt'],
   }

DEFAULT_METHODS
~~~~~~~~~~~~~~~

Dictionary mapping backend names to their default solver methods.

.. code-block:: python

   DEFAULT_METHODS = {
       'scipy': 'lu',
       'eigen': 'bicgstab',
       'pytorch': 'cg',
       'cupy': 'lu',
       'cudss': 'cholesky',
   }

Type Aliases
~~~~~~~~~~~~

- ``BackendType``: Literal type for backend names: ``'scipy'``, ``'eigen'``, ``'pytorch'``, ``'cupy'``, ``'cudss'``
- ``MethodType``: Literal type for solver methods: ``'lu'``, ``'umfpack'``, ``'cg'``, ``'cgs'``, ``'bicgstab'``, ``'gmres'``, ``'minres'``, ``'cholesky'``, ``'ldlt'``, ``'lsqr'``, ``'lsmr'``
