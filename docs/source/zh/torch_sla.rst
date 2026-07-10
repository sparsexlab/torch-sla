API 参考
========

本节提供 torch-sla 的完整 API 文档。

----

核心类
------

SparseTensor
~~~~~~~~~~~~

稀疏矩阵操作的主类。支持批量操作、自动微分和多种后端。

.. autoclass:: torch_sla.SparseTensor
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __matmul__

SparseTensorList
~~~~~~~~~~~~~~~~

不同稀疏模式的多个稀疏矩阵容器。适用于异构图的批量操作。

.. autoclass:: torch_sla.SparseTensorList
   :members:
   :undoc-members:
   :show-inheritance:

LUFactorization
~~~~~~~~~~~~~~~

LU 分解，用于相同矩阵的高效重复求解。

.. autoclass:: torch_sla.LUFactorization
   :members:
   :undoc-members:
   :show-inheritance:

----

分布式类
--------

DSparseTensor
~~~~~~~~~~~~~

支持域分解的分布式稀疏张量。使用 halo 交换进行分区间通信。

.. autoclass:: torch_sla.DSparseTensor
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __matmul__

Partition
~~~~~~~~~

表示分布式计算中单个分区/子域的数据类。

.. autoclass:: torch_sla.Partition
   :members:
   :undoc-members:

----

线性求解函数
------------

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

批量求解函数
------------

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

非线性求解
----------

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

持久化 (I/O)
------------

safetensors 格式
~~~~~~~~~~~~~~~~

save_sparse
^^^^^^^^^^^

.. autofunction:: torch_sla.save_sparse

load_sparse
^^^^^^^^^^^

.. autofunction:: torch_sla.load_sparse

load_metadata
^^^^^^^^^^^^^

.. autofunction:: torch_sla.load_metadata

save_sparse_sharded
^^^^^^^^^^^^^^^^^^^

.. autofunction:: torch_sla.save_sparse_sharded

load_sparse_shard
^^^^^^^^^^^^^^^^^

.. autofunction:: torch_sla.load_sparse_shard

save_dsparse
^^^^^^^^^^^^

.. autofunction:: torch_sla.save_dsparse

load_dsparse
^^^^^^^^^^^^

.. autofunction:: torch_sla.load_dsparse

Matrix Market 格式
~~~~~~~~~~~~~~~~~~

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

分区函数
--------

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

后端工具
--------

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

后端可用性检查
~~~~~~~~~~~~~~

.. autofunction:: torch_sla.is_scipy_available

.. autofunction:: torch_sla.is_pytorch_available

.. autofunction:: torch_sla.is_cudss_available

.. autofunction:: torch_sla.is_pyamg_available

.. autofunction:: torch_sla.is_amgx_available

.. autofunction:: torch_sla.is_strumpack_available

----

工具函数
--------

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

常量
----

BACKEND_METHODS
~~~~~~~~~~~~~~~

后端名称到可用求解方法的映射字典。

.. code-block:: python

   BACKEND_METHODS = {
       'scipy': ['lu', 'umfpack', 'cg', 'bicgstab', 'gmres', 'lgmres', 'minres', 'qmr'],
       'pytorch': ['cg', 'bicgstab', 'gmres', 'minres', 'lsqr', 'lsmr'],  # 设备无关 (CPU/CUDA/ROCm)
       'cudss': ['lu', 'cholesky', 'ldlt'],                               # 仅 NVIDIA CUDA
       'pyamg': ['amg', 'ruge_stuben', 'smoothed_aggregation', 'sa'],
       'amgx': ['amg', 'cg', 'pcg', 'bicgstab', 'pbicgstab', 'gmres', 'fgmres'],
       'strumpack': ['lu'],                                               # 多波前直接求解 (CPU/CUDA/ROCm)
   }

DEFAULT_METHODS
~~~~~~~~~~~~~~~

后端名称到默认求解方法的映射字典。

.. code-block:: python

   DEFAULT_METHODS = {
       'scipy': 'lu',
       'pytorch': 'cg',
       'cudss': 'cholesky',
       'pyamg': 'ruge_stuben',
       'amgx': 'pbicgstab',
       'strumpack': 'lu',
   }

类型别名
~~~~~~~~

- ``BackendType``：后端名称的 Literal 类型：``'scipy'``、``'pytorch'``、``'cudss'``、``'pyamg'``、``'amgx'``、``'strumpack'``、``'auto'``
- ``MethodType``：求解方法的 Literal 类型：``'lu'``、``'umfpack'``、``'cg'``、``'cgs'``、``'bicgstab'``、``'gmres'``、``'minres'``、``'cholesky'``、``'ldlt'``、``'lsqr'``、``'lsmr'``
