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

----

非线性求解
----------

nonlinear_solve
~~~~~~~~~~~~~~~

.. autofunction:: torch_sla.nonlinear_solve

adjoint_solve
~~~~~~~~~~~~~

.. autofunction:: torch_sla.adjoint_solve

----

持久化 (I/O)
------------

safetensors 格式
~~~~~~~~~~~~~~~~

.. autofunction:: torch_sla.save_sparse

.. autofunction:: torch_sla.load_sparse

.. autofunction:: torch_sla.save_distributed

.. autofunction:: torch_sla.load_partition

Matrix Market 格式
~~~~~~~~~~~~~~~~~~

.. autofunction:: torch_sla.save_mtx

.. autofunction:: torch_sla.load_mtx

----

后端工具
--------

.. autofunction:: torch_sla.get_available_backends

.. autofunction:: torch_sla.get_backend_methods

.. autofunction:: torch_sla.select_backend

.. autofunction:: torch_sla.select_method

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

