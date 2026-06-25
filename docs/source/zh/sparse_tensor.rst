SparseTensor
============

:class:`~torch_sla.SparseTensor` 是单进程的稀疏矩阵。它以 COO 形式存储矩阵
(``val``、``row``、``col`` 加上一个 ``shape``),可选地带一个前导 batch 维,
并携带完整的运算词汇表:线性与非线性求解、特征分解、SVD、矩阵—向量乘积、
标量/结构性查询、图分析以及可视化。每次求解都通过 ``torch.autograd`` 可微,
并分派到最适合该设备的后端(CPU 上的 SciPy / PyTorch 原生,GPU 上的
cuDSS / STRUMPACK)。

关于跨进程分片的矩阵,见 :doc:`dsparse_tensor`。关于逐运算参考,见
:doc:`operations`。

构造
----

一个 :class:`~torch_sla.SparseTensor` 可以直接从 COO 三元组构建,也可以通过
若干便捷构造器:

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

转换回去:

.. code-block:: python

   dense = A.to_dense()          # torch.Tensor
   crow, col, val = A.to_csr()   # CSR triplet
   t = A.to_torch_sparse()       # torch.sparse_coo_tensor

一个前导 batch 维(形状 ``[B, n, n]``)能让每个运算在一次调用中作用于一摞
同模式矩阵;见 :ref:`op-solve-batch`。对于模式*不同*的矩阵,使用
:class:`~torch_sla.SparseTensorList`。

运算目录
--------

每个运算都链接到它在 :doc:`operations` 中的详细条目,以及它的 API 对象。标记
**(grad)** 的运算通过伴随法传播梯度。

线性求解
~~~~~~~~

.. list-table::
   :widths: 22 18 60
   :header-rows: 1

   * - 运算
     - API
     - 描述
   * - :ref:`solve <op-solve>` **(grad)**
     - :meth:`~torch_sla.SparseTensor.solve`
     - 求解 :math:`Ax = b`;自动选择直接法或迭代法后端。
   * - :ref:`solve_batch <op-solve-batch>`
     - :meth:`~torch_sla.SparseTensor.solve_batch`
     - 共享一个稀疏模式的多个右端项 / 取值集合。
   * - :ref:`lu <op-lu>`
     - :meth:`~torch_sla.SparseTensor.lu`
     - 缓存一个 LU 分解以供重复求解。

非线性
~~~~~~

.. list-table::
   :widths: 22 18 60
   :header-rows: 1

   * - 运算
     - API
     - 描述
   * - :ref:`nonlinear_solve <op-nonlinear-solve>` **(grad)**
     - :meth:`~torch_sla.SparseTensor.nonlinear_solve`
     - 用 Newton / Picard / Anderson 求解 :math:`F(u, \theta) = 0`。

特征值 / 谱
~~~~~~~~~~~

.. list-table::
   :widths: 22 18 60
   :header-rows: 1

   * - 运算
     - API
     - 描述
   * - :ref:`eigsh <op-eigsh>` **(grad)**
     - :meth:`~torch_sla.SparseTensor.eigsh`
     - 对称/Hermitian 矩阵的前 k 个特征对。
   * - :ref:`eigs <op-eigsh>`
     - :meth:`~torch_sla.SparseTensor.eigs`
     - 一般矩阵的前 k 个特征对。
   * - :ref:`svd <op-svd>` **(grad)**
     - :meth:`~torch_sla.SparseTensor.svd`
     - 截断的秩-k 奇异值分解。

矩阵—向量
~~~~~~~~~

.. list-table::
   :widths: 22 18 60
   :header-rows: 1

   * - 运算
     - API
     - 描述
   * - :ref:`matvec / @ <op-matvec>`
     - :meth:`~torch_sla.SparseTensor.__matmul__`
     - 稀疏矩阵—向量 / 矩阵—矩阵乘积(SpMV)。

标量 / 结构性
~~~~~~~~~~~~~

.. list-table::
   :widths: 22 18 60
   :header-rows: 1

   * - 运算
     - API
     - 描述
   * - :ref:`det <op-det>` **(grad)**
     - :meth:`~torch_sla.SparseTensor.det`
     - 通过稀疏 LU 求行列式。
   * - :ref:`logdet <op-logdet>` **(grad)**
     - :meth:`~torch_sla.SparseTensor.logdet`
     - 对数行列式(对大矩阵数值稳定)。
   * - :ref:`norm <op-norm>` **(grad)**
     - :meth:`~torch_sla.SparseTensor.norm`
     - Frobenius / 1- / 2- 范数。
   * - :ref:`condition_number <op-condition-number>`
     - :meth:`~torch_sla.SparseTensor.condition_number`
     - 比值 :math:`\sigma_{\max}/\sigma_{\min}`。
   * - :ref:`is_symmetric / is_positive_definite <op-predicates>`
     - :meth:`~torch_sla.SparseTensor.is_symmetric`
     - 用于求解器选择的结构性判定。

图
~~

.. list-table::
   :widths: 22 18 60
   :header-rows: 1

   * - 运算
     - API
     - 描述
   * - :ref:`connected_components <op-connected-components>`
     - :meth:`~torch_sla.SparseTensor.connected_components`
     - 标注邻接模式的连通分量。

可视化
~~~~~~

.. list-table::
   :widths: 22 18 60
   :header-rows: 1

   * - 运算
     - API
     - 描述
   * - :ref:`spy <op-spy>`
     - :meth:`~torch_sla.SparseTensor.spy`
     - 把稀疏模式绘制成一张 matplotlib 图。

I/O 与归约
~~~~~~~~~~

除了上面这些主打运算,:class:`~torch_sla.SparseTensor` 还提供保存/加载
(safetensors 和 Matrix Market)、逐元素数学(``abs``、``sqrt``、``exp``、
``log`` 等)以及归约(``sum``、``mean``、``max``、``min``)。这些在
:doc:`API 参考 <torch_sla>` 中有文档。
