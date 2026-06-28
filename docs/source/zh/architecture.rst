架构 (Architecture)
===================

torch-sla 的类层级与分布式模型,镜像了 PyTorch 自身
``torch.Tensor`` / ``torch.distributed.tensor.DTensor`` 的分工:一个单独的
稀疏“本地数据”类,外加一个薄薄的分布式包装类,在其之上附加 placement + mesh
元数据。本页是每个新特性都应当遵守的设计契约的权威来源。

----

.. _arch-class-hierarchy:

类层级
------

.. list-table::
   :widths: 30 35 35
   :header-rows: 1

   * - 角色
     - PyTorch
     - torch-sla
   * - **本地数据**
     - ``torch.Tensor``
     - :class:`~torch_sla.SparseTensor`
   * - **分布式包装**\ (本地数据 + spec)
     - ``torch.distributed.tensor.DTensor``
     - :class:`~torch_sla.DSparseTensor`
   * - **每个 rank 的本地块**
     - ``DTensor._local_tensor``\ (一个 ``torch.Tensor``)
     - ``DSparseTensor._local_tensor``\ (一个 ``SparseTensor``)
   * - **分布式元数据**
     - ``DTensor._spec``\ (``DTensorSpec``)
     - ``DSparseTensor._spec``\ (``DSparseSpec``)
   * - **分片 placement**
     - ``Shard(dim)``、``Replicate()``、``Partial(op)``
     - :class:`~torch_sla.SparseShard(axis)`、:class:`~torch_sla.Replicated`

关键不变量:``SparseTensor`` **永远**是本地数据。``DSparseTensor`` **永远**是
一个分布式包装,持有某个 rank 的 ``SparseTensor`` 外加一个 spec。不存在
“混合”类。

----

.. _arch-shape-contract:

形状契约:``(*batch, M, N, *block)``
------------------------------------

一个 :class:`~torch_sla.SparseTensor` 总是具有规范形状

.. code-block:: text

   shape = (*batch_shape, M, N, *block_shape)
           └──dense───┘   └sparse└──dense──┘
            leading        2 dims  trailing

两个稀疏维度**永远**是矩阵轴 ``M`` 和 ``N`` —— 它们不能移动。稠密轴分列其
两侧:

* ``batch_shape``\ (左)—— 用于批量 SpMV / 求解的稠密 batch 维。
* ``block_shape``\ (右)—— 用于块稀疏格式(BSR / BCSC)的稠密块维。

如果用户的张量里稀疏轴不在这个位置,``SparseTensor.permute(...)`` 会把它重排
成规范布局。该契约是按位置确定的,而非靠稀疏维元数据,因此每个算法
(matvec、solve、eigsh 等)都知道该去哪里找。

----

.. _arch-placement-vocab:

Placement 词汇表
----------------

一个 ``DSparseSpec`` 携带:

* ``placement``:数据如何分片
* ``mesh``:它分片到哪些设备上
* ``global_shape``:原始的完整张量形状

``placement`` 要么是单个 placement(一维 mesh),要么是一个列表,mesh 的每个
维度对应一个元素(多维 mesh —— 与 DTensor 同一约定)。

.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - 类
     - 轴的种类
     - 适用场景
   * - :class:`~torch_sla.Replicated`
     - --
     - 每个 rank 上都有完整矩阵
   * - ``torch.distributed.tensor.Shard(dim)``
     - 稠密轴(batch 或 block)
     - 每个 rank 拿到一部分 batch;SpMV 无需跨 rank 通信
   * - :class:`~torch_sla.SparseShard(axis)`
     - 稀疏轴(行用 ``axis=len(batch_shape)``,列再 ``+1``)
     - 矩阵的不规则行/列分区;需要 halo 交换或 all-reduce
   * - :class:`~torch_sla.SparseShard(axis)` 配合超图导出的分区
     - 稀疏轴
     - 经 PaToH / Mondriaan 超图切分实现最小通信的 SpMV

便捷构造器 :func:`~torch_sla.row_shard` 和 :func:`~torch_sla.col_shard`
覆盖常见的二维矩阵情形:

.. code-block:: python

   from torch_sla import row_shard, col_shard, SparseShard

   row_shard()              # SparseShard(axis=0), plain (M, N)
   col_shard()              # SparseShard(axis=1)
   row_shard(batch_ndim=2)  # SparseShard(axis=2), for (B1, B2, M, N) tensor

在二维 mesh 上做多轴分片:传一个列表,和 DTensor 完全一样。

.. code-block:: python

   from torch.distributed.tensor import Shard
   from torch_sla import SparseShard

   # 2-D mesh: 4 batch shards × 8 row shards
   mesh = init_device_mesh("cuda", (4, 8))
   placement = [Shard(0),              # mesh dim 0: dense batch dim
                SparseShard(axis=2)]   # mesh dim 1: sparse row axis (batch_ndim=2)

----

.. _arch-matvec-dispatch:

按 placement 分派 matvec
------------------------

``DSparseTensor.__matmul__`` 会根据 placement 分派,以挑选正确的通信模式。
下表每一行都是一条独立的代码路径:

.. list-table::
   :widths: 30 35 35
   :header-rows: 1

   * - Placement
     - matvec 算法
     - 跨 rank 通信
   * - ``Replicated``
     - 本地 ``A @ x``\ (无通信)
     - 无
   * - ``Shard(batch_dim)``
     - 逐 batch 独立 SpMV
     - 无(完全并行)
   * - ``SparseShard(row_axis)``
     - halo 交换 + 本地 SpMV
     - O(halo nnz) 点对点
   * - ``SparseShard(col_axis)``
     - 本地部分 SpMV + ``all_reduce(SUM)``
     - O(M) all-reduce
   * - 二维 placement 列表 ``[SparseShard(M), SparseShard(N)]``
     - 二维 Cannon / SUMMA
     - 大 mesh 下比一维好 O(sqrt)

----

.. _arch-partition-algorithms:

分区算法
--------

如何分布行 / 列本身就是个子问题。下面这些选项给出了各自的评分取舍和
Python 绑定的成熟度:

.. list-table::
   :widths: 18 22 22 38
   :header-rows: 1

   * - 算法
     - 相对 METIS 的质量
     - Python 绑定
     - 最适合
   * - ``simple`` / striped
     - 差很多(无局部性)
     - 无(~10 行代码)
     - 健全性测试、跨 rank 确定性
   * - METIS(当前默认)
     - 基准线
     - ``pymetis`` 稳定
     - 最多约 1 亿节点的图
   * - Hilbert 空间填充曲线
     - 更差,但快约 10-100 倍
     - 纯 Python 或 ``pyhilbert``
     - PDE 网格 / 几何结构
   * - KaHIP
     - 质量 +20%
     - ``kahip-python`` 难搞;子进程 shell 调用也可
     - 最多约 10 亿节点的图
   * - Mt-METIS
     - 质量相同,快 4-16 倍
     - 无 Python 绑定;用 ctypes 调 C
     - CPU 核心多的中等规模用户
   * - PaToH(超图)
     - **SpMV 通信最小** —— 理论最优
     - ``pypatoh`` 半维护状态
     - 专门面向稀疏 matvec
   * - Mondriaan
     - 与 PaToH 相近,二维专用
     - 命令行封装
     - 专门面向稀疏矩阵
   * - ParMETIS
     - METIS 级质量,分布式
     - 无 Python 绑定(仅 MPI C)
     - 真正的 HPC 集群
   * - 基于 GNN 的学习式
     - 研究级
     - 自行实现
     - 超大 / 流式图(>10 亿边)

torch-sla 的 :meth:`~torch_sla.SparseTensor.partition_for_rank` 通过
``partition_method`` 关键字参数暴露分区。如今它支持 ``simple``、``metis``、
``rcb``、``slicing``。``hilbert`` 和 ``patoh`` 作为后续跟进项被跟踪。

----

为什么这与 DTensor 对齐
-----------------------

上面每个设计选择都 1:1 对应到一个相应的 DTensor 决策:

* ``SparseTensor`` ≅ ``torch.Tensor`` —— 同样的“本地数据”角色。
* ``DSparseTensor`` ≅ ``DTensor`` —— 同样的“(本地 + spec)”结构。
* ``SparseShard(axis)`` ≅ ``Shard(dim)`` —— 每个分片方向一个参数化的
  placement,而非各自独立的类。
* mesh 维度上的 placement *列表* ≅ DTensor 的多轴分片。
* spec 中 ``mesh`` + ``global_shape`` 的分离 ≅ DTensorSpec。

通过与 DTensor 保持平行,torch-sla 能干净地与 PyTorch 的分布式生态
(FSDP、TP、DCP)组合 —— ``DSparseTensor.matvec`` 给出的稀疏向量结果,本身
就是一个带有正确 placement 的 ``DTensor``,可以直接喂给下游的 FSDP 模块。
