分布式求解扩展性 (Distributed solve scaling)
============================================

本页是分布式线性求解扩展性基准的**交接指南**。它讲解如何在多 GPU 节点
(或多进程 CPU)上启动这套标准脚本、每个记录指标的含义、如何解读生成的图,
以及如何用新的求解器、分区器或问题来扩展这个基准。

该基准位于

.. code-block:: text

   benchmarks/distributed/scaling/distributed_solve_scaling.py

它是分布式求解扩展性的唯一标准入口。脚本会从一个可复现的 Poisson 问题
(:mod:`torch_sla.datasets`)构建一个按行分片的
:class:`~torch_sla.DSparseTensor`,运行统一的分布式 :func:`~torch_sla.solve`,
并记录求解的墙钟时间、相对残差 ``||A x - b|| / ||b||``\ (正确性门槛)、
吞吐量以及并行效率。

它衡量什么
----------

通过 ``--mode`` 选择三种实验模式:

.. list-table::
   :widths: 18 47 35
   :header-rows: 1

   * - 模式
     - 固定/变化的量
     - 理想曲线
   * - ``weak``
     - 固定**每个 rank** 的 DOF;总 DOF 随 world size 增长。
     - 随 rank 增加,求解时间**保持不变**。
   * - ``strong``
     - 固定**总** DOF;world size 增长。
     - rank 每翻一倍,求解时间**减半**\ (线性加速)。
   * - ``throughput``
     - 每秒处理的 DOF 数对 rank 数(由计时求解推导得到)。
     - 吞吐量随 rank **线性增长**。

这三种都是从*同一次*被插桩的求解中推导出来的,因此在给定 world size 下
启动一次,就能记录下该 world size 的全部数据。JSON 和图会**跨 world size 累积**:
对每个 ``--nproc_per_node`` 取值各跑一次脚本,各行会追加到同一个文件里。
``p=1`` 的运行确立了基线,效率和加速比都是相对它来衡量的,所以
**务必先运行** ``--nproc_per_node=1``。

硬件与启动模型
--------------

基准通过 ``torchrun`` 启动,它为每个 ``--nproc_per_node`` 派生一个进程
(*rank*),并设置脚本读取的标准环境变量:

* ``WORLD_SIZE`` —— 跨所有节点的 rank 总数。
* ``RANK`` —— 本进程的全局 rank(``0 .. WORLD_SIZE-1``)。
* ``LOCAL_RANK`` —— 本进程在其节点内的 rank;用于通过
  ``torch.cuda.set_device(LOCAL_RANK)`` 选定 GPU。

后端会自动选择:CUDA 可用时用 **NCCL**\ (每个 GPU 一个 rank),CPU 上用
**gloo**\ (多进程 CPU,适合在单 GPU 或纯 CPU 机器上跑通分布式代码路径)。

单节点、多 GPU
~~~~~~~~~~~~~~

.. code-block:: bash

   # 本地节点上每个 GPU 一个 rank
   torchrun --standalone --nproc_per_node=4 \
       benchmarks/distributed/scaling/distributed_solve_scaling.py \
       --mode weak --dof-per-rank 100000

``--standalone`` 会在本地一个空闲端口上做单节点 rendezvous;无需配置任何地址。

多节点
~~~~~~

.. code-block:: bash

   # 在每个节点上执行(HEAD_NODE_IP 对所有节点可达,--nnodes/--rdzv-id 一致):
   torchrun \
       --nnodes=2 --nproc_per_node=4 \
       --rdzv-id=sla-scaling --rdzv-backend=c10d \
       --rdzv-endpoint=HEAD_NODE_IP:29500 \
       benchmarks/distributed/scaling/distributed_solve_scaling.py \
       --mode weak --dof-per-rank 100000

在多节点集群上值得了解的 NCCL 环境变量:

.. code-block:: bash

   export NCCL_DEBUG=INFO          # 打印 NCCL 选择的传输方式
   export NCCL_SOCKET_IFNAME=eth0  # 当自动探测选错网卡时,手动指定 NIC
   export NCCL_IB_DISABLE=0        # 保持 InfiniBand 开启(置 1 可强制走 TCP)

可直接复制粘贴的启动命令
------------------------

先跑 ``p=1`` 基线,再跑更大的 world size。JSON 和图会累积,所以只需换一个
``--nproc_per_node`` 重新启动即可。

**弱扩展 (weak scaling)**\ (固定每 rank 的 DOF):

.. code-block:: bash

   for P in 1 2 4 8; do
     torchrun --standalone --nproc_per_node=$P \
       benchmarks/distributed/scaling/distributed_solve_scaling.py \
       --mode weak --dof-per-rank 100000 --method cg --partitioner simple
   done

**强扩展 (strong scaling)**\ (固定总 DOF):

.. code-block:: bash

   for P in 1 2 4 8; do
     torchrun --standalone --nproc_per_node=$P \
       benchmarks/distributed/scaling/distributed_solve_scaling.py \
       --mode strong --total-dof 4000000 --method cg --partitioner simple
   done

**吞吐量 (throughput)** 曲线(复用弱扩展的数据行;或单独跑专门的点):

.. code-block:: bash

   for P in 1 2 4 8; do
     torchrun --standalone --nproc_per_node=$P \
       benchmarks/distributed/scaling/distributed_solve_scaling.py \
       --mode throughput --dof-per-rank 100000
   done

随时渲染已累积的图 + 表格(无需 ``torchrun``):

.. code-block:: bash

   python benchmarks/distributed/scaling/distributed_solve_scaling.py --plot-only

输出示例
--------

下面这段是在一台单 iGPU 机器(``macor7``)上做的 **1-rank 与 2-rank
CPU/gloo 冒烟测试的示例输出**,用于确认脚本能端到端跑通、正确性门槛通过
(残差达到机器精度)。这里的数字按设计就很小 —— *真正的多 GPU 扩展性测量
是交接人的工作*;请用 NCCL 多 GPU 运行结果替换它。

.. code-block:: text

   ==============================================================================
   WEAK SCALING
   ==============================================================================
    ranks   DOF(global)   DOF/rank    time(s)    rel.res    thr(DOF/s)  efficiency
        1        10,000     10,000     0.0003   3.05e-13    31,201,532      100.0%
        2        19,881      9,940     0.0008   5.45e-13    26,079,256       42.0%

   ==============================================================================
   STRONG SCALING
   ==============================================================================
    ranks   DOF(global)   DOF/rank    time(s)    rel.res    thr(DOF/s)  efficiency
        1        19,881     19,881     0.0005   5.31e-13    36,180,624      100.0%
        2        19,881      9,940     0.0009   5.45e-13    23,255,295       32.1%

输出文件:

.. code-block:: text

   benchmarks/results/distributed_solve_scaling.json   # 所有数据行(累积)
   assets/benchmarks/distributed_solve_scaling.png     # 弱 / 强 / 吞吐量三联图

.. image:: ../../../assets/benchmarks/distributed_solve_scaling.png
   :width: 100%
   :alt: distributed solve scaling (weak / strong / throughput)

读懂这些指标
------------

.. list-table::
   :widths: 22 78
   :header-rows: 1

   * - 指标
     - 含义
   * - ``world_size``
     - rank 数(NCCL 下 = GPU 数,gloo/CPU 下 = 进程数)。
   * - ``DOF(global)`` / ``DOF/rank``
     - 总自由度,以及本 rank 所拥有的份额。``weak`` 模式下 DOF/rank
       大致固定;``strong`` 模式下 DOF/rank 随 rank 增加而减小。
   * - ``time(s)``
     - 墙钟求解时间,取 ``--repeat`` 次计时求解中的**最优值**,每次求解
       前后都有 barrier +(CUDA)``synchronize``,确保所有 rank 被一起计时。
   * - ``rel.res``
     - 相对残差 ``||A x - b|| / ||b||`` —— **正确性门槛**。仅由公开算子
       计算得到(``D @ x`` 加上对各 rank 所拥有残差平方和的一次 all-reduce,
       绝不做完整 gather)。一个收敛的 SPD CG 求解应落在请求的 ``--rtol``
       附近(远低于 ``1e-4``);残差很大意味着求解**未收敛**。
   * - ``thr(DOF/s)``
     - ``DOF(global) / time`` —— 问题吞吐量。
   * - ``efficiency``
     - 相对 ``p=1`` 基线的并行效率(理想 = 100%):
       弱扩展 ``= T(1)/T(p)``;强扩展 ``= T(1)/(p·T(p))``;
       吞吐量 ``= (thr(p)/p)/thr(1)``。

.. note::

   ``iterations`` 记录为 ``null``。分布式 Krylov 分片求解器
   (:meth:`~torch_sla.DSparseTensor.solve_distributed_shard`)只返回解向量;
   迭代次数没有通过公开 API 暴露。相对残差才是权威的正确性信号。如果以后
   需要逐次迭代的计数,可以从 ``torch_sla/distributed/solve.py``
   (``cg_shard`` 等)里穿出一个迭代计数器,并填入 ``iterations`` 字段。

良好的扩展性长什么样,以及注意事项
----------------------------------

* **弱扩展** —— 求解时间曲线应尽量保持平坦。出现一点向上的缓坡是正常的,
  因为每次 Krylov 迭代要付出一次 ``all_reduce``\ (用于点积)外加一次 halo
  交换,而这部分通信随 rank 数缓慢增长。
* **强扩展** —— 加速比一开始应贴着虚线表示的理想线性曲线,等到每个 rank
  的问题小到无法掩盖通信时就会拐弯。这个拐点正是有用的信号:它告诉你仍能
  扩展的最小 DOF/rank。
* **吞吐量** —— 在问题仍受计算瓶颈约束时应大致随 rank 线性上升,等通信占
  主导后趋于平坦。

交接人应牢记的注意事项:

* **通信量随 halo 大小增长,而非随 DOF 增长。** 每次迭代真正跨网络传输的是
  (a)每次 ``all_reduce`` 的几个标量,以及(b)每次 SpMV 前交换的 halo
  (ghost)项。halo 大小取决于**分区器**:``metis`` 最小化边切(halo 最小、
  扩展性最好),但构建代价更高;``simple``\ (连续块)便宜,但对不规则图可能
  产生很大的 halo;``coordinate``\ (RCB,基于几何)是网格类问题的折中,需要
  节点坐标(这里由 Poisson 网格合成)。
* **CPU/gloo 没有计算可以掩盖通信。** 在单 CPU 机器上,gloo 运行要承担集合
  通信的开销却没有真正的网络或 GPU 并行,因此强扩展效率预期会很差。用
  CPU/gloo 来验证**正确性和代码路径**,用 NCCL 多 GPU 来测量**真实扩展性**。
* **每 GPU 每 DOF 的字节数** —— 在 NCCL 上,关注 halo 字节数与所拥有 DOF
  之比。每 rank 问题越大,halo 的均摊就越好;这也是弱扩展通常看起来比强扩展
  更健康的原因。

如何扩展这个基准
----------------

脚本的结构使得每个维度都只是一处小而独立的改动:

* **加一个求解器 / 方法。** 传入 ``--method bicgstab``\ (或 ``gmres`` /
  ``minres``);凡是 ``solve_distributed_shard`` 接受的都能用。要加一个全新
  的方法,在 ``torch_sla/distributed/solve.py`` 里实现 ``<name>_shard``,
  并接入 ``solve_distributed_shard``;基准本身无需改动。
* **加一个分区器。** 扩展脚本里的 ``_PARTITIONER_ALIASES``,把你的 CLI
  名字映射到 :func:`torch_sla.partition.resolve_partition_ids` 能理解的
  ``(method_string, needs_coords)``。如果它需要坐标,``_grid_coords``
  已经为 Poisson 网格合成好了。
* **加一个问题。** ``build_problem`` 调用
  :func:`torch_sla.datasets.poisson_2d` / ``poisson_3d``。换上任何返回
  :class:`~torch_sla.datasets.SparseProblem` 的生成器(它必须带有
  ``val/row/col/shape`` 和一个 ``rhs``);加一个 ``--problem`` 标志并在
  ``build_problem`` 里分支处理。保留固定的 ``SEED`` 以保证运行可复现。

记录的 JSON schema 是稳定的(每个
``(mode, world_size, dof_per_rank, partitioner, method)`` 一行),因此外部
工具可以直接读取 ``benchmarks/results/distributed_solve_scaling.json``。
