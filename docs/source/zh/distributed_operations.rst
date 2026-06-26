分布式算子（多节点）— 交接指南
================================

本页是 ``examples/distributed/`` 中可运行分布式算子示例的**交接指南**。
每个脚本把 :class:`~torch_sla.DSparseTensor` 按行分片到设备网格上，运行一个
分布式算子，并用单进程参考结果做正确性校验，便于他人在真实的多节点、多 GPU
集群上验证算子确实可用。

所有示例都是**设备感知**的：当 ``torch.cuda.is_available()`` 为真时选择
``nccl`` + CUDA（按 ``LOCAL_RANK`` 用 ``torch.cuda.set_device(LOCAL_RANK)``
绑定单卡，并在 ``"cuda"`` 上构建网格），否则回退到 ``gloo`` + CPU，因此同一份
脚本在笔记本和 GPU 集群上都能跑。

启动方式
--------

``torchrun`` 为每个 ``--nproc_per_node`` 启动一个进程（rank），并设置
``WORLD_SIZE`` / ``RANK`` / ``LOCAL_RANK``。脚本读取 ``LOCAL_RANK`` 绑定 GPU。

**单节点**（``--standalone``）：

.. code-block:: bash

   cd examples/distributed
   torchrun --standalone --nproc_per_node=4 distributed_matvec.py

**多节点**（``c10d`` 汇合；在**每个**节点上运行**相同**命令，
``HEAD_NODE_IP:29500`` 需对所有节点可达）：

.. code-block:: bash

   export NCCL_DEBUG=INFO
   export NCCL_SOCKET_IFNAME=eth0
   export NCCL_IB_DISABLE=0

   # 在每个节点上（例如 2 节点 × 4 GPU = world size 8）：
   torchrun \
       --nnodes=2 --nproc_per_node=4 \
       --rdzv-id=sla --rdzv-backend=c10d \
       --rdzv-endpoint=HEAD_NODE_IP:29500 \
       distributed_matvec.py

``examples/distributed/launch.sh`` 封装了两种形式：设置 ``RDZV_ENDPOINT``
（以及可选的 ``NNODES`` / ``RDZV_ID``）即可从 ``--standalone`` 切换到多节点
``--rdzv-*`` 形式，例如
``RDZV_ENDPOINT=HEAD_NODE_IP:29500 NNODES=2 ./launch.sh all 4``。

五个算子
--------

每个算子均给出：脚本路径、单节点命令、多节点命令、以及打印并断言的正确性校验。

* **matvec** — 分布式稀疏矩阵向量积 ``y = A @ x``（自动 halo 交换）。
  脚本 ``examples/distributed/distributed_matvec.py``；校验：与单进程
  ``A @ x`` 的 ``max|y_dist − y_ref|`` 断言 ``< 1e-12``。
* **solve** — 分布式 Krylov 线性求解 ``A x = b``（CG + Jacobi 预条件）。
  脚本 ``examples/distributed/distributed_solve.py``；校验：相对残差
  ``‖b − A x‖ / ‖b‖ < 1e-8`` 且与 scipy CG 的相对差 ``< 1e-6``。
* **eigsh** — 分布式对称特征求解（LOBPCG，最小若干特征对）。
  脚本 ``examples/distributed/distributed_eigsh.py``；校验：相对
  ``scipy.sparse.linalg.eigsh`` 的每个特征值相对误差 ``< 1e-3``。
* **connected_components** — 行分片邻接矩阵上的分布式连通分量
  （标签传播 + 边界标签 halo 交换）。脚本
  ``examples/distributed/distributed_connected_components.py``；校验：
  全局分量数等于期望值且等于 ``scipy.sparse.csgraph.connected_components``
  的结果，且诱导的节点划分与 scipy 一致（与编号无关）。
* **nonlinear_solve** — 半线性系统 ``F(u) = A u − λ exp(u) = 0`` 的分布式
  Newton 求解（Bratu 型，对角 Jacobian 移位，Newton 步用分布式 GMRES）。
  脚本 ``examples/distributed/distributed_nonlinear_solve.py``；校验：全局
  残差 ``‖F(u*)‖ < 1e-8`` 且与单进程 Newton 的相对差 ``< 1e-6``。

每个算子的单节点命令为
``torchrun --standalone --nproc_per_node=4 <脚本>``，多节点命令为
``torchrun --nnodes=2 --nproc_per_node=4 --rdzv-id=sla --rdzv-backend=c10d
--rdzv-endpoint=HEAD_NODE_IP:29500 <脚本>``。

收集结果
--------

每个脚本在每个 rank 上打印 ``[rank R] ...``，并在 rank 0 上打印最终结论。
分布式结果以**owned 切片**（每个 rank 一个行块）形式存在；要查看或保存完整
向量，可用 :func:`torch_sla.distributed.gather_owned_to_global` 在每个 rank
上拼装：

.. code-block:: python

   from torch_sla.distributed import gather_owned_to_global

   owned = D._spec.placement.partition.owned_nodes
   full = gather_owned_to_global(owned, owned_values, N_global)

matvec / solve 示例则用 ``DTensor`` 封装（``D.scatter(x_global)`` 与
``.full_tensor()``）在全局向量和 Shard(0) 布局之间转换。多节点运行时，把每个
节点的 ``torchrun`` 输出重定向到各自文件——rank 0 的文件包含断言与最终的
``converged`` 行。

.. seealso::

   * :doc:`distributed_scaling` — 性能/扩展性交接指南。
   * ``examples/distributed/README.md`` — 相同的启动矩阵与 API 速览。
