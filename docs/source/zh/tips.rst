使用技巧 (Tips)
===============

过了 :doc:`快速开始 <introduction>` 之后,关于如何选择后端、方法和精度的
实用指引。直接法 vs 迭代法取舍背后的推理在 :doc:`backends`;实测数据在
:doc:`benchmarks`。

选择后端和方法
--------------

- 用迭代法求解器时,**float64 收敛更可靠**;float32 在病态 PDE 矩阵上可能
  卡住。
- 对 **SPD 矩阵**,``cholesky`` 大约比 ``lu`` 快一倍 —— 让
  ``matrix_type="auto"`` 检测对称性 / 正定性并自动选用它。
- 在 **CPU** 上,``scipy+lu`` 是默认,且能给到机器精度。
- 在 **NVIDIA** 上,对约 200 万 DOF 以下的问题,``cudss+cholesky`` 是最快的
  直接求解器。
- 对**更大的问题**,``pytorch+cg`` 是内存高效之选,也是在单 GPU 上达到
  169M DOF 的那个方法。
- 超出单 GPU 时,:class:`~torch_sla.DSparseTensor` 会把矩阵分区到多个设备上。
- 要做**可移植的 GPU 直接求解** —— 包括没有 cuDSS 的 AMD ROCm —— 用
  ``strumpack``(CPU/CUDA/ROCm 上的多波前 LU)。

跨求解复用计算
--------------

- 对**用同一矩阵的重复求解**,用 ``A.lu()`` 分解一次,并对每个右端项复用
  那个 :class:`~torch_sla.LUFactorization`。
- 要在许多次求解上套用同一组默认设置,把它们包进一个
  :class:`~torch_sla.SolverConfig` 作用域(见 :doc:`introduction` 的
  *Configuring solves* 一节),而不是反复传同样的 kwargs。
- 迭代求解是朝着一个容差精化的,所以一个好的 ``x0``(例如时间循环中上一步
  的解)能减少迭代次数。

精度 vs 速度
------------

- 迭代法的默认 ``atol ~ 1e-6`` 是个停止点,而非上限 —— 需要更多位数时把
  ``atol`` 收紧到 ``1e-12``,代价是更多迭代。见
  :ref:`直接法 vs 迭代法 <direct-vs-iterative>`。
- 一个好的预处理器(PDE 问题用 ``"amg"``)能在病态系统上把迭代次数减少
  10-100 倍。
