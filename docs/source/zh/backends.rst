后端与能力矩阵 (Backends and Capability Matrix)
===============================================

torch-sla 会把每次 :func:`~torch_sla.solve` 调用分派到若干后端之一。你可以
通过 ``backend="..."`` 显式指定后端,也可以让 ``backend="auto"`` 根据设备、
dtype、问题规模以及已安装的可选依赖来选择。

当前的后端阵容及各自支持的能力:

.. list-table:: **能力矩阵**
   :widths: 11 8 8 8 10 12 10 10 10 10
   :header-rows: 1
   :class: capability-table

   * - 后端
     - CPU
     - CUDA
     - ROCm
     - 直接法
     - 迭代法
     - 复数
     - 批量
     - 分布式
     - Autograd
   * - ``scipy``
     - ✔
     - --
     - --
     - LU / UMFPACK
     - CG, BiCGStab, GMRES
     - ✔
     - 经由批量辅助函数
     - --
     - ✔
   * - ``pytorch``
     - ✔
     - ✔
     - ✔
     - --
     - CG, BiCGStab, GMRES, MINRES, LSQR, LSMR (+ PCG / PBiCGStab)
     - ✔
     - ✔
     - 经由 ``DSparseTensor``
     - ✔
   * - ``strumpack``
     - ✔
     - ✔
     - ✔
     - LU (multifrontal)
     - --
     - ✔
     - --
     - --
     - ✔
   * - ``cudss``
     - --
     - ✔
     - --
     - LU / Cholesky / LDL\ :sup:`T` / LDL\ :sup:`H`
     - --
     - ✔
     - --
     - --
     - ✔
   * - ``pyamg``
     - ✔
     - ✔ (仅 V-cycle)
     - ✔ (仅 V-cycle)
     - --
     - Ruge-Stuben AMG, Smoothed Aggregation
     - --
     - --
     - --
     - ✔
   * - ``amgx``
     - --
     - ✔
     - --
     - --
     - AMG, PCG, PBiCGStab, FGMRES (NVIDIA AmgX)
     - --
     - --
     - --
     - ✔

.. note::

   **全部六个后端均经过正确性验证。** 每个后端都对照参考解做了校验,相对残差
   ‖Ax − b‖ / ‖b‖ 达到或接近机器精度。两条 GPU 直接法路径稳稳落在该范围内 ——
   在验证矩阵上实测 ``strumpack`` ≈ ``3e-13``、``amgx`` ≈ ``5.6e-13``。

.. note::

   ``cudss`` 和 ``pyamg`` 可从 PyPI 安装,但两个**原生编译**后端 ——
   ``strumpack``(``torch-strumpack``)和 ``amgx``(``torch-amgx``)—— 是以
   **GitHub Releases 上的预编译 wheel** 形式发布的(不在 PyPI),而且每个
   wheel 的 ABI 都绑定到特定的 CUDA *和* PyTorch 版本。wheel 选择规则及一个
   具体的 ``pip install --no-deps`` 示例见安装指南中的
   :ref:`prebuilt-native-wheels`。

----

STRUMPACK 后端
--------------

``backend="strumpack"`` 是一个**可移植的多波前稀疏直接求解器**。与 cuDSS
(仅支持 NVIDIA CUDA)不同,STRUMPACK 用同一套 API 就能在 **CPU、CUDA 和
AMD ROCm** 上运行,支持实数和复数矩阵,并提供多波前 LU 分解。它是完全可微的:
梯度通过伴随(A\ :sup:`H`)求解流动,因此能像其他后端一样接入 autograd 流程。

实践中,在 cuDSS 无法触及的硬件上要做 GPU **直接**求解时,STRUMPACK 就是
答案 —— 其中最重要的是 AMD ROCm GPU,那里没有 cuDSS。它需要可选的
``torch-strumpack`` 包,该包以 **GitHub Releases 上的预编译 wheel** 形式发布
(不在 PyPI;见 :ref:`prebuilt-native-wheels`),覆盖 Linux cpu / cuda / rocm
以及 macOS arm64。**Windows(CPU)也受支持** —— STRUMPACK 可在 Windows 上用
conda-forge 的 ``clang-cl``(C/C++)+ ``flang``(Fortran)构建,并链接到
MSVC 编译的 PyTorch(纯净环境下求解的相对残差约 1.7e-16);经 CI 构建的
预编译 Windows wheel 正在添加中::

    # 从下面这里获取匹配的 wheel
    #   https://github.com/sparsexlab/torch-strumpack/releases
    pip install --no-deps <release-url>/torch_strumpack-...-linux_x86_64.whl

----

平台可用性
----------

直接法后端会绑定到各厂商的库;下表记录了每个后端如今能在哪些操作系统上构建。

.. list-table::
   :widths: 18 14 14 14 40
   :header-rows: 1

   * - 后端
     - Linux
     - Windows
     - macOS
     - 备注
   * - ``scipy``
     - ✔
     - ✔
     - ✔
     - 纯 SciPy;UMFPACK 可经 ``scikit-umfpack`` 选装。
   * - ``pytorch``
     - ✔
     - ✔
     - ✔
     - PyTorch 原生;当 ``torch.cuda.is_available()`` 时 CUDA / ROCm 路径
       生效(ROCm 版的 torch 会自报为 ``cuda``)。
   * - ``strumpack``
     - ✔
     - ✔ (CPU)
     - ✔ (arm64)
     - 多波前稀疏直接求解器(多波前 LU,实数 + 复数)。Linux 上的
       CPU / CUDA / ROCm 以及 macOS arm64 经由 ``torch-strumpack``
       (GitHub-Release wheel,:ref:`prebuilt-native-wheels`)。
       **Windows(CPU)受支持** —— 用 ``clang-cl`` + ``flang`` 构建;
       经 CI 的预编译 Windows wheel 待发布。
   * - ``cudss``
     - ✔
     - ✔
     - --
     - 需要 ``nvmath-python[cu12]`` + NVIDIA CUDA。NVIDIA 不支持 macOS。
   * - ``pyamg``
     - ✔
     - ✔
     - ✔
     - setup 阶段经可选依赖 ``pyamg``(``pip install pyamg``)在 CPU 上
       运行;V-cycle 通过 ``torch.sparse`` 分派,因此 cycle 本身会在矩阵
       所在的任意设备上运行。**跨平台 AMG**:macOS 得到 CPU AMG,
       CUDA 机器得到 GPU V-cycle。

----

``backend="auto"`` 在何时选谁
--------------------------------

* **NVIDIA CUDA 张量**:先试 ``cudss``(最好的直接求解器)->
  ``pytorch``(迭代法兜底)。
* **AMD ROCm 张量**:cuDSS **仅限 NVIDIA**,在这里永不运行,因此 auto 路径
  使用 ``pytorch``(迭代法),需要直接求解时则用 ``strumpack``
  (ROCm 上可移植的多波前直接求解器)。
* **CPU 张量,小 / 中规模**:优先 ``scipy`` LU。
* **CPU 张量,大规模或重复求解**:``pytorch`` CG / BiCGStab 能让内存占用
  保持平稳。

当你需要精确控制时,随时用 ``backend="..."`` 覆盖(例如用 ``backend="cudss"``
强制在 NVIDIA 上做直接 GPU 求解,或在没有 cuDSS 的 AMD ROCm 上用
``backend="strumpack"`` 做直接 GPU 求解)。

----

.. _direct-vs-iterative:

直接法 vs 迭代法:精度与复杂度
------------------------------

精度表中,直接法后端标注 ``~1e-14``,迭代法标注 ``~1e-6``。这道差距是结构性
的,而不是 bug:

* **直接法**求解器对矩阵做分解(``LU`` / ``Cholesky`` / ``LDL``)再回代。结果
  *在浮点舍入意义下是精确的* —— 对一个良态的 ``float64`` 系统,相对残差落在
  机器 epsilon 附近(``~1e-14``..``1e-16``)。没有收敛旋钮;你一次性付出分解
  代价,换来一个完全精确的答案。
* **迭代法**求解器(CG、BiCGStab、GMRES 等)不断精化一个猜测,直到残差
  ``‖Ax − b‖ / ‖b‖`` 降到你设定的*容差*以下(``atol`` / ``rtol``,默认
  ``~1e-6``)。它们在容差处停止,所以答案只精确到你要求的程度。把 ``atol``
  收紧到 ``1e-12``,残差就跟着下去 —— 代价是更多迭代。``A`` 的病态程度
  (其条件数)决定了每一位精度要花多少次迭代。

所以迭代法的 ``~1e-6`` 是个*默认停止点*,而不是精度上限。正是这种取舍让迭代
路径具备可扩展性:它从不形成分解所产生的稠密填充。

.. list-table:: 直接法 vs 迭代法的代价(``n`` 个未知数,``nnz`` 个非零元,``m`` 次迭代)
   :widths: 18 27 27 28
   :header-rows: 1

   * - 求解器
     - 时间
     - 空间
     - 精度
   * - 直接法 (LU / Cholesky)
     - :math:`O(n^{1.5})`(二维)到 :math:`O(n^{2})`(三维)
     - :math:`O(n\log n)` 到 :math:`O(n^{4/3})` 的填充
     - 精确到舍入(``~1e-14``)
   * - 迭代法 (CG / GMRES)
     - :math:`O(m\cdot nnz)`
     - :math:`O(n + nnz)`
     - 受容差限制(``atol``,默认 ``~1e-6``)

对一个稀疏 PDE 矩阵,``nnz = O(n)``,所以一次迭代扫描在 :math:`O(n)` 内存下
花费 :math:`O(m\,n)` 时间,而直接法分解的填充才是超过几百万未知数后耗尽内存
的元凶(见 :doc:`benchmarks`)。当你需要最后几位精度、或有很多右端项可复用
同一次分解时,选直接法;当矩阵很大且 ``~1e-6`` 已经够用时,选迭代法。

----

把它们组合起来
--------------

能力矩阵直接映射到 :func:`~torch_sla.solve` 的参数:任何单元格为 ✔ 的组合都
受支持::

    import torch
    from torch_sla import solve, PreconditionerConfig

    A_csr = ...                          # any accepted matrix format
    b = torch.randn(n)

    # Direct GPU solve, automatic Cholesky/LDL^H selection
    x = solve(A_csr, b, backend="cudss", matrix_type="auto")

    # CPU iterative CG with a tuned SSOR preconditioner
    x = solve(A_csr, b,
              backend="pytorch", method="cg",
              preconditioner=PreconditionerConfig(kind="ssor", omega=1.2),
              atol=1e-10, maxiter=5_000)

    # CPU iterative CG with a real multi-level AMG preconditioner
    # (uses PyAMG when installed, falls back to the lightweight
    # 2-level stub otherwise). Reduces the iteration count by 10-100x
    # on ill-conditioned PDE problems.
    x = solve(A_csr, b,
              backend="pytorch", method="cg",
              preconditioner="amg",  # or PreconditionerConfig(kind="amg", ...)
              atol=1e-10, maxiter=200)

    # Diagnostic return -- iteration count + residual
    x, info = solve(A_csr, b, return_info=True)
    print(info.iter_count, info.residual, info.converged)

----

未来后端(路线图)
------------------

下一波后端将用跨平台 AMG 预处理和高端 GPU AMG 来扩充上面的表格:

.. list-table::
   :widths: 18 18 28 36
   :header-rows: 1

   * - 后端
     - 状态
     - 能力
     - 备注
   * - ``pyamg``
     - **已可用**(本次发布)
     - CPU AMG setup + 跨设备 V-cycle
     - 已经在用。见上文。独立求解器 +
       :class:`~torch_sla.backends.pyamg_backend.PyAMGHierarchy` 用于
       复用预处理器。
   * - ``amgx``
     - **已可用**(本次发布)
     - CUDA AMG + Krylov(Nvidia AmgX)
     - 仅 Linux + Windows。需要 NVIDIA GPU(含 cu12.8 上的 Blackwell
       ``sm_120``)。从
       `torch-amgx Releases <https://github.com/sparsexlab/torch-amgx/releases>`_
       安装预编译 wheel(不在 PyPI;见 :ref:`prebuilt-native-wheels`)。
   * - ``petsc``
     - 调研中
     - CPU/GPU 直接法 + 迭代法,分布式(PETSc/hypre BoomerAMG)
     - Linux + macOS 容易;Windows 经 WSL。
