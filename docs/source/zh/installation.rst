安装
====

.. raw:: html

   <ul class="feature-list">
     <li><span class="gradient-text">PyPI</span>: <code>pip install torch-sla</code> —— 最简单的安装方式</li>
     <li><span class="gradient-text">GitHub</span>: 克隆并以开发模式安装</li>
     <li><span class="gradient-text">可选后端</span>: cuDSS、STRUMPACK、PyAMG、AmgX 以获得更强性能</li>
   </ul>

----

使用 pip
--------

安装最新发布版本：

.. code-block:: bash

    pip install torch-sla

或从 GitHub 安装最新开发版本：

.. code-block:: bash

    pip install git+https://github.com/walkerchi/torch-sla.git

可选依赖
--------

核心安装会自带 ``torch``、``numpy``、``scipy`` 和 ``ninja`` —— 足以开箱即用地
运行 CPU 求解器。GPU 用户可以按需挑选所需的后端：

.. code-block:: bash

    # NVIDIA GPU 用户（CUDA 12+，Linux/Windows）：
    pip install torch-sla[cudss]    # + cuDSS 直接求解器（~10K-2M DOF 最快，仅 NVIDIA）

    # CPU 用户（所有平台，包括 macOS）：
    pip install torch-sla[pyamg]    # + PyAMG（CPU 端 AMG setup + 设备端 V-cycle）

    # STRUMPACK 直接求解器（CPU/CUDA/ROCm）和 AmgX（NVIDIA GPU AMG/Krylov）
    # 不是 pip extras —— 它们是 GitHub Releases 上的预编译 wheel（见下方）：
    #   https://github.com/sparsexlab/torch-strumpack/releases
    #   https://github.com/sparsexlab/torch-amgx/releases

    # 完整安装，包含所有 *PyPI 可安装的* 运行时后端
    # （不含 dev/docs；原生的 torch-amgx / torch-strumpack
    #  后端是 GitHub-Release wheel —— 见下方“预编译 wheel”）
    pip install torch-sla[all]

    # 开发工具（pytest、black、isort、mypy）
    pip install torch-sla[dev]

    # 文档工具（sphinx、furo）
    pip install torch-sla[docs]

.. note::

   两个**原生（编译型）后端** —— ``torch-amgx``（NVIDIA AmgX）与
   ``torch-strumpack``（STRUMPACK）—— **不在 PyPI 上**，也**不是 pip extras**。
   特别地，``torch-sla[all]`` **不会**把它们装进来。它们是 PyTorch 的
   C++/CUDA 扩展，以预编译 wheel 形式发布在 **GitHub Releases**。选 wheel 时
   必须同时匹配你的 CUDA 版本*和*已安装的 PyTorch 版本 —— 见下方
   :ref:`prebuilt-native-wheels-zh`。

.. _prebuilt-native-wheels-zh:
.. _prebuilt-native-wheels:

预编译 wheel：torch-amgx 与 torch-strumpack
-------------------------------------------

``torch-amgx`` 与 ``torch-strumpack`` 是编译型 PyTorch 扩展，因此**没有 PyPI
包**。请从 GitHub Releases 下载与你环境匹配的 wheel：

* **torch-amgx** —— https://github.com/sparsexlab/torch-amgx/releases ——
  Linux + Windows，Python 3.10-3.13，CUDA 12.4 / 12.6 / 12.8。每个 wheel
  的文件名都带有按 CUDA 区分的构建标签：``0_cu124`` / ``0_cu126`` /
  ``0_cu128``。cu12.8 的 wheel 含 NVIDIA Blackwell（``sm_100`` /
  ``sm_120``）。
* **torch-strumpack** —— https://github.com/sparsexlab/torch-strumpack/releases
  —— Linux（cpu / cuda / rocm）+ macOS arm64，Python 3.10-3.13。**支持
  Windows（CPU）** —— STRUMPACK 可在 Windows 上用 conda-forge 的
  ``clang-cl``\ （C/C++）+ ``flang``\ （Fortran）编译，链接 MSVC 构建的 PyTorch
  （干净环境下求解相对残差约 1.7e-16）。CI 预编译的 Windows wheel 正在添加中。

.. warning::

   **ABI 兼容性 —— 下载前必读。** 这些 wheel 是 torch C++ 扩展，因此每一个
   都同时绑定 **(a)** 编译时所针对的 CUDA 版本，*和* **(b)** 具体的 PyTorch
   版本。你必须：

   1. 选择 ``0_cuXXX`` 标签与 ``torch.version.cuda`` 一致的 wheel
      （用 ``python -c "import torch; print(torch.version.cuda)"`` 检查）。
   2. 安装与该 wheel 编译时一致的 PyTorch 版本（见 release notes）。

   不匹配会在 **import 时**\ （而非安装时）报错：Windows 上是
   ``DLL load failed ... procedure not found``，Linux 上是 ``undefined symbol``。

用 ``--no-deps`` 直接按 URL 安装 release wheel（这样 pip 不会去解析一个不存在
的 PyPI 包）：

.. code-block:: bash

   # 示例：CUDA 12.6 + CPython 3.13 的 torch-amgx（请用 Releases 页面上与你
   # torch / CUDA / Python 匹配的真实 asset URL 替换）
   pip install --no-deps \
     https://github.com/sparsexlab/torch-amgx/releases/download/<tag>/torch_amgx-<ver>-0_cu126-cp313-cp313-linux_x86_64.whl

   # torch-strumpack（页面上有 cuda / rocm / cpu / macos-arm64 各版本）
   pip install --no-deps \
     https://github.com/sparsexlab/torch-strumpack/releases/download/<tag>/torch_strumpack-<ver>-cp313-cp313-linux_x86_64.whl

安装后用 ``torch_sla.show_backends()`` 确认后端已加载。

.. admonition:: 验证你的环境

   安装后，你可以查看本机上有哪些后端可用：

   .. code-block:: python

      import torch_sla
      torch_sla.show_backends()

后端要求
--------

.. list-table::
   :widths: 20 30 50
   :header-rows: 1

   * - 后端
     - 安装方式
     - 说明
   * - ``scipy``
     - ``pip install scipy``
     - 默认，始终可用
   * - ``pytorch``
     - 随 PyTorch 一起提供
     - 原生 CG/BiCGStab 求解器。与设备无关 —— 可在 CPU / CUDA / ROCm 上运行。
   * - ``strumpack``
     - 来自 `torch-strumpack Releases
       <https://github.com/sparsexlab/torch-strumpack/releases>`_
       的预编译 wheel（见 :ref:`prebuilt-native-wheels-zh`）
     - 可移植的多波前稀疏**直接**求解器（多波前 LU，实数 + 复数，完整
       autograd）。可在 **CPU / CUDA / ROCm**（Linux）+ macOS arm64 上运行
       —— 在 cuDSS 不可用的 AMD ROCm 上作为直接求解路径。**不在 PyPI 上。**
       支持 Windows（CPU）—— STRUMPACK 用 ``clang-cl`` + ``flang`` 编译；
       CI 预编译的 Windows wheel 正在添加中。
   * - ``cudss``
     - ``pip install nvmath-python[cu12]``
     - 最适合中等规模的 GPU 问题（10K-2M DOF）。仅 NVIDIA CUDA。
   * - ``amgx``
     - 来自 `torch-amgx Releases
       <https://github.com/sparsexlab/torch-amgx/releases>`_
       的预编译 wheel（见 :ref:`prebuilt-native-wheels-zh`）
     - GPU AMG + Krylov（PCG / PBiCGStab / FGMRES）。仅 Linux / Windows +
       NVIDIA CUDA（含 cu12.8 上的 Blackwell ``sm_120``）。最适合 AMG 收敛快
       的超大 SPD / 非对称稀疏系统。**不在 PyPI 上** —— 绑定 CUDA + torch
       版本的 ABI。
   * - ``pyamg``
     - ``pip install pyamg``
     - CPU 端 AMG setup + 跨设备 V-cycle。在**所有**平台（包括 macOS）上可用；
       GPU V-cycle 使用 ``torch.sparse``。

按环境推荐配置
--------------

在下面的选择器中选择你的环境，它会显示推荐的后端以及确切的 ``pip install``
命令 —— 无需查表。``torch_sla.solve(..., backend="auto")`` 已经会针对已安装的
组件挑一个合理的默认值，但安装下面的 extras 能为每种环境解锁最快的路径。

.. raw:: html

   <div id="sla-recommend">
     <noscript>JavaScript 已禁用 —— 推荐配置见下表。</noscript>
   </div>

完整参考表（同时作为无 JavaScript 时的后备）：

.. list-table::
   :widths: 25 35 40
   :header-rows: 1

   * - 环境
     - 安装命令
     - 你会得到什么
   * - **Linux + NVIDIA GPU**
     - ``pip install torch-sla[cudss]`` + AmgX 与 STRUMPACK release wheel
       (:ref:`prebuilt-native-wheels-zh`)
     - GPU 直接 LU（cuDSS）、GPU AMG / Krylov（AmgX）、可移植直接求解
       (STRUMPACK)。完整的 GPU 栈 —— 中等偏稠密用 ``backend="cudss"``，
       AMG 友好的超大系统用 ``backend="amgx"``，另一种直接求解器选择用
       ``backend="strumpack"``。
   * - **Windows + NVIDIA GPU**
     - ``pip install torch-sla[cudss]`` + AmgX release wheel
       (:ref:`prebuilt-native-wheels-zh`)
     - cuDSS + AmgX 提供 Windows wheel。**STRUMPACK 现在可在 Windows（CPU）
       上编译**，通过 ``clang-cl`` + ``flang``\ （CI 预编译的 Windows wheel
       待定）；那里的 GPU 直接 / 迭代求解由 cuDSS / pytorch 覆盖。
   * - **Linux + AMD / Intel GPU**
     - ``pip install torch-sla[pyamg]`` + STRUMPACK rocm release wheel
       (:ref:`prebuilt-native-wheels-zh`)
     - cuDSS **仅限 NVIDIA**，在这里不能运行。PyTorch 原生的 Krylov 求解器
       (CG / BiCGStab / GMRES / MINRES / LSQR / LSMR) 与设备无关，可在 ROCm
       上运行。**STRUMPACK 在 ROCm 上提供 GPU 直接求解**\ （多波前 LU）。
       PyAMG 混合方案在 ROCm / XPU 的 torch 构建上做 CPU setup + 通过
       ``torch.sparse`` 的设备端 V-cycle。
   * - **Linux / Windows 纯 CPU**
     - ``pip install torch-sla[pyamg]``
     - PyAMG 用于 AMG，SciPy 用于直接法 + Krylov，``backend="pytorch"`` 用于
       autograd 友好的 CG / BiCGStab。
   * - **macOS（Intel / Apple Silicon）**
     - ``pip install torch-sla[pyamg]``
     - macOS 上没有 CUDA —— ``amgx`` 和 ``cudss`` 无法安装。PyAMG 混合方案
       + SciPy + pytorch 后端覆盖一切，STRUMPACK 提供 CPU 直接求解器
       （macOS arm64 wheel 见 `torch-strumpack Releases
       <https://github.com/sparsexlab/torch-strumpack/releases>`_，
       :ref:`prebuilt-native-wheels-zh`）；基于 MPS 的稀疏运算仍处于 beta
       但在改进中。

.. tip::

   ``torch_sla.show_backends()`` 会打印本机上实际加载了哪些后端 —— 安装后用
   它确认 GPU 路径拿到了正确的库很方便。
