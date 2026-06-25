安装
====

基本安装
--------

.. code-block:: bash

   # 基本安装（自带 torch / numpy / scipy / ninja，CPU 求解器开箱即用）
   pip install torch-sla

   # NVIDIA GPU 用户（CUDA 12+，Linux/Windows）
   pip install torch-sla[cudss]   # 添加 cuDSS 后端（NVIDIA GPU 最快的直接求解器）

   # CPU AMG
   pip install torch-sla[pyamg]   # 添加 PyAMG（CPU 端 AMG setup + 设备端 V-cycle）

   # 完整安装，包含所有 PyPI 可装的运行时后端（不含 dev / docs，也不含
   # 原生编译后端 torch-amgx / torch-strumpack —— 见下方“预编译 wheel”）
   pip install torch-sla[all]

.. note::

   两个**原生编译后端** —— ``torch-amgx``（NVIDIA AmgX）与
   ``torch-strumpack``（STRUMPACK）—— 是 PyTorch 的 C++/CUDA 扩展，
   **不在 PyPI 上**，而是以预编译 wheel 形式发布在 **GitHub Releases**。
   选 wheel 时必须同时匹配你的 CUDA 版本**和** PyTorch 版本，详见
   :ref:`prebuilt-native-wheels-zh`。

.. _prebuilt-native-wheels-zh:
.. _prebuilt-native-wheels:

预编译 wheel：torch-amgx 与 torch-strumpack
-------------------------------------------

这两个后端是编译型 PyTorch 扩展，**没有 PyPI 包**。请到 GitHub Releases
下载与你环境匹配的 wheel：

* **torch-amgx** —— https://github.com/sparsexlab/torch-amgx/releases ——
  Linux + Windows，Python 3.10-3.13，CUDA 12.4 / 12.6 / 12.8。每个 wheel
  文件名带有按 CUDA 区分的构建标签 ``0_cu124`` / ``0_cu126`` /
  ``0_cu128``；cu12.8 的 wheel 含 Blackwell（``sm_100`` / ``sm_120``）。
* **torch-strumpack** —— https://github.com/sparsexlab/torch-strumpack/releases
  —— Linux（cpu / cuda / rocm）+ macOS arm64，Python 3.10-3.13。
  **支持 Windows（CPU）** —— STRUMPACK 可在 Windows 上用 conda-forge 的
  ``clang-cl``（C/C++）+ ``flang``（Fortran）编译，链接 MSVC 构建的 PyTorch
  （干净环境下求解相对残差约 1.7e-16）；CI 预编译的 Windows wheel 正在添加中。

.. warning::

   **ABI 兼容性（下载前必读）。** 这些 wheel 是 torch C++ 扩展，每个都同时
   绑定 (a) 编译时的 CUDA 版本，**和** (b) 具体的 PyTorch 版本。你必须：
   (1) 选择 ``0_cuXXX`` 标签与 ``torch.version.cuda`` 一致的 wheel；
   (2) 安装与该 wheel 编译时一致的 PyTorch 版本。
   不匹配会在 **import 时**（而非安装时）报错：Windows 上是
   ``DLL load failed ... procedure not found``，Linux 上是 undefined symbol。

用 ``--no-deps`` 直接按 URL 安装 release wheel（避免 pip 去解析不存在的
PyPI 包）：

.. code-block:: bash

   # 示例：CUDA 12.6 + CPython 3.13 的 torch-amgx（请用 Releases 页面上与你
   # torch / CUDA / Python 匹配的真实 asset URL 替换）
   pip install --no-deps \
     https://github.com/sparsexlab/torch-amgx/releases/download/<tag>/torch_amgx-<ver>-0_cu126-cp313-cp313-linux_x86_64.whl

安装后用 ``torch_sla.show_backends()`` 确认后端已加载。

从源码安装
----------

.. code-block:: bash

   # 克隆仓库
   git clone https://github.com/walkerchi/torch-sla.git
   cd torch-sla

   # 开发工具（pytest、black、isort、mypy）
   pip install -e ".[dev]"

   # 文档工具（sphinx、furo）
   pip install -e ".[docs]"

系统要求
--------

- Python >= 3.8
- PyTorch >= 1.10.0
- NumPy >= 1.19（核心依赖，自动安装）
- SciPy >= 1.5（核心依赖，自动安装；CPU 默认后端）
- CUDA Toolkit（NVIDIA GPU 后端需要）
- nvmath-python（可选，cuDSS 后端需要，仅 NVIDIA CUDA）
- torch-amgx（可选，NVIDIA AmgX 后端；GitHub Releases wheel，非 PyPI）
- torch-strumpack（可选，STRUMPACK 直接求解器后端；支持 CPU/CUDA/ROCm，
  GitHub Releases wheel，非 PyPI；无 Windows 版）
- pyamg（可选，PyAMG 后端）

后端依赖
--------

.. list-table::
   :widths: 20 40 40
   :header-rows: 1

   * - 后端
     - 依赖
     - 安装方式
   * - ``scipy``
     - scipy
     - ``pip install scipy``
   * - ``cudss``
     - nvmath-python（仅 NVIDIA CUDA）
     - ``pip install nvmath-python[cu12]``
   * - ``strumpack``
     - torch-strumpack（CPU/CUDA/ROCm 可移植多波前稀疏直接求解器；
       Linux + macOS arm64，无 Windows）
     - GitHub Releases 预编译 wheel（非 PyPI，见
       :ref:`prebuilt-native-wheels-zh`）
   * - ``amgx``
     - torch-amgx（NVIDIA AmgX GPU AMG / Krylov；仅 Linux/Windows + CUDA，
       含 Blackwell ``sm_120``）
     - GitHub Releases 预编译 wheel（非 PyPI，见
       :ref:`prebuilt-native-wheels-zh`）
   * - ``pyamg``
     - pyamg（CPU 端 AMG setup + 设备端 V-cycle，全平台含 macOS）
     - ``pip install pyamg``
   * - ``pytorch``
     - torch（设备无关，支持 CPU/CUDA/ROCm 的 Krylov：CG/BiCGStab/GMRES/
       MINRES/LSQR/LSMR）
     - 已包含

验证安装
--------

.. code-block:: python

   import torch
   import torch_sla
   from torch_sla import SparseTensor

   # 打印后端状态表（已安装的后端 + 缺失后端的安装命令）
   torch_sla.show_backends()

   # 快速测试
   A = SparseTensor.from_dense(torch.eye(3, dtype=torch.float64))
   b = torch.ones(3, dtype=torch.float64)
   x = A.solve(b, verbose=True)  # 打印自动选中的 backend/method
   print("求解结果:", x)

