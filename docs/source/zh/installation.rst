安装
====

基本安装
--------

.. code-block:: bash

   # 基本安装（自带 torch / numpy / scipy / ninja，CPU 求解器开箱即用）
   pip install torch-sla

   # GPU 用户：按需选择一个或两个 CUDA 12+ 后端
   pip install torch-sla[cupy]    # 添加 CuPy 后端
   pip install torch-sla[cudss]   # 添加 cuDSS 后端（GPU 最快的直接求解器）

   # 完整安装，包含所有运行时后端（不含 dev / docs）
   pip install torch-sla[all]

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
- CUDA Toolkit（GPU 后端需要）
- nvmath-python（可选，cuDSS 后端需要）
- cupy-cuda12x（可选，CuPy 后端需要）

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
   * - ``cupy``
     - cupy-cuda12x
     - ``pip install cupy-cuda12x``
   * - ``cudss``
     - nvmath-python
     - ``pip install nvmath-python[cu12]``
   * - ``pytorch``
     - torch
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

