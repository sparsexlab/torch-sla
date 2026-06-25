.. torch-sla 中文文档
.. meta::
   :description: torch-sla: PyTorch 稀疏线性代数库。GPU加速的稀疏求解器，支持自动微分。
   :keywords: torch稀疏, pytorch稀疏矩阵, 稀疏线性代数, GPU稀疏求解, CUDA稀疏, 可微分稀疏求解器, 有限元, CFD
   :robots: index, follow

.. image:: ../_static/logo.jpg
   :alt: torch-sla - PyTorch 稀疏线性代数库
   :align: center
   :width: 300px

torch-sla: PyTorch 稀疏线性代数
================================

.. raw:: html

   <p><strong>torch-sla</strong> (<span class="gradient-text">Torch Sparse Linear Algebra</span>) 是一个高效、可微分的 PyTorch 稀疏线性方程求解器库，支持多种后端。适用于科学计算、有限元、计算流体动力学和需要自动微分的稀疏矩阵操作的机器学习应用。</p>

.. raw:: html

   <p align="center">
     <a href="https://arxiv.org/abs/2601.13994"><img src="https://img.shields.io/badge/arXiv-2601.13994-b31b1b.svg" alt="arXiv"></a>
     <a href="https://github.com/walkerchi/torch-sla"><img src="https://img.shields.io/badge/GitHub-torch--sla-blue?logo=github" alt="GitHub"></a>
     <a href="https://pypi.org/project/torch-sla/"><img src="https://img.shields.io/pypi/v/torch-sla?color=green" alt="PyPI"></a>
     <a href="https://www.apache.org/licenses/LICENSE-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License: Apache 2.0"></a>
   </p>

为什么选择 torch-sla？
----------------------

.. raw:: html

   <ul class="feature-list">
     <li>🚀 <span class="gradient-text">高性能</span>: GPU 加速求解，支持 NVIDIA CUDA（cuDSS）和 AMD ROCm（PyTorch 原生 + STRUMPACK）</li>
     <li>💾 <span class="gradient-text">内存高效</span>: 仅存储非零元素，支持求解百万级未知数的系统</li>
     <li>🔄 <span class="gradient-text">可微分</span>: 完整支持 <code>torch.autograd</code> 梯度计算</li>
     <li>📦 <span class="gradient-text">批量处理</span>: 并行求解数千个系统</li>
     <li>🌐 <span class="gradient-text">分布式计算</span>: 支持域分解和 halo 交换的大规模计算</li>
     <li>🔧 <span class="gradient-text">灵活配置</span>: 多种后端和求解方法</li>
   </ul>

核心特性
--------

.. raw:: html

   <ul class="feature-list">
     <li><span class="gradient-text">内存高效</span>: 仅存储非零元素 — 1M×1M 矩阵（1% 密度）仅需 ~80MB 而非 ~8TB</li>
     <li><span class="gradient-text">完整梯度支持</span>: 通过 torch.autograd 实现端到端可微分流程</li>
     <li><span class="gradient-text">多后端支持</span>: <a href="https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html">SciPy</a>、<a href="https://pytorch.org/">PyTorch原生</a>、<a href="https://docs.nvidia.com/cuda/cudss/">cuDSS</a>、STRUMPACK — 其中 PyTorch 原生与 STRUMPACK 也可在 AMD ROCm 上运行</li>
     <li><span class="gradient-text">批量求解</span>: 支持相同和不同稀疏模式的批量矩阵</li>
     <li><span class="gradient-text">分布式求解</span>: 域分解与 halo 交换</li>
     <li><span class="gradient-text">大规模测试</span>: 经过 1.69亿+ DOF 测试，近线性复杂度扩展</li>
   </ul>

快速开始
--------

安装
~~~~

.. code-block:: bash

   pip install torch-sla

基本用法
~~~~~~~~

.. code-block:: python

   import torch
   from torch_sla import SparseTensor

   # 从稠密矩阵创建稀疏矩阵（小矩阵更易读）
   dense = torch.tensor([[4.0, -1.0,  0.0],
                         [-1.0, 4.0, -1.0],
                         [ 0.0, -1.0, 4.0]], dtype=torch.float64)

   A = SparseTensor.from_dense(dense)

   # 求解 Ax = b
   b = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
   x = A.solve(b)

CUDA 加速
~~~~~~~~~

.. code-block:: python

   # 移动到 GPU 进行 CUDA 加速求解
   A_cuda = A.cuda()
   b_cuda = b.cuda()
   x = A_cuda.solve(b_cuda)  # 自动使用 cuDSS

应用场景
--------

torch-sla 非常适合:

- **有限元方法 (FEM)**: 求解来自 FEM 离散化的大型稀疏系统
- **计算流体动力学 (CFD)**: Navier-Stokes 方程的高效稀疏求解器
- **物理信息神经网络 (PINNs)**: 物理约束的可微分稀疏操作
- **图神经网络 (GNN)**: 稀疏消息传递和拉普拉斯操作
- **优化问题**: 涉及稀疏线性系统的梯度优化

.. toctree::
   :maxdepth: 1
   :hidden:

   introduction
   installation
   sparse_tensor
   dsparse_tensor
   distributed_scaling
   operations
   architecture
   backends
   examples
   benchmarks
   torch_sla

----

常见问题 (FAQ)
==============

什么是 torch-sla？
------------------

torch-sla (Torch Sparse Linear Algebra) 是一个 Python 库，提供可微分的 PyTorch 稀疏线性方程求解器。它求解 Ax = b 形式的系统，其中 A 是稀疏矩阵，完整支持自动微分 (autograd) 和 CUDA GPU 加速。

如何在 PyTorch 中求解稀疏线性系统？
-----------------------------------

使用 torch-sla 的 ``SparseTensor`` 类:

.. code-block:: python

   from torch_sla import SparseTensor
   
   # 从 COO 格式创建稀疏矩阵（值、行索引、列索引）
   A = SparseTensor(values, row, col, shape)
   
   # 求解 Ax = b
   x = A.solve(b)

支持 CPU 和 GPU，并支持梯度计算。

torch-sla 支持哪些稀疏求解器？
------------------------------

torch-sla 支持多种后端:

- **CPU**: SciPy (LU, UMFPACK, CG, BiCGStab, GMRES, MINRES)、PyTorch原生 (CG, BiCGStab, GMRES, MINRES)、STRUMPACK (多波前 LU 直接法)
- **GPU — NVIDIA (CUDA)**: PyTorch原生 (CG, BiCGStab, GMRES, MINRES)、cuDSS (LU, Cholesky, LDLT 直接法)、STRUMPACK (多波前 LU 直接法)
- **GPU — AMD (ROCm)**: PyTorch原生 (CG, BiCGStab, GMRES, MINRES)、STRUMPACK (多波前 LU 直接法)。cuDSS 仅支持 NVIDIA，在 ROCm 上不可用。

库会根据硬件和矩阵属性自动选择最佳求解器。

能否通过稀疏求解计算梯度？
--------------------------

可以。torch-sla 完整支持 PyTorch autograd:

.. code-block:: python

   val = torch.tensor([...], requires_grad=True)
   x = spsolve(val, row, col, shape, b)
   loss = x.sum()
   loss.backward()  # 计算 val 和 b 的梯度

如何求解批量稀疏系统？
----------------------

torch-sla 支持相同稀疏模式矩阵的批量求解:

.. code-block:: python

   # 批量值: [batch_size, nnz]
   A = SparseTensor(val_batch, row, col, (batch_size, M, N))
   x = A.solve(b_batch)  # 并行求解所有系统

对于不同模式的矩阵，使用 ``SparseTensorList``。参见 `批量求解示例 <https://github.com/walkerchi/torch-sla/blob/main/examples/batched_solve.py>`_。

如何在 GPU 上使用 torch-sla？
-----------------------------

只需将张量移动到 CUDA:

.. code-block:: python

   A_cuda = A.cuda()
   x = A_cuda.solve(b.cuda())  # 使用 cuDSS

SparseTensor 和 DSparseTensor 有什么区别？
------------------------------------------

- ``SparseTensor``: 单个稀疏矩阵（可选批量），用于标准求解
- ``DSparseTensor``: 分布式稀疏张量，支持域分解和 halo 交换，用于大规模并行计算

与其他方案的对比
================

torch-sla vs scipy.sparse.linalg
--------------------------------

.. list-table::
   :widths: 30 35 35
   :header-rows: 1
   :class: comparison-table

   * - 特性
     - **torch-sla** ✅
     - scipy.sparse.linalg
   * - PyTorch 集成
     - ✅ **原生张量**
     - ❌ 需要 numpy 复制
   * - GPU 加速
     - ✅ **NVIDIA: cuDSS + PyTorch；AMD ROCm: PyTorch + STRUMPACK**
     - ❌ 仅 CPU
   * - Autograd 梯度
     - ✅ **完整支持（伴随法）**
     - ❌ 无梯度
   * - 批量求解
     - ✅ **并行批量求解**
     - ❌ 需要循环
   * - 大规模 (>200万 DOF)
     - ✅ **1.69亿 DOF 已测试**
     - ⚠️ 内存受限
   * - 分布式计算
     - ✅ **DSparseTensor**
     - ❌ 不支持
   * - 特征值/SVD
     - ✅ **可微分**
     - ⚠️ 无梯度
   * - 非线性求解
     - ✅ **Newton/Anderson**
     - ❌ 不包含

torch-sla vs torch.linalg.solve
-------------------------------

.. list-table::
   :widths: 30 35 35
   :header-rows: 1
   :class: comparison-table

   * - 特性
     - **torch-sla** ✅
     - torch.linalg.solve
   * - 矩阵类型
     - ✅ **稀疏 (COO/CSR)**
     - ❌ 仅稠密
   * - 内存 (1M×1M, 1% 密度)
     - ✅ **~80 MB**
     - ❌ ~8 TB（不可能）
   * - 最大问题规模
     - ✅ **5亿+ DOF** （多卡可扩展）
     - ❌ ~5万（GPU 内存限制）
   * - 专用求解器
     - ✅ **LU, Cholesky, CG, BiCGStab**
     - ⚠️ 仅稠密 LU
   * - 批量操作
     - ✅ **相同/不同模式**
     - ⚠️ 仅相同形状
   * - GPU 支持
     - ✅ **NVIDIA: cuDSS + PyTorch；AMD ROCm: PyTorch + STRUMPACK**
     - ✅ 是
   * - Autograd
     - ✅ **O(1) 图节点**
     - ✅ 是

torch-sla vs NVIDIA AmgX
------------------------

.. list-table::
   :widths: 30 35 35
   :header-rows: 1
   :class: comparison-table

   * - 特性
     - **torch-sla** ✅
     - NVIDIA AmgX
   * - 安装
     - ✅ **pip install torch-sla**
     - ❌ 复杂的构建过程
   * - PyTorch 集成
     - ✅ **原生**
     - ❌ 需要封装
   * - Autograd 支持
     - ✅ **完整梯度流**
     - ❌ 无梯度
   * - Python API
     - ✅ **Pythonic**
     - ⚠️ 以 C++ 为主
   * - 多重网格 (AMG)
     - ❌ 暂不支持
     - ✅ **核心功能**
   * - 预处理器
     - ⚠️ Jacobi
     - ✅ **ILU, AMG 等**
   * - 文档
     - ✅ **完善**
     - ⚠️ 示例有限

torch-sla vs PETSc
------------------

.. list-table::
   :widths: 30 35 35
   :header-rows: 1
   :class: comparison-table

   * - 特性
     - **torch-sla** ✅
     - PETSc
   * - 安装
     - ✅ **pip install**
     - ❌ 复杂（MPI、编译器）
   * - 学习曲线
     - ✅ **简单 Python API**
     - ❌ 陡峭（C/Fortran 背景）
   * - PyTorch 集成
     - ✅ **原生张量**
     - ❌ 需要 petsc4py + 复制
   * - Autograd
     - ✅ **完整支持**
     - ❌ 无梯度
   * - 求解器种类
     - ⚠️ 基础求解器
     - ✅ **丰富 (KSP, SNES)**
   * - 分布式
     - ✅ **DSparseTensor 多卡并行**
     - ✅ **完整 MPI 支持**
   * - 生产规模
     - ✅ **5亿+ DOF** (多卡)
     - ✅ **百亿级已验证**

总结：何时使用 torch-sla
------------------------

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - 使用 torch-sla 当
     - 考虑其他方案当
   * - ✅ 需要 **PyTorch 集成**
     - 不使用 PyTorch
   * - ✅ 需要 **梯度流** 通过求解
     - 不需要梯度
   * - ✅ 问题规模 5亿+ DOF (多卡)
     - 百亿级问题（使用 PETSc）
   * - ✅ 想要 **简单的 pip 安装**
     - 需要 AMG 预处理器（AmgX）
   * - ✅ **批量** 稀疏系统
     - 复杂预处理（PETSc）
   * - ✅ **GPU 加速** 且配置简单
     - 完整 MPI 分布式（PETSc）

索引和搜索
==========

* :ref:`genindex`
* :ref:`search`

许可证
------

torch-sla 采用 Apache License 2.0 发布。版权所有 2024-2026 Mingyuan Chi 与 Shizheng Wen。详见 `LICENSE <https://github.com/walkerchi/torch-sla/blob/main/LICENSE>`_。

联系方式
--------

| **作者**: Mingyuan Chi
| **邮箱**: x = walker.chi.000@gmail.com
| **作者**: Shizheng Wen
| **邮箱**: shizheng.wen@sam.math.ethz.ch

引用
----

如果您在研究中使用了 torch-sla，请引用我们的论文:

.. code-block:: bibtex

   @article{chi2026torchsla,
     title={torch-sla: Differentiable Sparse Linear Algebra with Adjoint Solvers and Sparse Tensor Parallelism for PyTorch},
     author={Chi, Mingyuan and Wen, Shizheng},
     journal={arXiv preprint arXiv:2601.13994},
     year={2026},
     url={https://arxiv.org/abs/2601.13994}
   }

**论文**: `arXiv:2601.13994 <https://arxiv.org/abs/2601.13994>`_ - Differentiable Sparse Linear Algebra with Adjoint Solvers and Sparse Tensor Parallelism for PyTorch
