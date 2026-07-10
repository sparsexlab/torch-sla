示例
====

本节提供 torch-sla 的实用示例。

.. raw:: html

   <div class="recommendation-box">
     <h3><span class="gradient-text">快速导航</span></h3>
     <ul class="feature-list">
       <li><span class="gradient-text">可视化</span>: <code>spy()</code> 稀疏模式分析</li>
       <li><span class="gradient-text">I/O 操作</span>: Matrix Market 和 SafeTensors 格式支持</li>
       <li><span class="gradient-text">线性求解</span>: 直接法和迭代法求解器，支持梯度</li>
       <li><span class="gradient-text">矩阵分解</span>: SVD、特征值、LU 分解</li>
       <li><span class="gradient-text">高级应用</span>: 非线性求解、分布式计算</li>
     </ul>
   </div>

----

可视化
------

稀疏模式图 (Spy Plot)
~~~~~~~~~~~~~~~~~~~~~

使用 ``.spy()`` 方法可视化稀疏矩阵的非零元素分布。

**代码：**

.. code-block:: python

   import torch
   from torch_sla import SparseTensor

   # 创建 2D Poisson 矩阵（5点模板）
   n = 50
   val, row, col = [], [], []
   for i in range(n):
       for j in range(n):
           idx = i * n + j
           val.append(4.0); row.append(idx); col.append(idx)
           if j > 0: val.append(-1.0); row.append(idx); col.append(idx-1)
           if j < n-1: val.append(-1.0); row.append(idx); col.append(idx+1)
           if i > 0: val.append(-1.0); row.append(idx); col.append(idx-n)
           if i < n-1: val.append(-1.0); row.append(idx); col.append(idx+n)

   A = SparseTensor(torch.tensor(val), torch.tensor(row), torch.tensor(col), (n*n, n*n))

   # 可视化稀疏模式
   A.spy(title="2D Poisson (5点模板)")

**输出示例：**

.. list-table::
   :widths: 50 50
   :header-rows: 0

   * - .. figure:: ../../../assets/examples/spy_poisson_10x10.png
          :width: 100%
          :align: center

          **2D Poisson (10×10)** - 100 DOF，带网格线的5点模板

     - .. figure:: ../../../assets/examples/spy_poisson_50x50.png
          :width: 100%
          :align: center

          **2D Poisson (50×50)** - 2,500 DOF，可见带状结构

   * - .. figure:: ../../../assets/examples/spy_tridiag_30x30.png
          :width: 100%
          :align: center

          **三对角矩阵 (30×30)** - 经典1D Poisson模式

     - .. figure:: ../../../assets/examples/spy_random_100x100.png
          :width: 100%
          :align: center

          **随机稀疏 (100×100)** - 800个随机非零元素

每个非零元素渲染为一个彩色像素，强度与其绝对值成正比。零元素为白色。

----

I/O 操作
--------

Matrix Market 格式
~~~~~~~~~~~~~~~~~~

以标准 Matrix Market (.mtx) 格式保存和加载稀疏矩阵。

**代码：**

.. code-block:: python

   from torch_sla import SparseTensor, save_matrix_market, load_matrix_market

   # 创建稀疏矩阵
   A = SparseTensor(val, row, col, (100, 100))

   # 保存为 Matrix Market 格式
   save_matrix_market(A, "matrix.mtx", comment="My sparse matrix")

   # 从 Matrix Market 格式加载
   B = load_matrix_market("matrix.mtx", device="cuda")

   # 验证
   assert torch.allclose(A.to_dense(), B.to_dense())

**文件格式 (.mtx)：**

::

   %%MatrixMarket matrix coordinate real general
   % My sparse matrix
   100 100 500
   1 1 4.0
   1 2 -1.0
   ...

----

SafeTensors 格式
~~~~~~~~~~~~~~~~

使用高效的 safetensors 格式保存和加载。

**代码：**

.. code-block:: python

   from torch_sla import SparseTensor

   A = SparseTensor(val, row, col, shape)

   # 保存
   A.save("matrix.safetensors")

   # 加载
   B = SparseTensor.load("matrix.safetensors", device="cuda")

   # 分布式保存（用于多卡）
   A.save_distributed("matrix_dist/", num_partitions=4)

----

基本用法
--------

基本稀疏线性求解
~~~~~~~~~~~~~~~~

使用 ``SparseTensor`` 求解稀疏线性系统 :math:`Ax = b`。

**线性系统：**

给定稀疏矩阵 :math:`A \in \mathbb{R}^{n \times n}` 和右端项 :math:`b \in \mathbb{R}^n`，求 :math:`x \in \mathbb{R}^n` 使得：

.. math::

   Ax = b \quad \Leftrightarrow \quad x = A^{-1} b

**求解方法：**

- **直接求解器** （LU, Cholesky）：精确解，稀疏情况下 :math:`O(n^{1.5})`
- **迭代求解器** （CG, BiCGStab）：近似解，:math:`O(k \cdot \text{nnz})`，其中 :math:`k` 是迭代次数

**问题：**

.. math::

   A = \begin{pmatrix}
   4 & -1 & 0 \\
   -1 & 4 & -1 \\
   0 & -1 & 4
   \end{pmatrix}, \quad
   b = \begin{pmatrix} 1 \\ 2 \\ 3 \end{pmatrix}

这是来自1D Poisson离散化的3×3对称正定（SPD）三对角矩阵。

**COO 格式：**

.. list-table::
   :header-rows: 1
   :widths: 15 15 15 15

   * - 索引
     - 行
     - 列
     - 值
   * - 0
     - 0
     - 0
     - 4.0
   * - 1
     - 0
     - 1
     - -1.0
   * - 2
     - 1
     - 0
     - -1.0
   * - 3
     - 1
     - 1
     - 4.0
   * - 4
     - 1
     - 2
     - -1.0
   * - 5
     - 2
     - 1
     - -1.0
   * - 6
     - 2
     - 2
     - 4.0

**解：**

.. math::

   x = A^{-1}b = \begin{pmatrix} 0.4643 \\ 0.8571 \\ 0.9643 \end{pmatrix}

**代码：**

.. code-block:: python

   import torch
   from torch_sla import SparseTensor

   # 从稠密矩阵创建稀疏矩阵（小矩阵更易读）
   dense = torch.tensor([[4.0, -1.0,  0.0],
                         [-1.0, 4.0, -1.0],
                         [ 0.0, -1.0, 4.0]], dtype=torch.float64)

   A = SparseTensor.from_dense(dense)
   b = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

   x = A.solve(b)

----

属性检测
~~~~~~~~

检测矩阵属性以优化求解器选择。

**对称性：** :math:`A = A^T`

**正定性：** 所有特征值 :math:`\lambda_i > 0`

对于三对角矩阵：:math:`\lambda_1 \approx 2.59, \lambda_2 = 4.0, \lambda_3 \approx 5.41`\ （全为正 → SPD）

**代码：**

.. code-block:: python

   from torch_sla import SparseTensor

   # 使用上面的三对角矩阵
   dense = torch.tensor([[4.0, -1.0,  0.0],
                         [-1.0, 4.0, -1.0],
                         [ 0.0, -1.0, 4.0]], dtype=torch.float64)
   A = SparseTensor.from_dense(dense)

   is_sym = A.is_symmetric()              # tensor(True)
   is_pd = A.is_positive_definite()       # tensor(True)

----

梯度计算
~~~~~~~~

通过隐式微分计算稀疏求解的梯度。

**隐式微分：**

给定 :math:`x = A^{-1} b`，对于损失函数 :math:`L(x)`，我们需要 :math:`\frac{\partial L}{\partial A}` 和 :math:`\frac{\partial L}{\partial b}`。

由 :math:`Ax = b`，对两边求微分：

.. math::

   dA \cdot x + A \cdot dx = db

求解 :math:`dx`：

.. math::

   dx = A^{-1}(db - dA \cdot x)

**伴随法：**

定义伴随变量 :math:`\lambda = A^{-T} \frac{\partial L}{\partial x}`，则：

.. math::

   \frac{\partial L}{\partial A_{ij}} = -\lambda_i \cdot x_j, \quad
   \frac{\partial L}{\partial b} = \lambda

**梯度公式（总结）：**

.. math::

   \frac{\partial L}{\partial A} = -\lambda x^T, \quad
   \frac{\partial L}{\partial b} = A^{-T} \frac{\partial L}{\partial x}

**代码：**

.. code-block:: python

   import torch
   from torch_sla import spsolve

   val = torch.tensor([4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0],
                      dtype=torch.float64, requires_grad=True)
   row = torch.tensor([0, 0, 1, 1, 1, 2, 2])
   col = torch.tensor([0, 1, 0, 1, 2, 1, 2])
   b = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True)

   x = spsolve(val, row, col, (3, 3), b)
   loss = x.sum()
   loss.backward()

   # val.grad, b.grad 现在包含梯度

----

指定后端和方法
~~~~~~~~~~~~~~

显式选择求解器后端和方法。

**可用选项：**

.. list-table::
   :header-rows: 1
   :widths: 15 15 40

   * - 后端
     - 设备
     - 方法
   * - ``scipy``
     - CPU
     - ``lu``, ``umfpack``, ``cg``, ``bicgstab``, ``gmres``
   * - ``pytorch``
     - CPU/CUDA/ROCm
     - ``cg``, ``bicgstab``, ``gmres``, ``minres``
   * - ``strumpack``
     - CPU/CUDA/ROCm
     - ``lu``, ``cholesky``, ``ldlt``
   * - ``cudss``
     - CUDA
     - ``lu``, ``cholesky``, ``ldlt``

**代码：**

.. code-block:: python

   from torch_sla import SparseTensor

   A = SparseTensor(val, row, col, (n, n))
   b = torch.randn(n, dtype=torch.float64)

   x1 = A.solve(b, backend='scipy', method='lu')          # 直接法
   x2 = A.solve(b, backend='scipy', method='cg')         # 迭代法（SPD）
   x3 = A.solve(b, backend='scipy', method='bicgstab')   # 迭代法（一般）

----

矩阵操作
~~~~~~~~

计算范数、行列式和特征值。

**Frobenius 范数：**

.. math::

   \|A\|_F = \sqrt{\sum_{i,j} |a_{ij}|^2} = \sqrt{52} \approx 7.21

**行列式：**

.. math::

   \det(A) = \text{特征值之积}

对于三对角矩阵：:math:`\det(A) = 56`

**梯度公式：**

.. math::

   \frac{\partial \det(A)}{\partial A_{ij}} = \det(A) \cdot (A^{-1})_{ji}

**代码：**

.. code-block:: python

   from torch_sla import SparseTensor

   # 使用上面的三对角矩阵
   dense = torch.tensor([[4.0, -1.0,  0.0],
                         [-1.0, 4.0, -1.0],
                         [ 0.0, -1.0, 4.0]], dtype=torch.float64)
   A = SparseTensor.from_dense(dense)

   norm = A.norm('fro')                              # ≈ 7.21
   det = A.det()                                     # 56.0（支持梯度）
   eigenvalues, eigenvectors = A.eigsh(k=2, which='LM')  # 前2个特征值

----

批量求解
--------

批量 SparseTensor
~~~~~~~~~~~~~~~~~

求解具有相同稀疏模式但不同值的多个系统。

**问题：** 求解4个缩放矩阵的系统

.. math::

   A^{(0)} = A, \quad A^{(1)} = 1.1A, \quad A^{(2)} = 1.2A, \quad A^{(3)} = 1.3A

**代码：**

.. code-block:: python

   import torch
   from torch_sla import SparseTensor

   val = torch.tensor([4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0], dtype=torch.float64)
   row = torch.tensor([0, 0, 1, 1, 1, 2, 2])
   col = torch.tensor([0, 1, 0, 1, 2, 1, 2])

   batch_size = 4
   val_batch = val.unsqueeze(0).expand(batch_size, -1).clone()
   for i in range(batch_size):
       val_batch[i] = val * (1.0 + 0.1 * i)

   A = SparseTensor(val_batch, row, col, (batch_size, 3, 3))
   b = torch.randn(batch_size, 3, dtype=torch.float64)

   x = A.solve(b)  # x.shape: [4, 3]

----

多维批量
~~~~~~~~

处理形如 ``[B1, B2, M, N]`` 的形状。

**示例：** 2种材料 × 3种温度 = 6个系统

**代码：**

.. code-block:: python

   import torch
   from torch_sla import SparseTensor

   B1, B2, n = 2, 3, 8
   val_batch = val.unsqueeze(0).unsqueeze(0).expand(B1, B2, -1).clone()

   A = SparseTensor(val_batch, row, col, (B1, B2, n, n))
   b = torch.randn(B1, B2, n, dtype=torch.float64)

   x = A.solve(b)  # x.shape: [2, 3, 8]

----

用于重复求解的 solve_batch
~~~~~~~~~~~~~~~~~~~~~~~~~~~

结构相同但值不同时的高效批量求解。

**用例：** 稀疏模式固定的时间步进

**LU 分解：** :math:`A = LU`，然后求解 :math:`Ly = b`、:math:`Ux = y`

**代码：**

.. code-block:: python

   from torch_sla import SparseTensor

   A = SparseTensor(val, row, col, shape)

   val_batch = torch.stack([val * (1.0 + 0.01 * t) for t in range(100)])  # [100, nnz]
   b_batch = torch.randn(100, n, dtype=torch.float64)

   x_batch = A.solve_batch(val_batch, b_batch)  # [100, n]

----

SparseTensorList
~~~~~~~~~~~~~~~~

处理具有不同稀疏模式的矩阵。

**用例：** 不同元素数量的有限元网格

**代码：**

.. code-block:: python

   from torch_sla import SparseTensor, SparseTensorList

   A1 = SparseTensor(val1, row1, col1, (5, 5))
   A2 = SparseTensor(val2, row2, col2, (10, 10))
   A3 = SparseTensor(val3, row3, col3, (15, 15))

   matrices = SparseTensorList([A1, A2, A3])

   b_list = [torch.randn(5), torch.randn(10), torch.randn(15)]
   x_list = matrices.solve(b_list)

----

分布式求解
----------

基本 DSparseTensor
~~~~~~~~~~~~~~~~~~

使用域分解创建分布式稀疏张量。

**域分解：** 将16节点网格分成2个域

.. math::

   \text{域 0: } \{0,...,7\}, \quad \text{域 1: } \{8,...,15\}

每个域有 **自有节点** 和来自邻居的 **halo/ghost节点**。

**代码：**

.. code-block:: python

   from torch_sla import DSparseTensor

   D = DSparseTensor(val, row, col, (16, 16), num_partitions=2)

   for i in range(D.num_partitions):
       p = D[i]
       # p.num_owned, p.num_halo, p.num_local

----

2D Poisson 示例
~~~~~~~~~~~~~~~

用5点模板创建2D Poisson矩阵。

**模板：**

.. math::

   \frac{1}{h^2} \begin{pmatrix} & -1 & \\ -1 & 4 & -1 \\ & -1 & \end{pmatrix}

**代码：**

.. code-block:: python

   import torch
   from torch_sla import DSparseTensor

   def create_2d_poisson(nx, ny):
       N = nx * ny
       rows, cols, vals = [], [], []
       for i in range(ny):
           for j in range(nx):
               idx = i * nx + j
               rows.append(idx); cols.append(idx); vals.append(4.0)
               if j > 0:
                   rows.append(idx); cols.append(idx - 1); vals.append(-1.0)
               if j < nx - 1:
                   rows.append(idx); cols.append(idx + 1); vals.append(-1.0)
               if i > 0:
                   rows.append(idx); cols.append(idx - nx); vals.append(-1.0)
               if i < ny - 1:
                   rows.append(idx); cols.append(idx + nx); vals.append(-1.0)
       return (torch.tensor(vals), torch.tensor(rows), torch.tensor(cols), (N, N))

   val, row, col, shape = create_2d_poisson(4, 4)
   D = DSparseTensor(val, row, col, shape, num_partitions=2)

----

分发与收集
~~~~~~~~~~

将全局向量分发到各分区，再收集回来。

**示意图：**

::

   全局:    [x0, x1, x2, x3, x4, x5, x6, x7]
                     ↓ 分发 (scatter)
   本地:    [x0, x1, x2, x3, x6]  (P0 + halo)
            [x4, x5, x6, x7, x3]  (P1 + halo)
                     ↓ 收集 (gather)
   全局:    [x0, x1, x2, x3, x4, x5, x6, x7]

**代码：**

.. code-block:: python

   from torch_sla import DSparseTensor

   D = DSparseTensor(val, row, col, shape, num_partitions=2)
   x_global = torch.arange(16, dtype=torch.float64)

   x_local = D.scatter_local(x_global)
   x_gathered = D.gather_global(x_local)

----

Halo 交换
~~~~~~~~~

在相邻分区之间交换ghost节点值。

**参考资料：**

- `域分解方法 - Wikipedia <https://en.wikipedia.org/wiki/Domain_decomposition_methods>`_
- `模板代码 - Wikipedia <https://en.wikipedia.org/wiki/Stencil_code>`_
- `Halo 交换讲义 - UIUC CS598 <https://wgropp.cs.illinois.edu/courses/cs598-s15/lectures/lecture25.pdf>`_

**1D 分解示意图：**

::

   分区 0: 自有 [0,1,2,3], Halo [4] ← 来自 P1
   分区 1: 自有 [4,5,6,7], Halo [3] ← 来自 P0

**交换过程：**

::

   之前: P0=[x0,x1,x2,x3,?], P1=[x4,x5,x6,x7,?]
                        ↓ halo_exchange_local()
   之后: P0=[x0,x1,x2,x3,x4], P1=[x4,x5,x6,x7,x3]

**为什么需要：** 对于 :math:`y_3 = \sum_j A_{3,j} x_j`，节点3需要来自P1的 :math:`x_4`。

**代码：**

.. code-block:: python

   from torch_sla import DSparseTensor

   D = DSparseTensor(val, row, col, shape, num_partitions=4)
   x_list = [torch.randn(D[i].num_local) for i in range(D.num_partitions)]

   D.halo_exchange_local(x_list)

----

迭代求解器
----------

PyTorch CG 求解器
~~~~~~~~~~~~~~~~~

对于大规模问题（> 10万 DOF），迭代方法比直接求解器快得多。

**共轭梯度（CG）算法：**

对于对称正定（SPD）矩阵 :math:`A`，CG 最小化：

.. math::

   \phi(x) = \frac{1}{2} x^T A x - b^T x

最小值在 :math:`x^* = A^{-1} b` 处达到。

**CG 迭代：**

从 :math:`x_0` 出发，残差 :math:`r_0 = b - Ax_0`，搜索方向 :math:`p_0 = r_0`：

.. math::

   \alpha_k &= \frac{r_k^T r_k}{p_k^T A p_k} \\
   x_{k+1} &= x_k + \alpha_k p_k \\
   r_{k+1} &= r_k - \alpha_k A p_k \\
   \beta_k &= \frac{r_{k+1}^T r_{k+1}}{r_k^T r_k} \\
   p_{k+1} &= r_{k+1} + \beta_k p_k

**收敛性：**

CG 在最多 :math:`n` 次迭代内收敛（精确算术）。条件数为 :math:`\kappa = \lambda_{\max}/\lambda_{\min}` 时：

.. math::

   \|x_k - x^*\|_A \leq 2 \left( \frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1} \right)^k \|x_0 - x^*\|_A

**预处理：**

预处理 CG（PCG）求解 :math:`M^{-1} A x = M^{-1} b`，其中 :math:`M \approx A`：

- Jacobi：:math:`M = \text{diag}(A)` — 简单、有效
- 不完全 Cholesky：:math:`M = \tilde{L} \tilde{L}^T` — 更适合病态问题

**收敛示例：**

.. figure:: ../../../assets/examples/cg_convergence.png
   :width: 100%
   :align: center

   不同规模2D Poisson问题的CG收敛曲线。较大问题由于条件数更差需要更多迭代。

**性能对比（100万 DOF，NVIDIA H200）：**

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 20

   * - 方法
     - 时间
     - 内存
     - 最适用于
   * - ``pytorch+cg``
     - **0.5s** ✅
     - ~500 MB
     - > 10万 DOF，SPD
   * - ``cudss+cholesky``
     - 7.8s
     - ~300 MB
     - < 10万 DOF，高精度

**代码：**

.. code-block:: python

   from torch_sla import spsolve

   # 对于大型 SPD 系统，使用 PyTorch CG
   x = spsolve(val, row, col, shape, b,
               backend='pytorch',
               method='cg',
               preconditioner='jacobi')

----

预处理器
~~~~~~~~

迭代求解器可用的预处理器：

.. list-table::
   :header-rows: 1
   :widths: 15 40 20

   * - 名称
     - 描述
     - 最适用于
   * - ``jacobi``
     - 对角缩放（默认）
     - 通用，最快
   * - ``ssor``
     - 对称 SOR（ω=1.5）
     - 收敛缓慢的问题
   * - ``polynomial``
     - Neumann 级数（degree=2）
     - GPU 并行
   * - ``ic0``
     - 不完全 Cholesky
     - 极度病态
   * - ``amg``
     - 代数多重网格
     - Float32、类 Poisson 问题

**推荐：**

- **Float64**：使用 ``jacobi``\ （最简单、迭代次数少因而最快）
- **Float32**：使用 ``amg``\ （减少迭代次数，弥补精度损失）

**代码：**

.. code-block:: python

   # Jacobi（默认，推荐用于 float64）
   x = spsolve(val, row, col, shape, b,
               backend='pytorch', preconditioner='jacobi')

   # AMG（推荐用于 float32）
   x = spsolve(val.float(), row, col, shape, b.float(),
               backend='pytorch', preconditioner='amg')

----

混合精度
~~~~~~~~

对于内存受限的场景，使用混合精度：
- 矩阵以 Float32 存储（节省内存）
- 累加以 Float64 进行（高精度）

**代码：**

.. code-block:: python

   x = spsolve(val_f32, row, col, shape, b_f32,
               backend='pytorch',
               method='cg',
               mixed_precision=True)  # 返回 float64 解

----

CUDA 用法
---------

移动到 CUDA
~~~~~~~~~~~

传输到GPU进行CUDA加速求解。

**性能：** cuDSS 对于大型系统可快10-100倍。

.. note::

   同样的代码无需修改即可在 **AMD ROCm** 上运行：ROCm 版 PyTorch 会
   将 AMD GPU 暴露为 ``device='cuda'``，因此 ``A.cuda()`` /
   ``b.cuda()`` 会把张量移动到 AMD GPU 上。PyTorch 原生迭代求解器与
   ``strumpack`` 直接求解器都可在 ROCm 上运行；``cudss`` 不行（仅支持 NVIDIA）。

**代码：**

.. code-block:: python

   from torch_sla import SparseTensor

   A = SparseTensor(val, row, col, shape)
   A_cuda = A.cuda()

   x = A_cuda.solve(b.cuda())

----

CUDA上的后端选择
~~~~~~~~~~~~~~~~

**自动选择（NVIDIA）：** cuDSS（首选）→ pytorch（迭代备选）

**自动选择（AMD ROCm）：** strumpack（直接）/ pytorch（迭代）— cuDSS 仅支持 NVIDIA

**代码：**

.. code-block:: python

   x = A_cuda.solve(b_cuda, backend='cudss', method='lu')        # 仅 NVIDIA
   x = A_cuda.solve(b_cuda, backend='cudss', method='cholesky')  # 对于 SPD，仅 NVIDIA
   x = A_cuda.solve(b_cuda, backend='strumpack', method='lu')    # CUDA 或 ROCm 直接求解

----

高级示例
--------

非线性求解与伴随梯度
~~~~~~~~~~~~~~~~~~~~

使用伴随法求解非线性方程 :math:`F(u, \theta) = 0` 并自动计算梯度。

**问题描述：**

给定非线性残差函数 :math:`F: \mathbb{R}^n \times \mathbb{R}^p \to \mathbb{R}^n`，求 :math:`u^*` 使得：

.. math::

   F(u^*, \theta) = 0

其中 :math:`\theta` 为参数（例如强迫项、材料属性）。

**Newton-Raphson 方法：**

从初始猜测 :math:`u_0` 开始，迭代：

.. math::

   u_{k+1} = u_k - J_F^{-1}(u_k) F(u_k)

其中 :math:`J_F = \frac{\partial F}{\partial u}` 是Jacobian矩阵。

**梯度的伴随法：**

对于损失函数 :math:`L(u^*)`，关于参数的梯度为：

.. math::

   \frac{\partial L}{\partial \theta} = -\lambda^T \frac{\partial F}{\partial \theta}

其中伴随变量 :math:`\lambda` 满足：

.. math::

   J_F^T \lambda = \frac{\partial L}{\partial u}

这种方法内存高效：图节点数为 O(1) 而非 O(迭代次数)。

**示例：** 非线性扩散 :math:`Au + u^2 = f`

.. code-block:: python

   import torch
   from torch_sla import SparseTensor

   # 创建刚度矩阵
   A = SparseTensor(val, row, col, (n, n))

   # 定义非线性残差: F(u) = Au + u² - f
   def residual(u, A, f):
       return A @ u + u**2 - f

   # 带梯度的参数
   f = torch.randn(n, requires_grad=True)
   u0 = torch.zeros(n)

   # 使用 Newton-Raphson 求解
   u = A.nonlinear_solve(residual, u0, f, method='newton')

   # 通过伴随法计算梯度（内存高效）
   loss = u.sum()
   loss.backward()
   print(f.grad)  # ∂L/∂f

**方法：**

.. list-table::
   :header-rows: 1
   :widths: 15 25 30 30

   * - 方法
     - 更新规则
     - 收敛性
     - 最适用于
   * - ``newton``
     - :math:`u_{k+1} = u_k - J^{-1} F(u_k)`
     - 二次（快）
     - 一般非线性
   * - ``picard``
     - :math:`u_{k+1} = G(u_k)`\ （不动点）
     - 线性（慢）
     - 弱非线性
   * - ``anderson``
     - 带历史的加速不动点
     - 超线性
     - 内存受限

----

带梯度支持的行列式
~~~~~~~~~~~~~~~~~~

计算稀疏矩阵的行列式并支持自动微分。

**行列式定义：**

对于方阵 :math:`A \in \mathbb{R}^{n \times n}`，行列式是一个编码重要矩阵属性的标量值：

.. math::

   \det(A) = \sum_{\sigma \in S_n} \text{sgn}(\sigma) \prod_{i=1}^{n} a_{i,\sigma(i)}

**性质：**

- :math:`\det(AB) = \det(A) \det(B)`
- :math:`\det(A^T) = \det(A)`
- :math:`\det(A^{-1}) = 1/\det(A)`
- 矩阵奇异 ⟺ :math:`\det(A) = 0`

**梯度公式（Jacobi 公式）：**

对于可微损失 :math:`L(\det(A))`：

.. math::

   \frac{\partial \det(A)}{\partial A_{ij}} = \det(A) \cdot (A^{-1})_{ji}

这通过伴随法以 :math:`O(1)` 个图节点高效计算。

**实现：**

- **CPU**：通过 SciPy LU 的 LU 分解
- **CUDA**：稠密转换 + ``torch.linalg.det``
- **梯度**：伴随法（对 :math:`A^{-1}` 所需的列求解 :math:`A \mathbf{x} = \mathbf{e}_i`）

**示例 1：基本行列式**

.. code-block:: python

   import torch
   from torch_sla import SparseTensor

   # 从稠密矩阵得到的 3x3 三对角矩阵
   dense = torch.tensor([[4.0, -1.0,  0.0],
                         [-1.0, 4.0, -1.0],
                         [ 0.0, -1.0, 4.0]], dtype=torch.float64)

   A = SparseTensor.from_dense(dense)
   det = A.det()  # 56.0

**示例 2：梯度计算**

.. code-block:: python

   # 带梯度跟踪的矩阵
   dense = torch.tensor([[2.0, 1.0],
                         [1.0, 3.0]], dtype=torch.float64, requires_grad=True)

   A = SparseTensor.from_dense(dense)
   det = A.det()  # 5.0

   # 计算梯度
   det.backward()
   print(dense.grad)  # [[3.0, -1.0], [-1.0, 2.0]]

**示例 3：CUDA 支持**

.. code-block:: python

   # 移动到 CUDA
   A_cuda = A.cuda()
   det_cuda = A_cuda.det()  # 自动使用 CUDA 后端

**示例 4：批量行列式**

.. code-block:: python

   # 结构相同的多个矩阵
   val_batch = torch.tensor([
       [2.0, 0.0, 0.0, 3.0],  # det = 6
       [1.0, 0.5, 0.5, 1.0],  # det = 0.75
   ], dtype=torch.float64)

   A_batch = SparseTensor(val_batch, row, col, (2, 2, 2))
   det_batch = A_batch.det()  # [6.0, 0.75]

**示例 5：带行列式约束的优化**

.. code-block:: python

   # 优化矩阵以达到目标行列式
   val = torch.tensor([1.0, 0.5, 0.5, 1.0], requires_grad=True)
   target_det = torch.tensor(2.0)
   optimizer = torch.optim.Adam([val], lr=0.1)

   for _ in range(50):
       optimizer.zero_grad()
       A = SparseTensor(val, row, col, (2, 2))
       loss = (A.det() - target_det) ** 2
       loss.backward()
       optimizer.step()

**示例 6：分布式矩阵**

.. code-block:: python

   from torch_sla import DSparseTensor

   # 创建分布式稀疏张量
   D = DSparseTensor(val, row, col, (n, n), num_partitions=4)

   # 计算行列式（会收集所有分区）
   det = D.det()  # 警告：需要数据收集

**数值考量：**

- 对于大型矩阵，行列式可能溢出/下溢
- 为了数值稳定性，考虑使用对数行列式
- 奇异矩阵（det ≈ 0）可能导致 LU 分解失败
- 使用 ``torch.float64`` 以获得更好的数值精度

**性能：**

.. list-table::
   :header-rows: 1
   :widths: 15 15 15 15 40

   * - 矩阵规模
     - CPU（稀疏）
     - CUDA（稠密）
     - CPU-代替-CUDA
     - 备注
   * - 10×10
     - 0.3 ms
     - 1.0 ms
     - 0.5 ms
     - CUDA 慢3倍（稠密开销）
   * - 100×100
     - 0.3 ms
     - 0.3 ms
     - 0.5 ms
     - 性能相近
   * - 1000×1000
     - 0.7 ms
     - 2.5 ms
     - 1.2 ms
     - CUDA 慢3.6倍（O(n³) vs O(n^1.5)）

**⚠️ 重要性能提示：**

对于稀疏行列式，CUDA **比** CPU **慢**！这是因为：

- CPU 使用稀疏 LU 分解：O(nnz^1.5) 时间，O(nnz) 内存
- CUDA 需要稠密转换：O(n³) 时间，O(n²) 内存
- cuDSS 不提供稀疏行列式计算

**推荐：** 对于 CUDA 张量，使用 ``.cpu().det()`` 而非 ``.det()``

----

特征值分解
~~~~~~~~~~

计算稀疏矩阵的特征值和特征向量。

**特征值问题：**

对于矩阵 :math:`A \in \mathbb{R}^{n \times n}`，求特征值 :math:`\lambda_i` 和特征向量 :math:`v_i` 使得：

.. math::

   A v_i = \lambda_i v_i

**对称情况 (eigsh)：**

对于对称矩阵 :math:`A = A^T`，特征值为实数且特征向量正交归一：

.. math::

   A = V \Lambda V^T, \quad V^T V = I

其中 :math:`\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_n)`。

**算法：**

- **ARPACK/LOBPCG**：稀疏矩阵的迭代方法，计算前 k 个特征值
- **移位反转 (Shift-invert)**：用于内部特征值

**梯度公式：**

对于单重特征值 :math:`\lambda_i` 及其特征向量 :math:`v_i`：

.. math::

   \frac{\partial \lambda_i}{\partial A_{jk}} = v_i[j] \cdot v_i[k]

**代码：**

.. code-block:: python

   from torch_sla import SparseTensor

   A = SparseTensor(val, row, col, (n, n))

   # 最大特征值（ARPACK/LOBPCG）
   eigenvalues, eigenvectors = A.eigsh(k=6, which='LM')

   # 最小特征值
   eigenvalues, eigenvectors = A.eigsh(k=6, which='SM')

   # 非对称矩阵
   eigenvalues, eigenvectors = A.eigs(k=6)

**示例输出：**

.. figure:: ../../../assets/examples/eigenvalue_spectrum.png
   :width: 100%
   :align: center

   1D Laplacian (n=50) 的特征值谱。红点显示 ``eigsh()`` 计算的6个最小特征值。

**梯度支持：** 特征值分解是可微分的！

.. code-block:: python

   val = val.requires_grad_(True)
   A = SparseTensor(val, row, col, shape)
   eigenvalues, _ = A.eigsh(k=3)
   loss = eigenvalues.sum()
   loss.backward()  # 梯度流向 val

----

SVD（奇异值分解）
~~~~~~~~~~~~~~~~~

计算稀疏矩阵的截断SVD。

**SVD 定义：**

对于矩阵 :math:`A \in \mathbb{R}^{m \times n}`，SVD为：

.. math::

   A = U \Sigma V^T

其中：

- :math:`U \in \mathbb{R}^{m \times r}`：左奇异向量（列正交归一）
- :math:`\Sigma = \text{diag}(\sigma_1, \ldots, \sigma_r)`：奇异值（:math:`\sigma_1 \geq \sigma_2 \geq \ldots \geq 0`）
- :math:`V \in \mathbb{R}^{n \times r}`：右奇异向量（列正交归一）

**截断 SVD（秩-k 近似）：**

.. math::

   A_k = U_k \Sigma_k V_k^T = \sum_{i=1}^{k} \sigma_i u_i v_i^T

这是 Frobenius 范数下最优的秩-k 近似（Eckart-Young 定理）：

.. math::

   \|A - A_k\|_F = \sqrt{\sum_{i=k+1}^{r} \sigma_i^2}

**与特征值的关系：**

- :math:`\sigma_i^2` 是 :math:`A^T A`（或 :math:`A A^T`）的特征值
- :math:`v_i` 是 :math:`A^T A` 的特征向量
- :math:`u_i` 是 :math:`A A^T` 的特征向量

**应用：**

- **降维**：通过 SVD 实现 PCA
- **低秩近似**：矩阵压缩
- **伪逆**：:math:`A^+ = V \Sigma^{-1} U^T`

**示例输出：**

.. figure:: ../../../assets/examples/svd_lowrank.png
   :width: 100%
   :align: center

   左：奇异值谱显示真实秩后的快速衰减。右：近似误差随秩增加而减少。

**代码：**

.. code-block:: python

   from torch_sla import SparseTensor

   A = SparseTensor(val, row, col, (m, n))

   # 计算前k个奇异值/向量
   U, S, Vt = A.svd(k=10)

   # 低秩近似
   A_approx = U @ torch.diag(S) @ Vt

   # 相对近似误差
   error = (A.to_dense() - A_approx).norm() / A.norm('fro')

----

LU分解用于重复求解
~~~~~~~~~~~~~~~~~~

缓存LU分解以高效地对同一矩阵进行重复求解。

**LU 分解：**

对于矩阵 :math:`A`，求下三角矩阵 :math:`L` 和上三角矩阵 :math:`U` 使得：

.. math::

   PA = LU

其中 :math:`P` 是置换矩阵（用于数值稳定性）。

**用 LU 求解：**

求解 :math:`Ax = b`：

1. 分解一次：:math:`PA = LU` — 代价：:math:`O(n^3)`，稀疏时为 :math:`O(\text{nnz}^{1.5})`
2. 前代：:math:`Ly = Pb` — 代价：:math:`O(n^2)`，稀疏时为 :math:`O(\text{nnz})`
3. 回代：:math:`Ux = y` — 代价：:math:`O(n^2)`，稀疏时为 :math:`O(\text{nnz})`

**复杂度节省：**

对于同一矩阵的 :math:`k` 次求解：

- 无缓存：:math:`O(k \cdot n^{1.5})`\ （稀疏直接法）
- 使用LU缓存：:math:`O(n^{1.5} + k \cdot n)` — 最多快 :math:`\sqrt{n}` 倍！

**用例：** 固定刚度矩阵的时间步进

.. code-block:: python

   from torch_sla import SparseTensor

   A = SparseTensor(val, row, col, shape)

   # 分解一次（昂贵）
   lu = A.lu()

   # 高效地求解多个右端项（便宜）
   for t in range(100):
       b_t = compute_rhs(t)
       x_t = lu.solve(b_t)  # 使用缓存的LU快速求解

----

图神经网络示例
~~~~~~~~~~~~~~

将 torch-sla 用于 GNN 中的图 Laplacian 操作。

**代码：**

.. code-block:: python

   import torch
   from torch_sla import SparseTensor

   # 从边列表创建邻接矩阵
   edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
   edge_weight = torch.ones(4)

   A = SparseTensor(edge_weight, edge_index[0], edge_index[1], (3, 3))

   # 计算度矩阵
   D = A.sum(dim=1)

   # 归一化 Laplacian: L = I - D^(-1/2) A D^(-1/2)
   D_inv_sqrt = D.pow(-0.5)
   A_norm = A * D_inv_sqrt.unsqueeze(1) * D_inv_sqrt.unsqueeze(0)
   L = SparseTensor.eye(3) - A_norm

   # 求解 Laplacian 系统
   x = L.solve(b)

----

Jupyter Notebook 示例
---------------------

交互式示例在 ``examples/`` 目录中以Jupyter notebook形式提供：

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Notebook
     - 描述
   * - ``basic_usage.ipynb``
     - 基本求解、属性检测、可视化
   * - ``batched_solve.ipynb``
     - 批量操作和 SparseTensorList
   * - ``determinant.py``
     - 带梯度支持的行列式计算（CPU 和 CUDA）
   * - ``gcn_example.ipynb``
     - 稀疏 Laplacian 图神经网络
   * - ``nonlinear_solve.ipynb``
     - 带伴随梯度的非线性方程
   * - ``visualization.ipynb``
     - Spy图和稀疏可视化
   * - ``persistence.ipynb``
     - 使用 safetensors 和 Matrix Market 保存/加载
   * - ``suitesparse_demo.ipynb``
     - 从 `SuiteSparse Collection <https://sparse.tamu.edu/>`_ 加载矩阵
   * - ``distributed/``
     - 分布式计算示例（matvec, solve, eigsh）
