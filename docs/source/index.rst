.. torch-sla documentation master file
.. meta::
   :description: torch-sla: PyTorch Sparse Linear Algebra library. GPU-accelerated sparse solvers with autograd. Works with torch.sparse tensors. pip install torch-sla.
   :keywords: torch sparse, torch sparse matrix, torch sparse tensor, pytorch sparse, pytorch sparse matrix, pytorch sparse solver, sparse linear algebra, torch.sparse, GPU sparse solver, CUDA sparse, cuSOLVER, cuDSS, differentiable sparse, autograd sparse, scipy sparse pytorch, spsolve pytorch, FEM pytorch
   :google-site-verification: 
   :robots: index, follow

.. image:: _static/logo.jpg
   :alt: torch-sla - PyTorch Sparse Linear Algebra with GPU Acceleration
   :align: center
   :width: 300px

torch-sla: PyTorch Sparse Linear Algebra
========================================

.. raw:: html

   <p><strong>torch-sla</strong> (<span class="gradient-text">Torch Sparse Linear Algebra</span>) is a memory-efficient, differentiable sparse linear equation solver library for PyTorch with multiple backends. Perfect for scientific computing, FEM, CFD, and machine learning applications requiring sparse matrix operations with automatic differentiation.</p>

.. raw:: html

   <p align="center">
     <a href="https://arxiv.org/abs/2601.13994"><img src="https://img.shields.io/badge/arXiv-2601.13994-b31b1b.svg" alt="arXiv"></a>
     <a href="https://github.com/walkerchi/torch-sla"><img src="https://img.shields.io/badge/GitHub-torch--sla-blue?logo=github" alt="GitHub"></a>
     <a href="https://pypi.org/project/torch-sla/"><img src="https://img.shields.io/pypi/v/torch-sla?color=green" alt="PyPI"></a>
     <a href="https://www.apache.org/licenses/LICENSE-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License: Apache 2.0"></a>
   </p>

Why torch-sla?
--------------

.. raw:: html

   <ul class="feature-list">
     <li>🚀 <span class="gradient-text">High Performance</span>: CUDA-accelerated solvers via CuPy and cuDSS</li>
     <li>💾 <span class="gradient-text">Memory Efficient</span>: Store only non-zero elements, enabling solving of systems with millions of unknowns</li>
     <li>🔄 <span class="gradient-text">Differentiable</span>: Full gradient support through <code>torch.autograd</code></li>
     <li>📦 <span class="gradient-text">Batch Processing</span>: Solve thousands of systems in parallel</li>
     <li>🌐 <span class="gradient-text">Distributed</span>: Domain decomposition with halo exchange for large-scale problems</li>
     <li>🔧 <span class="gradient-text">Flexible</span>: Multiple backends and solver methods</li>
   </ul>

Key Features
------------

.. raw:: html

   <ul class="feature-list">
     <li><span class="gradient-text">Memory efficient</span>: Only stores non-zero elements — a 1M×1M matrix with 1% density uses ~80MB instead of ~8TB</li>
     <li><span class="gradient-text">Full gradient support</span> via torch.autograd for end-to-end differentiable pipelines</li>
     <li><span class="gradient-text">Multiple backends</span>: <a href="https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html">SciPy</a>, <a href="https://eigen.tuxfamily.org/">Eigen</a>, <a href="https://docs.cupy.dev/">CuPy</a>, <a href="https://docs.nvidia.com/cuda/cudss/">cuDSS</a></li>
     <li><span class="gradient-text">Batch solving</span>: Same-layout and different-layout sparse matrices</li>
     <li><span class="gradient-text">Distributed solving</span>: Domain decomposition with halo exchange</li>
     <li><span class="gradient-text">169M+ DOF tested</span>: Scales to very large problems with near-linear complexity</li>
   </ul>

Quick Start
-----------

Installation
~~~~~~~~~~~~

.. code-block:: bash

   pip install torch-sla

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   import torch
   from torch_sla import SparseTensor

   # Create a sparse matrix from dense (easier to read for small matrices)
   dense = torch.tensor([[4.0, -1.0,  0.0],
                         [-1.0, 4.0, -1.0],
                         [ 0.0, -1.0, 4.0]], dtype=torch.float64)

   A = SparseTensor.from_dense(dense)

   # Solve Ax = b
   b = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
   x = A.solve(b)

CUDA Acceleration
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Move to GPU for CUDA-accelerated solving
   A_cuda = A.cuda()
   b_cuda = b.cuda()
   x = A_cuda.solve(b_cuda)  # Uses cuDSS or CuPy automatically

Use Cases
---------

torch-sla is ideal for:

- **Finite Element Method (FEM)**: Solve large sparse systems from FEM discretization
- **Computational Fluid Dynamics (CFD)**: Efficient sparse solvers for Navier-Stokes
- **Physics-Informed Neural Networks (PINNs)**: Differentiable sparse operations for physics constraints
- **Graph Neural Networks**: Sparse message passing and Laplacian operations
- **Optimization**: Gradient-based optimization involving sparse linear systems

.. toctree::
   :maxdepth: 1
   :hidden:

   introduction
   installation
   architecture
   torch_sla
   backends
   examples
   benchmarks

----

Frequently Asked Questions (FAQ)
================================

What is torch-sla?
------------------

torch-sla (Torch Sparse Linear Algebra) is a Python library that provides differentiable sparse linear equation solvers for PyTorch. It solves systems of the form Ax = b where A is a sparse matrix, with full support for automatic differentiation (autograd) and GPU acceleration via CUDA.

How do I solve a sparse linear system in PyTorch?
-------------------------------------------------

Use torch-sla's ``SparseTensor`` class:

.. code-block:: python

   from torch_sla import SparseTensor
   
   # Create sparse matrix from COO format (values, row indices, column indices)
   A = SparseTensor(values, row, col, shape)
   
   # Solve Ax = b
   x = A.solve(b)

This works on both CPU and GPU, and supports gradient computation.

What sparse solvers does torch-sla support?
-------------------------------------------

torch-sla supports multiple backends:

- **CPU**: SciPy (LU, UMFPACK, CG, BiCGStab, GMRES), Eigen (CG, BiCGStab)
- **GPU**: CuPy (LU, CG, GMRES), cuDSS (LU, Cholesky, LDLT)

The library automatically selects the best solver based on your hardware and matrix properties.

Can I compute gradients through sparse solve?
---------------------------------------------

Yes. torch-sla fully supports PyTorch autograd:

.. code-block:: python

   val = torch.tensor([...], requires_grad=True)
   x = spsolve(val, row, col, shape, b)
   loss = x.sum()
   loss.backward()  # Computes gradients w.r.t. val and b

How do I solve batched sparse systems?
--------------------------------------

torch-sla supports batched solving for matrices with the same sparsity pattern:

.. code-block:: python

   # Batched values: [batch_size, nnz]
   A = SparseTensor(val_batch, row, col, (batch_size, M, N))
   x = A.solve(b_batch)  # Solves all systems in parallel

For matrices with different patterns, use ``SparseTensorList``. See `batched solve examples <https://github.com/walkerchi/torch-sla/blob/main/examples/batched_solve.py>`_.

How do I use torch-sla on GPU?
------------------------------

Simply move your tensors to CUDA:

.. code-block:: python

   A_cuda = A.cuda()
   x = A_cuda.solve(b.cuda())  # Uses cuDSS or CuPy

What is the difference between SparseTensor and DSparseTensor?
--------------------------------------------------------------

- ``SparseTensor``: Single sparse matrix (optionally batched), for standard solving
- ``DSparseTensor``: Distributed sparse tensor with domain decomposition, for large-scale parallel computing with halo exchange

Comparison with Alternatives
============================

torch-sla vs scipy.sparse.linalg
--------------------------------

.. list-table::
   :widths: 30 35 35
   :header-rows: 1
   :class: comparison-table

   * - Feature
     - **torch-sla** ✅
     - scipy.sparse.linalg
   * - PyTorch Integration
     - ✅ **Native tensors**
     - ❌ Requires numpy copy
   * - GPU Acceleration
     - ✅ **CUDA (cuDSS, CuPy)**
     - ❌ CPU only
   * - Autograd Gradients
     - ✅ **Full support (adjoint)**
     - ❌ No gradients
   * - Batched Solve
     - ✅ **Parallel batch solve**
     - ❌ Loop required
   * - Large Scale (>2M DOF)
     - ✅ **169M DOF tested**
     - ⚠️ Memory limited
   * - Distributed Computing
     - ✅ **DSparseTensor**
     - ❌ Not supported
   * - Eigenvalue/SVD
     - ✅ **Differentiable**
     - ⚠️ No gradients
   * - Nonlinear Solve
     - ✅ **Newton/Anderson**
     - ❌ Not included

torch-sla vs torch.linalg.solve
-------------------------------

.. list-table::
   :widths: 30 35 35
   :header-rows: 1
   :class: comparison-table

   * - Feature
     - **torch-sla** ✅
     - torch.linalg.solve
   * - Matrix Type
     - ✅ **Sparse (COO/CSR)**
     - ❌ Dense only
   * - Memory (1M×1M, 1% density)
     - ✅ **~80 MB**
     - ❌ ~8 TB (impossible)
   * - Max Problem Size
     - ✅ **500M+ DOF** (multi-GPU, scalable)
     - ❌ ~50K (GPU memory)
   * - Specialized Solvers
     - ✅ **LU, Cholesky, CG, BiCGStab**
     - ⚠️ Dense LU only
   * - Batched Operations
     - ✅ **Same/different patterns**
     - ⚠️ Same shape only
   * - GPU Support
     - ✅ **cuDSS, CuPy, PyTorch**
     - ✅ Yes
   * - Autograd
     - ✅ **O(1) graph nodes**
     - ✅ Yes

torch-sla vs NVIDIA AmgX
------------------------

.. list-table::
   :widths: 30 35 35
   :header-rows: 1
   :class: comparison-table

   * - Feature
     - **torch-sla** ✅
     - NVIDIA AmgX
   * - Installation
     - ✅ **pip install torch-sla**
     - ❌ Complex build process
   * - PyTorch Integration
     - ✅ **Native**
     - ❌ Requires wrapper
   * - Autograd Support
     - ✅ **Full gradient flow**
     - ❌ No gradients
   * - Python API
     - ✅ **Pythonic**
     - ⚠️ C++ focused
   * - Multigrid (AMG)
     - ❌ Not yet
     - ✅ **Core feature**
   * - Preconditioners
     - ⚠️ Jacobi
     - ✅ **ILU, AMG, etc.**
   * - Documentation
     - ✅ **Comprehensive**
     - ⚠️ Limited examples

torch-sla vs PETSc
------------------

.. list-table::
   :widths: 30 35 35
   :header-rows: 1
   :class: comparison-table

   * - Feature
     - **torch-sla** ✅
     - PETSc
   * - Installation
     - ✅ **pip install**
     - ❌ Complex (MPI, compilers)
   * - Learning Curve
     - ✅ **Simple Python API**
     - ❌ Steep (C/Fortran heritage)
   * - PyTorch Integration
     - ✅ **Native tensors**
     - ❌ Requires petsc4py + copies
   * - Autograd
     - ✅ **Full support**
     - ❌ No gradients
   * - Solver Variety
     - ⚠️ Core methods
     - ✅ **Extensive (KSP, SNES)**
   * - Distributed
     - ✅ **DSparseTensor multi-GPU**
     - ✅ **Full MPI support**
   * - Production Scale
     - ✅ **500M+ DOF** (multi-GPU)
     - ✅ **Exascale proven**

Summary: When to Use torch-sla
------------------------------

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Use torch-sla When
     - Consider Alternatives When
   * - ✅ You need **PyTorch integration**
     - You're not using PyTorch
   * - ✅ You need **gradient flow** through solve
     - Gradients not needed
   * - ✅ Problem size up to **500M+ DOF** (multi-GPU)
     - Exascale problems (use PETSc)
   * - ✅ You want **simple pip install**
     - You need AMG preconditioners (AmgX)
   * - ✅ **Batched** sparse systems
     - Complex preconditioning (PETSc)
   * - ✅ **GPU acceleration** with minimal setup
     - Full MPI distributed (PETSc)

Indices and Search
==================

* :ref:`genindex`
* :ref:`search`

License
-------

torch-sla is released under the Apache License 2.0. Copyright 2024-2026 Mingyuan Chi and Shizheng Wen. See `LICENSE <https://github.com/walkerchi/torch-sla/blob/main/LICENSE>`_ for details.

Contact
-------

| **Author**: Mingyuan Chi
| **Email**: walker.chi.000@gmail.com
| **Author**: Shizheng Wen
| **Email**: shizheng.wen@sam.math.ethz.ch

Citation
--------

If you use torch-sla in your research, please cite our paper:

.. code-block:: bibtex

   @article{chi2026torchsla,
     title={torch-sla: Differentiable Sparse Linear Algebra with Adjoint Solvers and Sparse Tensor Parallelism for PyTorch},
     author={Chi, Mingyuan and Wen, Shizheng},
     journal={arXiv preprint arXiv:2601.13994},
     year={2026},
     url={https://arxiv.org/abs/2601.13994}
   }

**Paper**: `arXiv:2601.13994 <https://arxiv.org/abs/2601.13994>`_ - Differentiable Sparse Linear Algebra with Adjoint Solvers and Sparse Tensor Parallelism for PyTorch
