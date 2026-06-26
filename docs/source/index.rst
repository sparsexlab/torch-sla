.. torch-sla documentation master file
.. meta::
   :description: torch-sla solves sparse linear systems Ax=b in PyTorch, with autograd and GPU support.
   :keywords: pytorch sparse, sparse linear solver, torch.sparse, differentiable sparse solver, cuDSS, STRUMPACK, FEM, CFD
   :robots: index, follow

.. image:: _static/logo.jpg
   :alt: torch-sla
   :align: center
   :width: 300px

torch-sla: PyTorch Sparse Linear Algebra
========================================

torch-sla solves sparse linear systems :math:`Ax = b` in PyTorch. The matrix
``A`` is stored in sparse form, the solve runs on CPU or GPU, and gradients
flow back through it via ``torch.autograd``. It targets workloads that already
live in PyTorch -- FEM/CFD discretizations, physics-informed networks, graph
operators -- where you would otherwise copy out to SciPy or PETSc and lose the
gradient.

.. raw:: html

   <p align="center">
     <a href="https://arxiv.org/abs/2601.13994"><img src="https://img.shields.io/badge/arXiv-2601.13994-b31b1b.svg" alt="arXiv"></a>
     <a href="https://github.com/walkerchi/torch-sla"><img src="https://img.shields.io/badge/GitHub-torch--sla-blue?logo=github" alt="GitHub"></a>
     <a href="https://pypi.org/project/torch-sla/"><img src="https://img.shields.io/pypi/v/torch-sla?color=green" alt="PyPI"></a>
     <a href="https://www.apache.org/licenses/LICENSE-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License: Apache 2.0"></a>
   </p>

The two core classes
--------------------

Everything in torch-sla hangs off two tensor types. Both expose the same
operation vocabulary (solve, eigsh, matvec, det, ...); they differ only in
where the matrix lives.

:class:`~torch_sla.SparseTensor`
   A single-process sparse matrix in COO form, with an optional batch
   dimension. Runs on CPU or one GPU, dispatches to SciPy / PyTorch-native /
   cuDSS / STRUMPACK backends, and is differentiable through every solve.
   See :doc:`sparse_tensor`.

:class:`~torch_sla.DSparseTensor`
   A row-partitioned sparse matrix sharded across processes/GPUs with domain
   decomposition and halo exchange. Mirrors ``torch.distributed.tensor.DTensor``
   and reuses the single-process operations rank-locally. See
   :doc:`dsparse_tensor`.

For the per-operation reference -- signatures, examples, sparsity figures,
scaling plots -- see :doc:`operations`.

Quick start
-----------

.. code-block:: bash

   pip install torch-sla

.. code-block:: python

   import torch
   from torch_sla import SparseTensor

   dense = torch.tensor([[ 4.0, -1.0,  0.0],
                         [-1.0,  4.0, -1.0],
                         [ 0.0, -1.0,  4.0]], dtype=torch.float64)
   A = SparseTensor.from_dense(dense)

   b = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
   x = A.solve(b)               # auto-selects scipy+lu on CPU

Move the tensors to the GPU and the solve follows, picking cuDSS on NVIDIA:

.. code-block:: python

   x = A.cuda().solve(b.cuda())

To take gradients, set ``requires_grad`` on the values (or use the functional
:func:`~torch_sla.spsolve`) and call ``backward()``:

.. code-block:: python

   from torch_sla import spsolve

   val = torch.tensor([...], requires_grad=True)
   x = spsolve(val, row, col, shape, b)
   x.sum().backward()           # grads w.r.t. val and b

.. toctree::
   :maxdepth: 1
   :hidden:

   introduction
   installation
   sparse_tensor
   dsparse_tensor
   distributed_scaling
   distributed_operations
   operations
   architecture
   backends
   tips
   examples
   benchmarks
   torch_sla

----

How torch-sla compares
======================

The tables below summarize where torch-sla fits relative to common
alternatives. The short version: it earns its place when you need sparse
solves *inside* a PyTorch autograd graph. If you are not using PyTorch, or you
need an established preconditioner stack or true exascale distribution, the
mature libraries below remain the better tools.

vs. ``scipy.sparse.linalg``
---------------------------

.. list-table::
   :widths: 30 35 35
   :header-rows: 1

   * - Feature
     - torch-sla
     - scipy.sparse.linalg
   * - PyTorch tensors
     - Native
     - Requires NumPy copy
   * - GPU
     - cuDSS (NVIDIA), STRUMPACK (CUDA/ROCm)
     - CPU only
   * - Gradients
     - Adjoint through solve/eig/svd
     - None
   * - Batched solve
     - One call
     - Loop
   * - Large scale
     - 169M DOF measured (CG, single GPU)
     - Memory-bound on CPU
   * - Distributed
     - DSparseTensor
     - No

vs. ``torch.linalg.solve``
--------------------------

.. list-table::
   :widths: 30 35 35
   :header-rows: 1

   * - Feature
     - torch-sla
     - torch.linalg.solve
   * - Matrix type
     - Sparse (COO/CSR)
     - Dense only
   * - Memory (1M×1M, 1% density)
     - ~80 MB
     - ~8 TB (infeasible)
   * - Solvers
     - LU, Cholesky, LDLT, CG, BiCGStab, GMRES, MINRES
     - Dense LU
   * - Batching
     - Same or different patterns
     - Same shape only
   * - Gradients
     - O(1) graph nodes via adjoint
     - Standard autograd

vs. NVIDIA AmgX
---------------

.. list-table::
   :widths: 30 35 35
   :header-rows: 1

   * - Feature
     - torch-sla
     - NVIDIA AmgX
   * - Install
     - ``pip install torch-sla``
     - Build from source
   * - PyTorch integration
     - Native
     - Needs a wrapper
   * - Gradients
     - Yes
     - No
   * - Algebraic multigrid
     - Via the ``amgx`` backend (wraps AmgX) or ``pyamg``
     - Core feature
   * - Preconditioners
     - Jacobi, SSOR, polynomial, IC(0), AMG
     - ILU, AMG, and more

vs. PETSc
---------

.. list-table::
   :widths: 30 35 35
   :header-rows: 1

   * - Feature
     - torch-sla
     - PETSc
   * - Install
     - ``pip install``
     - MPI + compilers
   * - PyTorch integration
     - Native tensors
     - petsc4py + copies
   * - Gradients
     - Yes
     - No
   * - Solver coverage
     - Core Krylov + direct
     - Extensive (KSP, SNES)
   * - Distributed
     - DSparseTensor, multi-GPU
     - Full MPI, exascale-proven

PETSc remains the right choice past the scales torch-sla has been tested at, or
when you need its solver and preconditioner breadth.

Indices and search
==================

* :ref:`genindex`
* :ref:`search`

License
-------

torch-sla is released under the Apache License 2.0. Copyright 2024-2026
Mingyuan Chi and Shizheng Wen. See `LICENSE
<https://github.com/walkerchi/torch-sla/blob/main/LICENSE>`_ for details.

Contact
-------

| **Author**: Mingyuan Chi
| **Email**: walker.chi.000@gmail.com
| **Author**: Shizheng Wen
| **Email**: shizheng.wen@sam.math.ethz.ch

Citation
--------

.. code-block:: bibtex

   @article{chi2026torchsla,
     title={torch-sla: Differentiable Sparse Linear Algebra with Adjoint Solvers and Sparse Tensor Parallelism for PyTorch},
     author={Chi, Mingyuan and Wen, Shizheng},
     journal={arXiv preprint arXiv:2601.13994},
     year={2026},
     url={https://arxiv.org/abs/2601.13994}
   }

`arXiv:2601.13994 <https://arxiv.org/abs/2601.13994>`_ — Differentiable Sparse
Linear Algebra with Adjoint Solvers and Sparse Tensor Parallelism for PyTorch.
