Backends and Capability Matrix
==============================

torch-sla dispatches each :func:`~torch_sla.solve` call to one of several
backends. Pick a backend explicitly via ``backend="..."`` or let
``backend="auto"`` choose based on device, dtype, problem size, and
which optional dependencies are installed.

The current backend lineup and what each supports:

.. list-table:: **Capability matrix**
   :widths: 12 10 10 10 12 10 12 12 12
   :header-rows: 1
   :class: capability-table

   * - Backend
     - CPU
     - CUDA
     - Direct
     - Iterative
     - Complex
     - Batched
     - Distributed
     - Autograd
   * - ``scipy``
     - ✔
     - --
     - LU / UMFPACK
     - CG, BiCGStab, GMRES
     - ✔
     - via batch helpers
     - --
     - ✔
   * - ``eigen``
     - ✔
     - --
     - --
     - CG, BiCGStab
     - --
     - --
     - --
     - ✔
   * - ``pytorch``
     - ✔
     - ✔
     - --
     - CG, BiCGStab, PCG, PBiCGStab
     - ✔
     - ✔
     - via ``DSparseTensor``
     - ✔
   * - ``cupy``
     - --
     - ✔
     - LU (cuSPARSE)
     - CG, GMRES
     - ✔
     - via batch helpers
     - --
     - ✔
   * - ``cudss``
     - --
     - ✔
     - LU / Cholesky / LDL\ :sup:`T` / LDL\ :sup:`H`
     - --
     - ✔
     - --
     - --
     - ✔

----

Platform availability
---------------------

Direct-solver backends bind to vendor libraries; the table below records
which OS each one builds on today.

.. list-table::
   :widths: 18 14 14 14 40
   :header-rows: 1

   * - Backend
     - Linux
     - Windows
     - macOS
     - Notes
   * - ``scipy``
     - ✔
     - ✔
     - ✔
     - Pure SciPy; UMFPACK optional via ``scikit-umfpack``.
   * - ``eigen``
     - ✔
     - ✔
     - ✔
     - C++ extension, compiled at install time.
   * - ``pytorch``
     - ✔
     - ✔
     - ✔
     - PyTorch-native; CUDA path active when ``torch.cuda.is_available()``.
   * - ``cupy``
     - ✔
     - ✔
     - --
     - Requires NVIDIA CUDA. CuPy has no native macOS wheels.
   * - ``cudss``
     - ✔
     - ✔
     - --
     - Requires ``nvmath-python[cu12]`` + NVIDIA CUDA. macOS is not
       supported by Nvidia.

----

When ``backend="auto"`` picks what
---------------------------------

* **CUDA tensors**: try ``cudss`` (best direct solver) -> ``cupy`` (LU) ->
  ``pytorch`` (iterative fallback).
* **CPU tensors, small / medium**: prefer ``scipy`` LU.
* **CPU tensors, large or repeated**: ``pytorch`` CG / BiCGStab keeps the
  memory footprint flat.

Override via ``backend="..."`` whenever you need exact control (e.g.
``backend="cudss"`` to force a direct GPU solve for a single
ill-conditioned system that iterative methods cannot crack).

----

Putting it together
-------------------

The capability matrix maps directly to the :func:`~torch_sla.solve`
parameters: any combination where the cell is ✔ is supported::

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

    # Diagnostic return -- iteration count + residual
    x, info = solve(A_csr, b, return_info=True)
    print(info.iter_count, info.residual, info.converged)

----

Future backends (roadmap)
-------------------------

The next wave of backends will extend the table with cross-platform AMG
preconditioning and high-end GPU AMG:

.. list-table::
   :widths: 18 18 28 36
   :header-rows: 1

   * - Backend
     - Status
     - Capability
     - Notes
   * - ``pyamg``
     - planned
     - CPU AMG (smoother + V-cycle hierarchy)
     - Pure-Python; cross-platform. Setup on CPU, V-cycle dispatched
       through ``torch.sparse`` so the cycle itself runs on any device.
   * - ``amgx``
     - planned
     - CUDA AMG + Krylov (Nvidia AmgX)
     - Linux + Windows only. NVIDIA GPU required.
   * - ``petsc``
     - investigating
     - CPU/GPU direct + iterative, distributed (PETSc/hypre BoomerAMG)
     - Linux + macOS easy; Windows via WSL.
