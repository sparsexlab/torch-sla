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
   * - ``pytorch``
     - ✔
     - ✔
     - --
     - CG, BiCGStab, GMRES, MINRES, PCG, PBiCGStab
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
   * - ``pyamg``
     - ✔
     - ✔ (V-cycle only)
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
     - AMG, PCG, PBiCGStab, FGMRES (NVIDIA AmgX)
     - --
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
   * - ``pyamg``
     - ✔
     - ✔
     - ✔
     - Setup runs on CPU via the optional ``pyamg`` dependency
       (``pip install pyamg``); the V-cycle dispatches through
       ``torch.sparse`` so the cycle itself runs on whatever device
       the matrix lives on. **Cross-platform AMG**: macOS gets CPU AMG,
       CUDA boxes get GPU V-cycles.

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
     - **available** (this release)
     - CPU AMG setup + cross-device V-cycle
     - Already shipping. See above. Standalone solver +
       :class:`~torch_sla.backends.pyamg_backend.PyAMGHierarchy` for
       preconditioner re-use.
   * - ``amgx``
     - **available** (this release)
     - CUDA AMG + Krylov (Nvidia AmgX)
     - Linux + Windows only. NVIDIA GPU required.
       Install via ``pip install torch-sla[amgx]`` (pulls torch-amgx wheels).
   * - ``petsc``
     - investigating
     - CPU/GPU direct + iterative, distributed (PETSc/hypre BoomerAMG)
     - Linux + macOS easy; Windows via WSL.
