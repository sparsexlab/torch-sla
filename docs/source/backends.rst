Backends and Capability Matrix
==============================

torch-sla dispatches each :func:`~torch_sla.solve` call to one of several
backends. Pick a backend explicitly via ``backend="..."`` or let
``backend="auto"`` choose based on device, dtype, problem size, and
which optional dependencies are installed.

The current backend lineup and what each supports:

.. list-table:: **Capability matrix**
   :widths: 11 8 8 8 10 12 10 10 10 10
   :header-rows: 1
   :class: capability-table

   * - Backend
     - CPU
     - CUDA
     - ROCm
     - Direct
     - Iterative
     - Complex
     - Batched
     - Distributed
     - Autograd
   * - ``scipy``
     - ✔
     - --
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
     - ✔
     - --
     - CG, BiCGStab, GMRES, MINRES, LSQR, LSMR (+ PCG / PBiCGStab)
     - ✔
     - ✔
     - via ``DSparseTensor``
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
     - ✔ (V-cycle only)
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
     - --
     - AMG, PCG, PBiCGStab, FGMRES (NVIDIA AmgX)
     - --
     - --
     - --
     - ✔

.. note::

   **All six backends are verified correct.** Each is checked against a
   reference solution with the relative residual ‖Ax − b‖ / ‖b‖ at or near
   machine precision. The two direct GPU paths land well inside that
   envelope — measured ``strumpack`` ≈ ``3e-13`` and ``amgx`` ≈ ``5.6e-13``
   on the verification matrices.

.. note::

   ``cudss`` and ``pyamg`` are PyPI-installable, but the two **native
   compiled** backends — ``strumpack`` (``torch-strumpack``) and ``amgx``
   (``torch-amgx``) — ship as **prebuilt wheels on GitHub Releases** (not
   PyPI), and each wheel is ABI-tied to a specific CUDA *and* PyTorch
   version. See :ref:`prebuilt-native-wheels` in the installation guide for
   the wheel-selection rules and a concrete ``pip install --no-deps`` example.

----

The STRUMPACK backend
--------------------

``backend="strumpack"`` is a **portable multifrontal sparse direct
solver**. Unlike cuDSS (which is NVIDIA CUDA only), STRUMPACK runs on
**CPU, CUDA, and AMD ROCm** from the same API, supports both real and
complex matrices, and offers a multifrontal LU factorization. It is fully differentiable: gradients flow through the
adjoint (A\ :sup:`H`) solve, so it drops into autograd pipelines like the
other backends.

In practice STRUMPACK is the answer for a GPU **direct** solve on
hardware where cuDSS cannot go — most importantly AMD ROCm GPUs, where
cuDSS is unavailable. It requires the optional ``torch-strumpack``
package, published as **prebuilt wheels on GitHub Releases** (not PyPI;
see :ref:`prebuilt-native-wheels`) for Linux cpu / cuda / rocm and macOS
arm64. There is **no Windows wheel** — STRUMPACK needs a Fortran compiler
that MSVC does not provide::

    # Grab the matching wheel from
    #   https://github.com/sparsexlab/torch-strumpack/releases
    pip install --no-deps <release-url>/torch_strumpack-...-linux_x86_64.whl

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
     - PyTorch-native; CUDA / ROCm path active when ``torch.cuda.is_available()``
       (ROCm torch builds report as ``cuda``).
   * - ``strumpack``
     - ✔
     - --
     - ✔ (arm64)
     - Multifrontal sparse direct solver (multifrontal LU,
       real + complex). CPU / CUDA / ROCm on Linux + macOS arm64 via
       ``torch-strumpack`` (GitHub-Release wheels,
       :ref:`prebuilt-native-wheels`). **No Windows** — needs Fortran.
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

* **NVIDIA CUDA tensors**: try ``cudss`` (best direct solver) ->
  ``pytorch`` (iterative fallback).
* **AMD ROCm tensors**: cuDSS is **NVIDIA-only** and never runs here, so
  the auto path uses ``pytorch`` (iterative) and, when a direct solve is
  needed, ``strumpack`` (portable multifrontal direct solver on ROCm).
* **CPU tensors, small / medium**: prefer ``scipy`` LU.
* **CPU tensors, large or repeated**: ``pytorch`` CG / BiCGStab keeps the
  memory footprint flat.

Override via ``backend="..."`` whenever you need exact control (e.g.
``backend="cudss"`` to force a direct GPU solve on NVIDIA, or
``backend="strumpack"`` for a direct GPU solve on AMD ROCm where cuDSS is
unavailable).

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
     - Linux + Windows only. NVIDIA GPU required (incl. Blackwell
       ``sm_120`` on cu12.8). Install the prebuilt wheel from
       `torch-amgx Releases <https://github.com/sparsexlab/torch-amgx/releases>`_
       (not PyPI; see :ref:`prebuilt-native-wheels`).
   * - ``petsc``
     - investigating
     - CPU/GPU direct + iterative, distributed (PETSc/hypre BoomerAMG)
     - Linux + macOS easy; Windows via WSL.
