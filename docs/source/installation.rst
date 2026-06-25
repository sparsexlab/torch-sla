Installation
============

.. raw:: html

   <ul class="feature-list">
     <li><span class="gradient-text">PyPI</span>: <code>pip install torch-sla</code> — simplest installation</li>
     <li><span class="gradient-text">GitHub</span>: Clone and install for development</li>
     <li><span class="gradient-text">Optional backends</span>: cuDSS, STRUMPACK, PyAMG, AmgX for enhanced performance</li>
   </ul>

----

Using pip
---------

To install the latest release:

.. code-block:: bash

    pip install torch-sla

Or install from GitHub for the latest development version:

.. code-block:: bash

    pip install git+https://github.com/walkerchi/torch-sla.git

Optional Dependencies
---------------------

The core install pulls in ``torch``, ``numpy``, ``scipy``, and ``ninja`` — enough to
run CPU solvers out of the box. GPU users can pick the backend(s) they need:

.. code-block:: bash

    # NVIDIA GPU users (CUDA 12+, Linux/Windows):
    pip install torch-sla[cudss]    # + cuDSS direct solver (fastest for ~10K-2M DOF, NVIDIA-only)

    # CPU users (all platforms, including macOS):
    pip install torch-sla[pyamg]    # + PyAMG (CPU AMG setup + on-device V-cycle)

    # STRUMPACK direct solver (CPU/CUDA/ROCm) and AmgX (NVIDIA GPU AMG/Krylov)
    # are NOT pip extras — they are prebuilt wheels on GitHub Releases (see below):
    #   https://github.com/sparsexlab/torch-strumpack/releases
    #   https://github.com/sparsexlab/torch-amgx/releases

    # Full installation with all *PyPI-installable* runtime backends
    # (does not include dev/docs; the native torch-amgx / torch-strumpack
    #  backends are GitHub-Release wheels — see "Prebuilt wheels" below)
    pip install torch-sla[all]

    # For development tools (pytest, black, isort, mypy)
    pip install torch-sla[dev]

    # For documentation tools (sphinx, furo)
    pip install torch-sla[docs]

.. note::

   The two **native (compiled) backends** — ``torch-amgx`` (NVIDIA AmgX)
   and ``torch-strumpack`` (STRUMPACK) — are **not on PyPI** and are **not
   pip extras**. In particular, ``torch-sla[all]`` does **not** pull them in.
   They are PyTorch C++/CUDA extensions published as prebuilt wheels on
   **GitHub Releases**. Pick the wheel that matches *both* your CUDA
   version *and* your installed PyTorch version — see
   :ref:`prebuilt-native-wheels` below.

.. _prebuilt-native-wheels:

Prebuilt wheels: torch-amgx & torch-strumpack
---------------------------------------------

``torch-amgx`` and ``torch-strumpack`` are compiled PyTorch extensions, so
**no PyPI package exists**. Download the matching wheel from GitHub Releases:

* **torch-amgx** — https://github.com/sparsexlab/torch-amgx/releases —
  Linux + Windows, Python 3.10-3.13, CUDA 12.4 / 12.6 / 12.8. Each wheel
  carries a per-CUDA build tag in its filename: ``0_cu124`` / ``0_cu126`` /
  ``0_cu128``. The cu12.8 wheels include NVIDIA Blackwell (``sm_100`` /
  ``sm_120``).
* **torch-strumpack** — https://github.com/sparsexlab/torch-strumpack/releases —
  Linux (cpu / cuda / rocm) + macOS arm64, Python 3.10-3.13. **Windows (CPU) is
  supported** — STRUMPACK builds on Windows with ``clang-cl`` (C/C++) + ``flang``
  (Fortran) from conda-forge, linked against MSVC-built PyTorch (a clean-env solve
  gives relative residual ~1.7e-16). A prebuilt Windows wheel via CI is being added.

.. warning::

   **ABI compatibility — read before downloading.** These wheels are torch
   C++ extensions, so each one is ABI-tied to **both** (a) the CUDA version
   it was built against *and* (b) the specific PyTorch version. You must:

   1. Pick the wheel whose ``0_cuXXX`` tag matches ``torch.version.cuda``
      (check with ``python -c "import torch; print(torch.version.cuda)"``).
   2. Have a PyTorch version matching the one the wheel was built against
      (shown in the release notes).

   A mismatch fails **at import**, not at install — with
   ``DLL load failed ... procedure not found`` on Windows or an
   ``undefined symbol`` error on Linux.

Install a release wheel directly by URL with ``--no-deps`` (so pip does not
try to resolve a non-existent PyPI package):

.. code-block:: bash

   # Example: torch-amgx for CUDA 12.6 + CPython 3.13 (replace with the
   # exact asset URL from the Releases page for your torch / CUDA / Python)
   pip install --no-deps \
     https://github.com/sparsexlab/torch-amgx/releases/download/<tag>/torch_amgx-<ver>-0_cu126-cp313-cp313-linux_x86_64.whl

   # torch-strumpack (cuda / rocm / cpu / macos-arm64 variants on the page)
   pip install --no-deps \
     https://github.com/sparsexlab/torch-strumpack/releases/download/<tag>/torch_strumpack-<ver>-cp313-cp313-linux_x86_64.whl

After install, confirm the backend loaded with ``torch_sla.show_backends()``.

.. admonition:: Verify your environment

   After installation, you can inspect which backends are available on your
   machine:

   .. code-block:: python

      import torch_sla
      torch_sla.show_backends()

Backend Requirements
--------------------

.. list-table::
   :widths: 20 30 50
   :header-rows: 1

   * - Backend
     - Installation
     - Notes
   * - ``scipy``
     - ``pip install scipy``
     - Default, always available
   * - ``pytorch``
     - Included with PyTorch
     - Native CG/BiCGStab solvers. Device-agnostic — runs on CPU / CUDA / ROCm.
   * - ``strumpack``
     - prebuilt wheel from `torch-strumpack Releases
       <https://github.com/sparsexlab/torch-strumpack/releases>`_
       (see :ref:`prebuilt-native-wheels`)
     - Portable multifrontal sparse **direct** solver (multifrontal LU,
       real + complex, full autograd). Runs on **CPU / CUDA / ROCm**
       (Linux) + macOS arm64 — the direct-solver path on AMD ROCm where
       cuDSS is unavailable. **Not on PyPI.** Windows (CPU) is supported —
       STRUMPACK builds with ``clang-cl`` + ``flang``; a prebuilt Windows
       wheel via CI is being added.
   * - ``cudss``
     - ``pip install nvmath-python[cu12]``
     - Best for medium-scale GPU problems (10K-2M DOF). NVIDIA CUDA only.
   * - ``amgx``
     - prebuilt wheel from `torch-amgx Releases
       <https://github.com/sparsexlab/torch-amgx/releases>`_
       (see :ref:`prebuilt-native-wheels`)
     - GPU AMG + Krylov (PCG / PBiCGStab / FGMRES). Linux / Windows +
       NVIDIA CUDA only (incl. Blackwell ``sm_120`` on cu12.8). Best for
       very large SPD / non-symmetric sparse systems where AMG converges
       fast. **Not on PyPI** — ABI-tied to CUDA + torch version.
   * - ``pyamg``
     - ``pip install pyamg``
     - CPU AMG setup + cross-device V-cycle. Works on **all** platforms
       (including macOS); GPU V-cycle uses ``torch.sparse``.

Recommended Setup by Environment
--------------------------------

Use this table to pick the "best-bang-for-buck" backend mix for your machine.
``torch_sla.solve(..., backend="auto")`` will already pick a reasonable
default for whatever's installed, but installing the extras below unlocks
the fastest path for each environment.

.. list-table::
   :widths: 25 35 40
   :header-rows: 1

   * - Environment
     - Install command
     - What you get
   * - **Linux + NVIDIA GPU**
     - ``pip install torch-sla[cudss]`` + AmgX & STRUMPACK release wheels
       (:ref:`prebuilt-native-wheels`)
     - Direct GPU LU (cuDSS), GPU AMG / Krylov (AmgX), portable direct solve
       (STRUMPACK). The full GPU stack — use ``backend="cudss"`` for medium
       dense-ish, ``backend="amgx"`` for AMG-friendly very-large systems, and
       ``backend="strumpack"`` as an alternative direct solver.
   * - **Windows + NVIDIA GPU**
     - ``pip install torch-sla[cudss]`` + AmgX release wheel
       (:ref:`prebuilt-native-wheels`)
     - cuDSS + AmgX ship Windows wheels. **STRUMPACK now builds on Windows
       (CPU)** via ``clang-cl`` + ``flang`` (prebuilt Windows wheel via CI
       pending); cuDSS / pytorch cover GPU direct / iterative solves there.
   * - **Linux + AMD / Intel GPU**
     - ``pip install torch-sla[pyamg]`` + STRUMPACK rocm release wheel
       (:ref:`prebuilt-native-wheels`)
     - cuDSS is **NVIDIA-only** and does not run here. The PyTorch-native
       Krylov solvers (CG / BiCGStab / GMRES / MINRES / LSQR / LSMR) are
       device-agnostic and run on ROCm. **STRUMPACK gives a GPU direct
       solve on ROCm** (multifrontal LU). PyAMG-hybrid does CPU setup
       + on-device V-cycle via ``torch.sparse`` on ROCm / XPU torch builds.
   * - **Linux / Windows CPU-only**
     - ``pip install torch-sla[pyamg]``
     - PyAMG for AMG, SciPy for direct + Krylov, ``backend="pytorch"`` for
       autograd-friendly CG / BiCGStab.
   * - **macOS (Intel / Apple Silicon)**
     - ``pip install torch-sla[pyamg]``
     - CUDA is unavailable on macOS — ``amgx`` and ``cudss`` won't
       install. PyAMG-hybrid + SciPy + pytorch backends cover everything,
       and STRUMPACK provides a CPU direct solver (macOS arm64 wheel on
       `torch-strumpack Releases
       <https://github.com/sparsexlab/torch-strumpack/releases>`_,
       :ref:`prebuilt-native-wheels`); MPS-backed sparse ops are still
       beta but improving.

.. tip::

   ``torch_sla.show_backends()`` prints which backends actually loaded on
   your machine — handy after install to confirm the GPU paths picked up
   the right libraries.
