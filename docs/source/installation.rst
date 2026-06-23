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
    pip install torch-sla[amgx]     # + NVIDIA AmgX GPU AMG / Krylov

    # Portable GPU direct solver (CPU / NVIDIA CUDA / AMD ROCm):
    pip install torch-strumpack     # + STRUMPACK multifrontal direct solver (LU)

    # CPU users (all platforms, including macOS):
    pip install torch-sla[pyamg]    # + PyAMG (CPU AMG setup + on-device V-cycle)

    # Full installation with all runtime backends (does not include dev/docs)
    pip install torch-sla[all]

    # For development tools (pytest, black, isort, mypy)
    pip install torch-sla[dev]

    # For documentation tools (sphinx, furo)
    pip install torch-sla[docs]

.. raw:: html

   <div class="recommendation-box">
     <h4><span class="gradient-text">Verify your environment</span></h4>
     <p>After installation, you can inspect which backends are available on your machine:</p>
     <pre><code>import torch_sla
torch_sla.show_backends()</code></pre>
   </div>

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
     - ``pip install torch-strumpack``
     - Portable multifrontal sparse **direct** solver (multifrontal LU,
       real + complex, full autograd). Runs on **CPU / CUDA / ROCm** —
       the direct-solver path on AMD ROCm where cuDSS is unavailable.
   * - ``cudss``
     - ``pip install nvmath-python[cu12]``
     - Best for medium-scale GPU problems (10K-2M DOF). NVIDIA CUDA only.
   * - ``amgx``
     - ``pip install torch-amgx`` *(Linux/Windows + NVIDIA CUDA only)*
     - GPU AMG + Krylov (PCG / PBiCGStab / FGMRES). Best for very large
       SPD / non-symmetric sparse systems where AMG converges fast.
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
     - ``pip install torch-sla[cudss,amgx] torch-strumpack``
     - Direct GPU LU (cuDSS), GPU AMG / Krylov (AmgX), portable direct solve
       (STRUMPACK). The full GPU stack — use ``backend="cudss"`` for medium
       dense-ish, ``backend="amgx"`` for AMG-friendly very-large systems, and
       ``backend="strumpack"`` as an alternative direct solver.
   * - **Windows + NVIDIA GPU**
     - ``pip install torch-sla[cudss,amgx] torch-strumpack``
     - Same as Linux; cuDSS, AmgX and STRUMPACK all ship Windows wheels.
   * - **Linux + AMD / Intel GPU**
     - ``pip install torch-sla[pyamg] torch-strumpack``
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
       and STRUMPACK provides a CPU direct solver
       (``pip install torch-strumpack``); MPS-backed sparse ops are still
       beta but improving.

.. tip::

   ``torch_sla.show_backends()`` prints which backends actually loaded on
   your machine — handy after install to confirm the GPU paths picked up
   the right libraries.
