Installation
============

.. raw:: html

   <ul class="feature-list">
     <li><span class="gradient-text">PyPI</span>: <code>pip install torch-sla</code> — simplest installation</li>
     <li><span class="gradient-text">GitHub</span>: Clone and install for development</li>
     <li><span class="gradient-text">Optional backends</span>: cuDSS, Eigen for enhanced performance</li>
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

    # GPU users: choose one or both CUDA 12+ backends
    pip install torch-sla[cupy]     # + CuPy backend
    pip install torch-sla[cudss]    # + cuDSS backend (fastest direct solver on GPU)

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
     - Native CG/BiCGStab solvers
   * - ``cupy``
     - ``pip install cupy-cuda12x``
     - GPU direct + iterative solvers via cupyx.scipy
   * - ``cudss``
     - ``pip install nvmath-python[cu12]``
     - Best for medium-scale GPU problems (10K-2M DOF)
