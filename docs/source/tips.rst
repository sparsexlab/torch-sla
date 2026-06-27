Performance Tips
================

Practical guidance for picking a backend, method, and precision once you are
past the :doc:`quick start <introduction>`. The reasoning behind the
direct-vs-iterative trade-off is in :doc:`backends`; the measured numbers are
in :doc:`benchmarks`.

Choosing a backend and method
-----------------------------

``backend="auto"`` already picks a sane default per device; reach for the table
when you want the *fastest* option for a known regime.

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Hardware / regime
     - Backend + method
     - Why
   * - CPU, general
     - ``scipy+lu``
     - default; machine precision (~\ :math:`10^{-14}`)
   * - NVIDIA, < ~2M DOF
     - ``cudss+cholesky``
     - fastest direct solver
   * - Very large (> 2M DOF), single GPU
     - ``pytorch+cg``
     - :math:`O(n)` memory; reached 169M DOF (:doc:`benchmarks`)
   * - AMD ROCm direct
     - ``strumpack``
     - multifrontal LU where cuDSS is absent
   * - Portable GPU direct (CPU/CUDA/ROCm)
     - ``strumpack``
     - one code path on every device
   * - Multi-GPU / multi-node
     - :class:`~torch_sla.DSparseTensor`
     - row-sharded with halo exchange
   * - SPD matrix
     - ``cholesky`` (via ``matrix_type="auto"``)
     - ~2x faster than ``lu``; auto-detected

.. tip::

   Use **float64** for the iterative solvers --- float32 can stall on
   ill-conditioned PDE matrices.

Choosing a preconditioner
-------------------------

A preconditioner :math:`M \approx A` shrinks the iteration count by clustering
the spectrum of :math:`M^{-1}A`. Pass a bare string to ``preconditioner=``
(default ``"jacobi"``):

.. list-table::
   :header-rows: 1
   :widths: 32 24 44

   * - Problem type
     - ``preconditioner``
     - Effect
   * - Elliptic PDE (Poisson, diffusion)
     - ``"amg"``
     - 10--100x fewer iterations on ill-conditioned systems
   * - SPD, moderately conditioned
     - ``"ic0"``
     - incomplete Cholesky; symmetry-preserving, cheap
   * - General ill-conditioned
     - ``"ilu0"`` / ``"ssor"``
     - broad spectral clustering
   * - Mild / unsure
     - ``"jacobi"`` *(default)*
     - diagonal scaling, essentially free
   * - Well-conditioned
     - ``"none"``
     - skip the setup cost entirely

Reusing work across solves
--------------------------

- For **repeated solves with the same matrix**, factor once with ``A.lu()``
  and reuse the :class:`~torch_sla.LUFactorization` for each right-hand side.
- To apply one set of defaults across many solves, wrap them in a
  :class:`~torch_sla.SolverConfig` scope (see *Configuring solves* in
  :doc:`introduction`) rather than repeating kwargs.
- An iterative solve refines toward a tolerance, so a good ``x0`` (e.g. the
  previous step's solution in a time loop) cuts the iteration count.

Accuracy vs. speed
------------------

The iterative solvers stop as soon as the residual meets

.. math::

   \lVert b - A x_k \rVert_2 \;\le\; \mathrm{atol} + \mathrm{rtol}\,\lVert b \rVert_2 .

So ``atol`` (default ``~1e-6``) is a *stopping point, not a ceiling* --- tighten
it toward ``1e-12`` for more digits, at the cost of iterations. Direct solvers
ignore tolerance and return machine precision (~\ :math:`10^{-14}`); see
:ref:`Direct vs iterative <direct-vs-iterative>`.

Troubleshooting
---------------

.. list-table::
   :header-rows: 1
   :widths: 32 34 34

   * - Symptom
     - Likely cause
     - Fix
   * - Iterative solve stalls / will not converge
     - float32 on an ill-conditioned PDE
     - switch to ``float64``
   * - Solution has too few correct digits
     - ``atol`` too loose
     - tighten ``atol`` toward ``1e-12``
   * - Convergence is slow (many iterations)
     - no preconditioner
     - add ``preconditioner="amg"`` (PDE) or ``"ilu0"``
   * - Direct solve runs out of memory
     - :math:`O(n^{1.5})` LU fill-in
     - use ``pytorch+cg`` (iterative, :math:`O(n)` memory)
   * - ``cudss`` unavailable on AMD
     - cuDSS is NVIDIA-only
     - use ``strumpack`` for a GPU-direct solve on ROCm
