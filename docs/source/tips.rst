Performance Tips
================

Practical guidance for picking a backend, method, and precision once you are
past the :doc:`quick start <introduction>`. The reasoning behind the
direct-vs-iterative trade-off is in :doc:`backends`; the measured numbers are
in :doc:`benchmarks`.

Choosing a backend and method
-----------------------------

- **float64 converges more reliably** with the iterative solvers; float32 can
  stall on ill-conditioned PDE matrices.
- For **SPD matrices**, ``cholesky`` is roughly twice as fast as ``lu`` -- let
  ``matrix_type="auto"`` detect symmetry / positive-definiteness and pick it.
- On **CPU**, ``scipy+lu`` is the default and gives machine precision.
- On **NVIDIA** for problems under ~2M DOF, ``cudss+cholesky`` is the fastest
  direct solver.
- For **larger problems**, ``pytorch+cg`` is the memory-efficient choice and
  the one that reached 169M DOF on a single GPU.
- Beyond a single GPU, :class:`~torch_sla.DSparseTensor` partitions the matrix
  across devices.
- For a **portable GPU direct solve** -- including AMD ROCm, where cuDSS is not
  available -- use ``strumpack`` (multifrontal LU on CPU/CUDA/ROCm).

Reusing work across solves
--------------------------

- For **repeated solves with the same matrix**, factor once with ``A.lu()``
  and reuse the :class:`~torch_sla.LUFactorization` for each right-hand side.
- To apply one set of defaults across many solves, wrap them in a
  :class:`~torch_sla.SolverConfig` scope (see the *Configuring solves*
  section of :doc:`introduction`) rather than repeating kwargs.
- An iterative solve refines toward a tolerance, so a good ``x0`` (e.g. the
  previous step's solution in a time loop) cuts the iteration count.

Accuracy vs. speed
------------------

- The iterative default ``atol ~ 1e-6`` is a stopping point, not a ceiling --
  tighten ``atol`` toward ``1e-12`` when you need more digits, at the cost of
  iterations. See :ref:`Direct vs iterative <direct-vs-iterative>`.
- A good preconditioner (``"amg"`` for PDE problems) can cut the iteration
  count 10-100x on ill-conditioned systems.
