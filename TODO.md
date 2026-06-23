## Functionality

- [x] sparse linear solve
- [x] sparse eigen vals
- [x] sparse determination
- [x] non linear solve (Newton + implicit-function-theorem gradients; differentiable wrt A.values + params)
- [ ] ODE operator (genuinely net-new; needs a spec -- time integration du/dt=-Au? operator assembly?)
- [x] matrix partition (already implemented: torch_sla/partition.py -- metis/simple/coordinates/RCB + build_partition; 12 tests pass; pymetis optional)
- [x] **GPU-adapt graph ops** -- `connected_components` reimplemented as parallel pure-torch
      (label-propagation + pointer-jumping via `scatter_reduce`, stays on device, no CPU
      round-trip, no Python edge loop) -> GPU-ready; batched supported (broadcast). Verified
      vs scipy `csgraph.connected_components` (+8 tests). `to_connected_components` batched
      still NotImplementedError (single-matrix path works).
- [~] low precision: fp32 ok for solvers (direct + iterative, ~1e-5..1e-7) and eigsh (fp32
      bug fixed); **det overflows to inf in fp32** (warns -> use logdet/fp64). bf16 largely
      unusable (no CPU sparse-matvec bf16; cuDSS/det/eigsh/QR reject bf16; GPU iterative
      ~1e-2). Recommend fp32 min, fp64 for det/eigsh.
- [x] complex number support (CPU + CUDA(4070ti) + ROCm(780M) all verified, fwd+bwd, incl. eigsh GPU fix)
  - [x] complex dtype in SparseTensor (val complex64/complex128)
  - [x] solve / matvec with complex (scipy + cuDSS + pytorch cg/bicgstab/gmres/minres -- verified relerr ~1e-13)
  - [x] det / eigs with complex -- forward verified (relerr ~1e-16); **backward had
        missing-conjugate bugs (det + eigsh complex adjoint), fixed + regression test
        (tests/test_complex_grad_det_eigsh.py)**. CPU verified vs dense autograd.
  - [x] conjugate-transpose (Hermitian) handling, distinguish `.T` vs `.H`
  - [x] gradient (Wirtinger / conjugate) correctness for complex autograd (solve + det + eigsh, CPU)
  - [x] det / eigs complex backward verified on **ROCm/780M** (device-parametrised
        tests/test_complex_grad_det_eigsh.py): det fwd+bwd real+complex PASS; eigsh
        fwd+bwd real PASS, eigsh complex bwd math verified (grad == analytic using
        eigsh's own vector, 0.0).
  - [x] verify det/eigs complex backward on **NVIDIA CUDA (4070ti, cuDSS path)**: det
        fwd+bwd real+complex PASS (cuDSS complex backward path confirmed); gmres/minres
        all CUDA PASS (23 passed). Test code shipped to 4070ti via scp + native-Windows
        python (torch 2.6+cu124).
  - [x] **FIXED: complex eigsh (LOBPCG) GPU non-convergence** -- root cause: Rayleigh-Ritz
        Gram matrix used plain transpose `X.T @ AX` (non-Hermitian for complex) fed to
        `torch.linalg.eigh`, whose behaviour on non-Hermitian input is undefined -> LAPACK
        (CPU) limped to convergence, cuSOLVER (GPU) did not (residual ~3.5). Fix: conjugate
        transpose `X.mH @ AX` in `_lobpcg_core` (2 RR steps). Verified converged on CPU
        (1e-16), **4070ti CUDA (9.6e-15)**, ROCm 780M (4e-15); xfail removed, full device
        matrix green (8 passed on 4070ti). (scipy lobpcg uses conj-transpose; torch.lobpcg
        has no complex support.)

## Distributed correctness & performance

> Validated: distributed Krylov solves (cg/bicgstab/gmres/fgmres/minres) match
> scipy in multiprocess tests; benchmarked to 400M DOF on 4 GPUs. The earlier
> "NOT trusted / atol=1e-2 does not converge" note (TODO added 2026-05, before
> the 2026-06 preconditioner + true-overlap work) is now stale.

- [x] validate distributed solve correctness at scale
  - [x] `atol=1e-2` non-convergence — was weak Jacobi; resolved by block-Jacobi / AMG preconditioners
  - [x] multiprocess correctness vs scipy (tests/test_distributed_krylov_shard_multiprocess.py)
  - [x] 400M DOF / 4-GPU benchmark
- [~] better preconditioner
  - [x] block-Jacobi
  - [x] AMG preconditioner (PyAMG + torch-amgx, per-rank block-Jacobi AMG)
  - [ ] ILU / additive Schwarz (optional, future)
- [x] halo exchange correctness — validated implicitly by distributed Krylov matching scipy
- [x] communication/computation overlap — true NCCL overlap (#7) via `batch_isend_irecv`

## Documentation Completeness Check

### Sparse Determination (det) Implementation

| Feature | README | Examples | Docs | Benchmarks | Tests | Status |
|---------|--------|----------|------|------------|-------|--------|
| **Basic Usage** | ✅ | ✅ | ✅ | ✅ | ✅ | Complete |
| **Gradient Support** | ✅ | ✅ | ✅ | ✅ | ✅ | Complete |
| **CPU Backend** | ✅ | ✅ | ✅ | ✅ | ✅ | Complete |
| **CUDA Backend** | ✅ | ✅ | ✅ | ✅ | ✅ | Complete |
| **Batched Matrices** | ✅ | ✅ | ✅ | N/A | ✅ | Complete |
| **Distributed (DSparseTensor)** | ✅ | ✅ | ✅ | N/A | ✅ | Complete |
| **Mathematical Formulas** | ✅ | N/A | ✅ | N/A | N/A | Complete |
| **Performance Benchmarks** | ✅ | N/A | ✅ | ✅ | N/A | Complete |
| **Error Handling** | ✅ | ✅ | ✅ | N/A | N/A | Complete |
| **Numerical Stability Notes** | ✅ | ✅ | ✅ | ✅ | N/A | Complete |

**Summary:**
- ✅ Core functionality: 100% complete (10/10 features)
- ✅ Documentation: 100% complete (README, Examples, Docs)
- ✅ Examples: 8 comprehensive examples covering all use cases
- ✅ Tests: All functionality tested and verified
- ✅ Benchmarks: Performance benchmark script with visualization
- ✅ Error handling: Proper error messages and warnings
- ✅ Numerical stability: Documented overflow/underflow issues

**Files Created/Updated:**
- `README.md`: Added det() to Matrix Operations, Gradient Support, and Performance Tips
- `examples/determinant.py`: 8 examples (basic, gradient, CUDA, batched, distributed, optimization, stability, properties)
- `docs/source/examples.rst`: Dedicated "Determinant with Gradient Support" section with math formulas
- `benchmarks/benchmark_determinant.py`: Performance benchmark with CPU/CUDA comparison
- `torch_sla/sparse_tensor.py`: DetAdjoint class and det() method
- `torch_sla/backends/scipy_backend.py`: scipy_det() function using LU decomposition
- `torch_sla/distributed.py`: det() for DSparseTensor (via `full_tensor()` gather)

**Implementation Details:**
- **Gradient formula**: ∂det(A)/∂A_ij = det(A) · (A⁻¹)_ji (Jacobi's formula)
- **CPU backend**: LU decomposition via SciPy LU (~0.3-0.8ms for n=10-1000)
- **CUDA backend**: torch.linalg.det for forward, torch.linalg.solve for gradient
- **Memory efficiency**: O(1) graph nodes via adjoint method (no iteration history)
- **Supported classes**: SparseTensor, DSparseTensor (via `full_tensor()` gather)
- **Numerical considerations**: 
  - Determinant values overflow for large matrices (det → ±∞ for n > 1000)
  - Singular matrices cause LU decomposition to fail
  - Use float64 for better numerical stability
  - Gradient computation ~100x slower than forward-only (requires n linear solves)

**Performance Summary (from benchmarks/benchmark_determinant.py):**
```
Matrix Size | CPU (Sparse) | CUDA (Dense) | CPU-for-CUDA | Gradient | CUDA/CPU Ratio
------------|--------------|--------------|--------------|----------|----------------
n = 10      | 0.30 ms      | 0.96 ms      | 0.52 ms      | 3.5 ms   | 3.2x SLOWER
n = 100     | 0.30 ms      | 0.27 ms      | 0.54 ms      | 21 ms    | 0.9x (similar)
n = 500     | 0.45 ms      | 1.29 ms      | 0.82 ms      | 154 ms   | 2.9x SLOWER
n = 1000    | 0.71 ms      | 2.51 ms      | 1.20 ms      | 431 ms   | 3.6x SLOWER
```

**Key Findings:**
- ⚠️  **CUDA is SLOWER than CPU for sparse determinants!**
- CPU uses sparse LU (O(nnz^1.5)), CUDA requires dense conversion (O(n²) memory + O(n³) compute)
- CUDA is 1-3.6x slower than CPU across all matrix sizes
- **Recommendation**: Always use `.cpu().det()` for sparse matrices, even on CUDA
- Reason: cuDSS doesn't expose sparse determinant computation
- Gradient computation ~100x slower (requires n linear solves for (A^{-1})^T)
- Determinant values overflow for n > 1000

## Efficiency

- [x] sparse matmul (already implemented: torch_sla/sparse_tensor/matmul.py -- `__matmul__`,
      SparseSparseMatmulFunction with sparse gradients; sparse@dense + sparse@sparse verified)
  - [ ] dedicated cusparse backend -- NOTE: torch.sparse.mm already dispatches to cuSPARSE on
        CUDA, so a separate backend is likely redundant; revisit only if a specific cuSPARSE
        routine (e.g. SpGEMM tuning) is needed
- [ ] sparse linear solve
  - [x] cudss backend
  - [x] torch backend
    - [x] cg
    - [x] bicgstab
    - [x] gmres    (single-node CPU verified vs scipy)
    - [x] minres   (single-node CPU verified; stopping criterion ~15-20% more iters than scipy, tighten later)
    - [ ] verify gmres / minres on CUDA (4070ti) -- run tests/test_pytorch_gmres_minres.py on GPU
  - [x] torch distributed backend  (incl. gmres / fgmres / minres -- multiprocess tests pass)
- [x] sparse eigen vals
- [x] sparse determination

## Documentation & Outreach

- [~] update documentation
  - [x] document pytorch-backend `gmres` / `minres` in README (methods + backend tables)
  - [x] scrub `eigen` from sphinx docs + README (backends/introduction/index/installation/torch_sla/examples.rst + zh/ mirrors + _templates/page.html)
  - [x] document pytorch `gmres` / `minres` in sphinx backend/method tables
  - [x] refresh ROADMAP.md (complex solve marked LANDED; multi-GPU status updated)
  - [ ] sphinx examples narrative + API autodoc pass for gmres/minres (deeper than tables)
  - [ ] cupy deprecation note (cupy -> pytorch+cudss), pending lsqr/lsmr decision
- [ ] make poster

## Backends

- [~] STRUMPACK backend (`backend="strumpack"`) -- portable sparse **direct** solver
      (CPU / CUDA / **ROCm**) via torch-strumpack; fills the AMD-GPU direct-solve gap
      cuDSS (NVIDIA-only) leaves.
  - [x] adapter + autograd (COO->CSR, factor/solve/solve_transpose, adjoint) -- verified
        with a SciPy stand-in (tests/test_strumpack_backend.py); full suite green
  - [x] verify against the **real compiled STRUMPACK** on **ROCm (780M)**: built the HIP
        ext in the rocm/pytorch container (Dockerfile.rocm, gfx1100), solved SuiteSparse
        `bcsstk16` (4884^2 SPD) -> true residual 2.1e-15, relerr vs scipy ~1e-21, multi-RHS ok.
  - [x] complex support -- added `complex<double>` (`factorize_z`/`solve_z`) to
        torch-strumpack's `strumpack_ext.cpp` + dtype dispatch in `_core.py`; torch-sla
        complex adjoint solves `A^H` (conj-transpose) with `conj(u)`. Verified on
        ROCm/780M: SuiteSparse `qc324` (complex-sym) resid 6.7e-14, `mhd1280b` (Hermitian)
        resid 6.1e-14, complex gradcheck PASS. **NOTE: torch-strumpack C++ change lives in
        a local clone (/tmp/ts-src) -- still to land in sparsexlab/torch-strumpack.**
  - [x] GPU-offload behaviour characterised on 780M (added a `STRUMPACK_VERBOSE` env to
        the binding): 3D-Poisson 32^3 stays on CPU (fronts too small, VRAM flat, resid 7e-16);
        48^3 DOES offload to the iGPU -> **780M GPU Hang** (HW exception). So the 780M
        validates STRUMPACK *correctness on CPU* but is NOT a usable GPU offload target
        (matches the hipsparse OOM seen for gmres/minres). Real GPU offload needs the
        4070ti (CUDA build) or a datacenter AMD GPU (gfx90a/942, blind-compiled in Dockerfile).
  - [ ] verify GPU offload on 4070ti (CUDA build) / MI-series
  - [ ] optional `select_backend` auto-pick: AMD GPU + direct -> strumpack

## Backend cleanup

- [x] remove eigen backend (redundant with scipy; was load-bearing for batch_solve cg/bicgstab -> rerouted to pytorch backend). Full test suite green on macor7 CPU.
- [ ] remove cupy backend -- blocked on: (a) gmres/minres now in pytorch [done], (b) decide GPU least-squares lsqr/lsmr (drop or implement LSQR in pytorch backend)
  - DEFERRED until 4070ti is connected: verify gmres/minres on CUDA + decide lsqr/lsmr on real GPU, then remove cupy