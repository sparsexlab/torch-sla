# LOBPCG examples

User-facing perf demos for the LOBPCG eigensolver.

## `convergence_benchmark.py`

Demonstrates the current `_lobpcg_core` against a deliberately weakened
block-steepest-descent reference on a clustered SPD spectrum, so you can
see how much the 3-block `[X | R | P]` subspace + conjugate direction
buy versus the naive update.

```bash
python examples/lobpcg/convergence_benchmark.py
# OLD (block-steepest-descent): 111 matvecs / max eig err 8.8e-4
# NEW (LOBPCG, this codebase):  20 matvecs / max eig err 3.6e-11
# Plot: assets/examples/lobpcg/convergence.png
```

For the **algorithmic-fix verification benches** (v1 pre-fix vs v2 this
code, against `torch.lobpcg` baseline; multi-device sweep; distributed
collective bench) see `tests/lobpcg/` — those are regression material
rather than user-facing examples.
