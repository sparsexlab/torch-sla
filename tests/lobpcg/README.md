# LOBPCG fix verification benches

Internal benches verifying the PR #45 LOBPCG fix series (residual-norm
convergence + LAPACK QR re-orthonormalisation, with MPS CPU-roundtrip
guards). Not collected by pytest — they're driver scripts that produce
artifacts, not assertions. Numerical assertions live in
`tests/test_lobpcg_eigsh.py` and `tests/test_dsparse_eigsh_multiprocess.py`.

| script | what it does |
| --- | --- |
| `bench_v1_vs_v2.py` | A/B v1 (pre-fix) / v1.5 (conv. fix only) / v2 (this PR) / `torch.lobpcg` on banded SPD. Prints a table; no plot. |
| `bench_multi_device.py` | Same sweep across CPU / MPS (if available) / CUDA (if available). Saves raw data to `data/<devs>.json`. Used as the source of the headline plot. |
| `bench_distributed.py` | Single-GPU multi-rank A/B (gloo or NCCL). Verifies the QR-swap win compounds with collective overhead. |
| `merge_plots.py` | Loads all `data/*.json` and renders the combined `assets/comparison_all.png`. Run after collecting on each device. |
| `data/cpu_mps.json` | Mac M4 run (CPU + MPS). |
| `data/cpu_cuda.json` | walker-4070ti run (CPU + CUDA). |
| `assets/comparison_all.png` | Headline figure referenced in PR #45 description. |

## Refresh the headline plot

```bash
# 1. collect per machine (writes data/<devs>.json)
python tests/lobpcg/bench_multi_device.py            # Mac    -> cpu_mps.json
ssh box-with-cuda 'python tests/lobpcg/bench_multi_device.py'
scp .../cpu_cuda.json tests/lobpcg/data/

# 2. render combined plot
python tests/lobpcg/merge_plots.py
# -> tests/lobpcg/assets/comparison_all.png
```

`INCLUDE_MPS=1 python tests/lobpcg/merge_plots.py` puts the MPS column
back in the plot (off by default per the "MPS not recommended" caveat
in `_lobpcg_core`).
