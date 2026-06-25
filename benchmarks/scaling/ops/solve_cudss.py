#!/usr/bin/env python
"""Scaling benchmark: linear solve via NVIDIA cuDSS (CUDA only).

Skipped gracefully unless device == 'cuda' and cuDSS is available.

Run::

    python benchmarks/scaling/ops/solve_cudss.py --device cuda
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import OpSpec, SWEEP_MID, SWEEP_MID_QUICK, main_for  # noqa: E402
from setups import (  # noqa: E402
    setup_solve_cudss, verify_solve, is_cudss_available,
)

SPEC = OpSpec(
    name="linear solve (cuDSS)",
    setup=setup_solve_cudss,
    backend="cudss",
    png_name="cudss",
    reps=2,
    avail=lambda dev: dev == "cuda" and is_cudss_available(),
    sweep=SWEEP_MID,
    sweep_quick=SWEEP_MID_QUICK,
    verify=verify_solve("cudss"),
)

main = main_for(SPEC)

if __name__ == "__main__":
    main()
