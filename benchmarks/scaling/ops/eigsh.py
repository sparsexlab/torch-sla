#!/usr/bin/env python
"""Scaling benchmark: eigsh (k=6 smallest algebraic eigenvalues).

Emits ``eigsh_scaling.png`` (the name the docs reference).

Run::

    python benchmarks/scaling/ops/eigsh.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import OpSpec, SWEEP_EIG, SWEEP_EIG_QUICK, main_for  # noqa: E402
from setups import setup_eigsh, verify_eigsh  # noqa: E402

SPEC = OpSpec(
    name="eigsh (smallest-k)",
    setup=setup_eigsh,
    backend="lobpcg",
    png_name="eigsh",
    reps=2,
    sweep=SWEEP_EIG,
    sweep_quick=SWEEP_EIG_QUICK,
    verify=verify_eigsh,
    verify_ok=lambda v: v > 0,  # smallest eigenvalue of SPD Laplacian must be > 0
)

main = main_for(SPEC)

if __name__ == "__main__":
    main()
