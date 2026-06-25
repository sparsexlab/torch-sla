#!/usr/bin/env python
"""Scaling benchmark: linear solve via STRUMPACK direct solver.

Skipped gracefully if STRUMPACK is not available.

Run::

    python benchmarks/scaling/ops/solve_strumpack.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import OpSpec, SWEEP_MID, SWEEP_MID_QUICK, main_for  # noqa: E402
from setups import (  # noqa: E402
    setup_solve_strumpack, verify_solve, is_strumpack_available,
)

SPEC = OpSpec(
    name="linear solve (STRUMPACK)",
    setup=setup_solve_strumpack,
    backend="strumpack",
    png_name="strumpack",
    reps=2,
    avail=lambda dev: is_strumpack_available(),
    sweep=SWEEP_MID,
    sweep_quick=SWEEP_MID_QUICK,
    verify=verify_solve("strumpack"),
)

main = main_for(SPEC)

if __name__ == "__main__":
    main()
