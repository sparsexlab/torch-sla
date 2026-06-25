#!/usr/bin/env python
"""Scaling benchmark: transpose (A^T).

Run::

    python benchmarks/scaling/ops/transpose.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import OpSpec, SWEEP_CHEAP, SWEEP_CHEAP_QUICK, main_for  # noqa: E402
from setups import setup_transpose  # noqa: E402

SPEC = OpSpec(
    name="transpose (A^T)",
    setup=setup_transpose,
    backend="torch",
    png_name="transpose",
    reps=5,
    sweep=SWEEP_CHEAP,
    sweep_quick=SWEEP_CHEAP_QUICK,
)

main = main_for(SPEC)

if __name__ == "__main__":
    main()
