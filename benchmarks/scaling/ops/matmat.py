#!/usr/bin/env python
"""Scaling benchmark: sparse-sparse product (A @ A).

Run::

    python benchmarks/scaling/ops/matmat.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import OpSpec, SWEEP_CHEAP, SWEEP_CHEAP_QUICK, main_for  # noqa: E402
from setups import setup_matmat  # noqa: E402

SPEC = OpSpec(
    name="sparse matmat (A @ A)",
    setup=setup_matmat,
    backend="torch",
    png_name="matmat",
    reps=3,
    sweep=SWEEP_CHEAP,
    sweep_quick=SWEEP_CHEAP_QUICK,
)

main = main_for(SPEC)

if __name__ == "__main__":
    main()
