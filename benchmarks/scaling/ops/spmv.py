#!/usr/bin/env python
"""Scaling benchmark: sparse matrix-vector product (A @ x).

Run::

    python benchmarks/scaling/ops/spmv.py
    python benchmarks/scaling/ops/spmv.py --device cuda
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import OpSpec, SWEEP_CHEAP, SWEEP_CHEAP_QUICK, main_for  # noqa: E402
from setups import setup_spmv  # noqa: E402

SPEC = OpSpec(
    name="sparse matvec (A @ x)",
    setup=setup_spmv,
    backend="torch",
    png_name="spmv",
    reps=5,
    sweep=SWEEP_CHEAP,
    sweep_quick=SWEEP_CHEAP_QUICK,
)

main = main_for(SPEC)

if __name__ == "__main__":
    main()
