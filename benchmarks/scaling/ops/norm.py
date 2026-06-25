#!/usr/bin/env python
"""Scaling benchmark: Frobenius norm (A.norm('fro')).

Run::

    python benchmarks/scaling/ops/norm.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import OpSpec, SWEEP_CHEAP, SWEEP_CHEAP_QUICK, main_for  # noqa: E402
from setups import setup_norm  # noqa: E402

SPEC = OpSpec(
    name="Frobenius norm",
    setup=setup_norm,
    backend="torch",
    png_name="norm",
    reps=5,
    sweep=SWEEP_CHEAP,
    sweep_quick=SWEEP_CHEAP_QUICK,
)

main = main_for(SPEC)

if __name__ == "__main__":
    main()
