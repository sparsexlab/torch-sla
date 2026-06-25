#!/usr/bin/env python
"""Scaling benchmark: connected_components (A.connected_components()).

Emits ``connected_components_scaling.png`` (the name the docs reference).

Run::

    python benchmarks/scaling/ops/connected_components.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import OpSpec, SWEEP_CHEAP, SWEEP_CHEAP_QUICK, main_for  # noqa: E402
from setups import setup_connected_components  # noqa: E402

SPEC = OpSpec(
    name="connected_components",
    setup=setup_connected_components,
    backend="torch (pure)",
    png_name="connected_components",
    reps=3,
    sweep=SWEEP_CHEAP,
    sweep_quick=SWEEP_CHEAP_QUICK,
)

main = main_for(SPEC)

if __name__ == "__main__":
    main()
