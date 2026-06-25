#!/usr/bin/env python
"""Run EVERY per-op scaling benchmark and emit each op's own scaling-curve PNG.

This is a thin loop over the per-op modules in this directory -- each module
declares an :class:`~harness.OpSpec` named ``SPEC``. Each op is *also* runnable
alone (e.g. ``python benchmarks/scaling/ops/solve_cg.py``); this runner just
sweeps them all, reusing the same harness so behaviour is identical.

Run::

    python benchmarks/scaling/ops/run_all.py
    python benchmarks/scaling/ops/run_all.py --device cuda
    python benchmarks/scaling/ops/run_all.py --quick --time-cap 8
    python benchmarks/scaling/ops/run_all.py --ops spmv,solve_cg,eigsh
"""
import argparse
import importlib.util
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from harness import add_common_args, run_and_plot  # noqa: E402


def _load_op(name):
    """Load a per-op module by file path (robust to any cwd / sys.path churn
    a previous op's sweep may have caused)."""
    path = _HERE / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_op_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# op module name -> imported lazily so a single broken op can't kill the rest.
# Ordered cheap -> expensive.
OP_MODULES = [
    "spmv", "matmat", "norm", "transpose", "connected_components",
    "solve_cg", "logdet", "nonlinear_solve", "solve_batch",
    "solve_strumpack", "solve_pyamg", "solve_cudss",
    "lu", "det", "det_backward",
    "eigsh", "svd", "condition_number",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--ops", default="all",
                    help="comma list of op module names (see OP_MODULES) or 'all'")
    args = ap.parse_args()

    names = OP_MODULES if args.ops == "all" else args.ops.split(",")

    t0 = time.perf_counter()
    done, skipped, failed = [], [], []
    for name in names:
        try:
            mod = _load_op(name)
            spec = mod.SPEC
        except Exception as e:  # noqa: BLE001
            print(f"[error] could not load op module {name!r}: "
                  f"{type(e).__name__}: {e}", flush=True)
            failed.append(name)
            continue
        try:
            path = run_and_plot(spec, args)
        except Exception as e:  # noqa: BLE001 -- one bad op must not abort the rest
            print(f"[error] {name} crashed: {type(e).__name__}: {e}", flush=True)
            failed.append(name)
            continue
        if path is None:
            skipped.append(name)
        else:
            done.append((name, path))

    print("\n" + "=" * 70)
    print(f"run_all done in {time.perf_counter() - t0:.1f}s")
    print("=" * 70)
    print(f"generated curves ({len(done)}):")
    for name, path in done:
        print(f"  {name:<22} -> {path}")
    if skipped:
        print(f"skipped ({len(skipped)}): {', '.join(skipped)}")
    if failed:
        print(f"failed ({len(failed)}): {', '.join(failed)}")


if __name__ == "__main__":
    main()
