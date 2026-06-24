
import sys, json, time, warnings, gc, importlib.util
warnings.filterwarnings("ignore")
sys.path.insert(0, '/home/walkerchi/Code/torch-sla')
import torch
import torch_sla.datasets as d
from torch_sla import SparseTensor, spsolve, DetConfig
# load the benchmark module by path (benchmarks/ is not a package)
_spec = importlib.util.spec_from_file_location("_bench", '/home/walkerchi/Code/torch-sla/benchmarks/benchmark_all_ops_scaling.py')
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)

op = sys.argv[1]; side = int(sys.argv[2]); device = sys.argv[3]
try:
    A, dof, nnz = B.build(side, device)
    run = B.OPS[op]["setup"](A, dof, device)
    t0 = time.perf_counter()
    run()
    if device == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter() - t0
    try:
        import psutil, os
        peak = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    except Exception:
        peak = float("nan")
    print("OK " + json.dumps(dict(dof=dof, nnz=nnz, time_s=t, peak_mb=peak)))
except Exception as e:
    print("ERR " + type(e).__name__ + ": " + str(e)[:200])
