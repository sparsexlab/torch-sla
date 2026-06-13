#!/usr/bin/env python
"""
Distributed Sparse Solve Benchmark

Benchmarks distributed solve(D, b_dt) on CPU (Gloo) and CUDA (NCCL).

Usage:
    # CPU with 4 processes
    torchrun --standalone --nproc_per_node=4 benchmark_distributed.py --device cpu
    
    # CUDA with 4 GPUs  
    torchrun --standalone --nproc_per_node=4 benchmark_distributed.py --device cuda
    
    # Generate plots
    python benchmark_distributed.py --only-plot
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

import torch

IN_DISTRIBUTED = 'RANK' in os.environ


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--dofs', type=str, default=None,
                       help='Comma-separated DOFs. Default: auto-scale')
    parser.add_argument('--maxiter', type=int, default=5000)
    parser.add_argument('--atol', type=float, default=1e-8)
    parser.add_argument('--output-dir', type=str, default='results/benchmark_distributed')
    parser.add_argument('--only-plot', action='store_true')
    parser.add_argument('--no-title', action='store_true')
    return parser.parse_args()


def create_poisson_2d(n: int, device: str = 'cpu'):
    """Create 2D Poisson matrix (5-point stencil) - vectorized version."""
    grid = int(n ** 0.5)
    actual_n = grid * grid
    
    # Create all node indices
    idx = torch.arange(actual_n, dtype=torch.int64)
    i = idx // grid  # row in grid
    j = idx % grid   # col in grid
    
    # Diagonal entries (all nodes)
    row_list = [idx]
    col_list = [idx]
    val_list = [torch.full((actual_n,), 4.0, dtype=torch.float64)]
    
    # Left neighbor (j > 0)
    mask = j > 0
    row_list.append(idx[mask])
    col_list.append(idx[mask] - 1)
    val_list.append(torch.full((mask.sum().item(),), -1.0, dtype=torch.float64))
    
    # Right neighbor (j < grid-1)
    mask = j < grid - 1
    row_list.append(idx[mask])
    col_list.append(idx[mask] + 1)
    val_list.append(torch.full((mask.sum().item(),), -1.0, dtype=torch.float64))
    
    # Top neighbor (i > 0)
    mask = i > 0
    row_list.append(idx[mask])
    col_list.append(idx[mask] - grid)
    val_list.append(torch.full((mask.sum().item(),), -1.0, dtype=torch.float64))
    
    # Bottom neighbor (i < grid-1)
    mask = i < grid - 1
    row_list.append(idx[mask])
    col_list.append(idx[mask] + grid)
    val_list.append(torch.full((mask.sum().item(),), -1.0, dtype=torch.float64))
    
    row = torch.cat(row_list)
    col = torch.cat(col_list)
    val = torch.cat(val_list)
    
    return val, row, col, (actual_n, actual_n), actual_n


def get_memory_mb():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0


def benchmark_distributed(args):
    """Run distributed benchmark."""
    import torch.distributed as dist
    # torch ≥2.2: device_mesh promoted to top level. torch 2.1.x: still under _tensor.
    try:
        from torch.distributed.device_mesh import init_device_mesh
    except ImportError:
        from torch.distributed._tensor.device_mesh import init_device_mesh

    from torch_sla import SparseTensor, DSparseTensor, solve, SolverConfig

    # Choose backend based on device
    backend = 'nccl' if args.device == 'cuda' else 'gloo'
    dist.init_process_group(backend=backend)
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    if args.device == 'cuda':
        local_rank = rank % torch.cuda.device_count()
        torch.cuda.set_device(local_rank)
        device = f'cuda:{local_rank}'
    else:
        device = 'cpu'
    
    if rank == 0:
        print("=" * 70)
        print(f"Distributed Solve Benchmark")
        print(f"  Backend: {backend}")
        print(f"  World size: {world_size}")
        print(f"  Device: {args.device}")
        if args.device == 'cuda':
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"  GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
        print("=" * 70)
    
    # Auto-scale DOFs based on device
    if args.dofs:
        dofs = [int(d) for d in args.dofs.split(',')]
    elif args.device == 'cuda':
        # CUDA: scale up to 10M DOF (tested stable)
        dofs = [10000, 100000, 1000000, 4000000, 10000000]
    else:
        # CPU: smaller scales
        dofs = [10000, 100000, 500000, 1000000]
    
    total_dofs = len(dofs)
    if rank == 0:
        print(f"\nTesting {total_dofs} problem sizes: {dofs}")
        print("-" * 70)
    
    results = []
    cumulative_time = 0.0
    
    mesh = init_device_mesh(args.device, (world_size,))

    for idx, target_dof in enumerate(dofs):
        # Build the global SparseTensor on CPU, ship to device.
        val, row, col, shape, actual_dof = create_poisson_2d(target_dof, device='cpu')

        try:
            if args.device == 'cuda':
                torch.cuda.reset_peak_memory_stats()

            A = SparseTensor(val, row, col, shape).to(device)

            # Row-shard across the mesh -- one Partition per rank.
            D = DSparseTensor.partition(A, mesh, partition_method="simple")

            # RHS as a DTensor[Shard(0)] sized to owned rows.
            b_global = torch.ones(shape[0], dtype=torch.float64, device=device)
            b_dt = D.scatter(b_global)

            # Warmup
            dist.barrier()
            try:
                with SolverConfig(method="cg", atol=1e-4, rtol=0.0,
                                   maxiter=50):
                    _ = solve(D, b_dt)
            except Exception:
                pass
            dist.barrier()

            if args.device == 'cuda':
                torch.cuda.synchronize()

            # Benchmark
            dist.barrier()
            start = time.perf_counter()

            with SolverConfig(method="cg", atol=args.atol, rtol=0.0,
                               maxiter=args.maxiter):
                x_dt = solve(D, b_dt)

            if args.device == 'cuda':
                torch.cuda.synchronize()
            dist.barrier()

            elapsed = time.perf_counter() - start

            # Per-GPU peak memory (max across ranks).
            if args.device == 'cuda':
                local_peak_mem = torch.cuda.max_memory_allocated() / 1024**3
                mem_tensor = torch.tensor([local_peak_mem], device=device)
                all_mems = [torch.zeros(1, device=device) for _ in range(world_size)]
                dist.all_gather(all_mems, mem_tensor)
                all_mems = [m.item() for m in all_mems]
                peak_mem = max(all_mems)
            else:
                local_peak_mem = 0
                all_mems = [0] * world_size
                peak_mem = 0

            # Residual via public ops only: r_dt = b_dt - D @ x_dt; gather.
            r_dt = b_dt - D @ x_dt
            # ``full_tensor()`` is torch 2.3+; fall back to redistribute
            # for older releases (e.g. the torch 2.1 on autodl).
            if hasattr(r_dt, "full_tensor"):
                r_full = r_dt.full_tensor()
            else:
                try:
                    from torch.distributed.tensor import Replicate
                except ImportError:
                    from torch.distributed._tensor import Replicate
                r_full = r_dt.redistribute(mesh, [Replicate()]).to_local()
            global_res_sq = (r_full ** 2).sum()
            
            residual = global_res_sq.sqrt().item()
            
            cumulative_time += elapsed
            
            if rank == 0:
                progress = (idx + 1) / total_dofs * 100
                if peak_mem > 0:
                    mem_str = f", mem={peak_mem:.2f}GB/GPU"
                else:
                    mem_str = ""
                # Estimate remaining time
                avg_time_per_dof = cumulative_time / (idx + 1)
                remaining = (total_dofs - idx - 1) * avg_time_per_dof
                eta_str = f", ETA={remaining:.0f}s" if remaining > 0 else ""
                print(f"[{idx+1}/{total_dofs}] DOF={actual_dof:>10,}, time={elapsed:.3f}s, res={residual:.2e}{mem_str}{eta_str}")
            
            results.append({
                'dof': actual_dof,
                'time': elapsed,
                'residual': residual,
                'memory_gb': peak_mem,
                'world_size': world_size,
                'device': args.device
            })
            
            # Clean up
            del A, D, x_dt, b_dt, b_global, r_dt
            if args.device == 'cuda':
                torch.cuda.empty_cache()
            
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            import traceback
            if rank == 0:
                print(f"DOF={actual_dof:>10,}: Error - {type(e).__name__}: {e}")
                traceback.print_exc()
            
            if args.device == 'cuda':
                torch.cuda.empty_cache()
            break
    
    # Save results
    if rank == 0:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        result_file = output_dir / f'results_{args.device}_p{world_size}.json'
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {result_file}")
    
    dist.destroy_process_group()


def plot_results(args):
    """Generate comparison plots - only P=4, with time, memory, and error."""
    import matplotlib.pyplot as plt
    
    output_dir = Path(args.output_dir)
    
    # Load only P=4 results
    cpu_file = output_dir / 'results_cpu_p4.json'
    cuda_file = output_dir / 'results_cuda_p4.json'
    
    results = {}
    if cpu_file.exists():
        with open(cpu_file) as f:
            results['cpu'] = json.load(f)
    if cuda_file.exists():
        with open(cuda_file) as f:
            results['cuda'] = json.load(f)
    
    if not results:
        print("No P=4 result files found.")
        return
    
    # Create 3-panel figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    
    colors = {'cpu': '#2E86AB', 'cuda': '#E94F37'}
    labels = {'cpu': 'CPU (Gloo, 4 proc)', 'cuda': 'CUDA (NCCL, 4 GPU)'}
    
    # Panel 1: Time
    ax = axes[0]
    for device, data in results.items():
        dofs = [d['dof'] for d in data]
        times = [d['time'] for d in data]
        ax.loglog(dofs, times, 'o-', color=colors[device], label=labels[device],
                  linewidth=2.5, markersize=9, markeredgecolor='white', markeredgewidth=1)
    
    ax.set_xlabel('DOF', fontsize=12)
    ax.set_ylabel('Time (s)', fontsize=12)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3, which='both')
    ax.set_title('Solve Time', fontsize=13, fontweight='bold')
    
    # Panel 2: Memory (GPU and estimated CPU RAM)
    ax = axes[1]
    
    # GPU memory
    if 'cuda' in results:
        data = results['cuda']
        dofs = [d['dof'] for d in data if d.get('memory_gb', 0) > 0]
        mems = [d['memory_gb'] for d in data if d.get('memory_gb', 0) > 0]
        
        if dofs:
            ax.loglog(dofs, mems, 's-', color=colors['cuda'], label='GPU Memory/card',
                     linewidth=2.5, markersize=9, markeredgecolor='white', markeredgewidth=1)
    
    # Estimate CPU RAM: ~500 bytes/DOF for sparse matrix + vectors
    # For distributed: each process has ~n/P DOF + halo
    if 'cpu' in results:
        data = results['cpu']
        dofs = [d['dof'] for d in data]
        # Estimate: (5*nnz + 3*n) * 8 bytes / P, nnz ≈ 5*n for 2D Poisson
        # ≈ (25n + 3n) * 8 / 4 = 56n bytes/proc = 56 bytes/DOF per proc
        mems = [d * 56 / 4 / 1e9 for d in dofs]  # GB per process
        ax.loglog(dofs, mems, 'o--', color=colors['cpu'], label='CPU RAM/proc (est)',
                 linewidth=2, markersize=8, markeredgecolor='white', markeredgewidth=1)
    
    ax.set_xlabel('DOF', fontsize=12)
    ax.set_ylabel('Memory per Process (GB)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_title('Memory Usage', fontsize=13, fontweight='bold')
    
    # Panel 3: Residual (Error)
    ax = axes[2]
    markers = {'cuda': 's', 'cpu': 'o'}
    # Add slight offset to avoid overlapping
    offsets = {'cuda': 1.0, 'cpu': 1.02}
    for device, data in results.items():
        dofs = [d['dof'] * offsets[device] for d in data]  # slight x-offset
        residuals = [d['residual'] for d in data]
        ax.loglog(dofs, residuals, f'{markers[device]}-', color=colors[device], label=labels[device],
                  linewidth=2.5, markersize=9, markeredgecolor='white', markeredgewidth=1)
    
    ax.set_xlabel('DOF', fontsize=12)
    ax.set_ylabel('Residual ||Ax - b||', fontsize=12)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3, which='both')
    ax.set_title('Convergence (atol=1e-8)', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    
    for fmt in ['png', 'pdf']:
        fig.savefig(output_dir / f'distributed_benchmark.{fmt}', dpi=150, bbox_inches='tight')
    
    print(f"Plots saved to {output_dir}")
    plt.close()


def main():
    args = get_args()
    
    if args.only_plot:
        plot_results(args)
        return
    
    if IN_DISTRIBUTED:
        benchmark_distributed(args)
    else:
        print("Run with torchrun:")
        print(f"  torchrun --standalone --nproc_per_node=4 {sys.argv[0]} --device cpu")
        print(f"  torchrun --standalone --nproc_per_node=4 {sys.argv[0]} --device cuda")


if __name__ == '__main__':
    main()
