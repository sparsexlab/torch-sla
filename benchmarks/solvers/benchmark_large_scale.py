#!/usr/bin/env python3
"""
大规模迭代法性能测试 - 支持增量运行和断点续跑
结果保存到 JSON 文件，每个点完成后立即保存
"""

import torch
import time
import numpy as np
import scipy.sparse as sp
import json
import os
import gc
import argparse
from pathlib import Path

# 添加项目路径
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from torch_sla import spsolve


def generate_poisson_problem(n, device, dtype):
    """生成 2D Poisson 问题"""
    N = n * n
    diag = 4 * np.ones(N)
    off_diag = -np.ones(N - 1)
    off_diag[np.arange(1, N) % n == 0] = 0
    
    A_scipy = sp.diags([diag, off_diag, off_diag, -np.ones(N - n), -np.ones(N - n)],
                       [0, -1, 1, -n, n], format='coo')
    
    indices = torch.tensor(np.vstack([A_scipy.row, A_scipy.col]), dtype=torch.long, device=device)
    values = torch.tensor(A_scipy.data, dtype=dtype, device=device)
    A = torch.sparse_coo_tensor(indices, values, (N, N)).coalesce()
    
    x_true = torch.ones(N, dtype=dtype, device=device)
    b = torch.sparse.mm(A, x_true.unsqueeze(1)).squeeze()
    
    return A, b, x_true


def get_gpu_memory():
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024**2


def load_results(result_file):
    """加载已有结果"""
    if os.path.exists(result_file):
        with open(result_file, 'r') as f:
            return json.load(f)
    return {'results': [], 'completed_dofs': []}


def save_results(result_file, data):
    """保存结果"""
    with open(result_file, 'w') as f:
        json.dump(data, f, indent=2)


def run_benchmark(n, device, dtype, maxiter=20000, tol=1e-6, num_runs=3):
    """运行单个规模的 benchmark"""
    N = n * n
    
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    A, b, x_true = generate_poisson_problem(n, device=device, dtype=dtype)
    
    indices = A.indices()
    row = indices[0]
    col = indices[1]
    val = A.values()
    
    # 预热
    _ = spsolve(val, row, col, (N, N), b, backend='pytorch', method='cg', 
                preconditioner='jacobi', tol=tol, maxiter=maxiter)
    torch.cuda.synchronize()
    
    # 多次运行取平均
    times = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        x = spsolve(val, row, col, (N, N), b, backend='pytorch', method='cg',
                    preconditioner='jacobi', tol=tol, maxiter=maxiter)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    # 计算残差
    r = b - torch.sparse.mm(A, x.unsqueeze(1)).squeeze()
    residual = torch.norm(r).item() / torch.norm(b).item()
    
    mem = get_gpu_memory()
    
    # 清理
    del A, b, x_true, x, indices, row, col, val
    gc.collect()
    torch.cuda.empty_cache()
    
    return {
        'n': n,
        'dof': N,
        'time_ms': avg_time,
        'time_min_ms': min_time,
        'time_max_ms': max_time,
        'memory_mb': mem,
        'residual': residual,
        'tol': tol,
        'maxiter': maxiter,
        'num_runs': num_runs,
        'dtype': str(dtype),
    }


def main():
    parser = argparse.ArgumentParser(description='Large-scale iterative solver benchmark')
    parser.add_argument('--result-file', type=str, 
                        default='benchmarks/results/benchmark_large_scale.json',
                        help='Result file path')
    parser.add_argument('--dtype', type=str, default='float64', choices=['float32', 'float64'])
    parser.add_argument('--maxiter', type=int, default=50000)
    parser.add_argument('--tol', type=float, default=1e-6)
    parser.add_argument('--num-runs', type=int, default=3)
    parser.add_argument('--only-plot', action='store_true', help='Only plot existing results')
    args = parser.parse_args()
    
    result_file = args.result_file
    os.makedirs(os.path.dirname(result_file), exist_ok=True)
    
    dtype = torch.float64 if args.dtype == 'float64' else torch.float32
    device = 'cuda'
    
    # DOF 规模列表 - 从 10K 到 1B
    n_sizes = [
        100,    # 10K
        200,    # 40K
        316,    # 100K
        500,    # 250K
        707,    # 500K
        1000,   # 1M
        1414,   # 2M
        2000,   # 4M
        2828,   # 8M
        4000,   # 16M
        5000,   # 25M
        5657,   # 32M
        6000,   # 36M
        7000,   # 49M
        8000,   # 64M
        9000,   # 81M
        10000,  # 100M
        11000,  # 121M
        12000,  # 144M
        13000,  # 169M
        14000,  # 196M
        15000,  # 225M
        16000,  # 256M
        18000,  # 324M
        20000,  # 400M
        22000,  # 484M
        25000,  # 625M
        28000,  # 784M
        30000,  # 900M
        31623,  # 1000M (1B)
    ]
    
    # 加载已有结果
    data = load_results(result_file)
    completed_dofs = set(data.get('completed_dofs', []))
    
    if args.only_plot:
        print("Only plotting existing results...")
        plot_results(data, result_file.replace('.json', '.png'))
        return
    
    print('=== PyTorch CG 大规模增量测试 ===')
    print(f'结果文件: {result_file}')
    print(f'已完成: {len(completed_dofs)} 个规模')
    print()
    print(f"{'DOF':>15} {'Time (ms)':>12} {'Time/DOF':>12} {'Memory (MB)':>12} {'Residual':>12}")
    print('-' * 70)
    
    # 打印已完成的结果
    for r in data.get('results', []):
        time_per_dof = r['time_ms'] / r['dof'] * 1000
        print(f"{r['dof']:>15,} {r['time_ms']:>12.1f} {time_per_dof:>10.2f} μs {r['memory_mb']:>12.1f} {r['residual']:>12.2e} [cached]")
    
    # 继续测试未完成的规模
    for n in n_sizes:
        N = n * n
        
        if N in completed_dofs:
            continue
        
        try:
            result = run_benchmark(n, device, dtype, args.maxiter, args.tol, args.num_runs)
            
            time_per_dof = result['time_ms'] / result['dof'] * 1000
            print(f"{result['dof']:>15,} {result['time_ms']:>12.1f} {time_per_dof:>10.2f} μs {result['memory_mb']:>12.1f} {result['residual']:>12.2e}")
            
            # 保存结果
            data['results'].append(result)
            data['completed_dofs'].append(N)
            save_results(result_file, data)
            
        except torch.cuda.OutOfMemoryError:
            print(f"{N:>15,} OOM - stopping")
            break
        except Exception as e:
            print(f"{N:>15,} Error: {e}")
            break
    
    print()
    print(f'结果已保存到: {result_file}')
    
    # 生成图表
    plot_results(data, result_file.replace('.json', '.png'))


def plot_results(data, output_file):
    """生成性能图表"""
    import matplotlib.pyplot as plt
    
    results = sorted(data.get('results', []), key=lambda x: x['dof'])
    if not results:
        print("No results to plot")
        return
    
    dofs = [r['dof'] for r in results]
    times = [r['time_ms'] for r in results]
    mems = [r['memory_mb'] for r in results]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 时间 vs DOF
    ax1 = axes[0]
    ax1.loglog(dofs, times, 'b-o', linewidth=2, markersize=6, label='PyTorch CG+Jacobi')
    ax1.set_xlabel('Degrees of Freedom (DOF)', fontsize=12)
    ax1.set_ylabel('Time (ms)', fontsize=12)
    ax1.set_title('Solver Performance', fontsize=14)
    ax1.grid(True, which='both', linestyle='--', alpha=0.5)
    ax1.legend()
    
    # 计算复杂度斜率
    import math
    if len(dofs) >= 2:
        log_dofs = [math.log10(d) for d in dofs]
        log_times = [math.log10(t) for t in times]
        n = len(log_dofs)
        slope = (n * sum(x*y for x,y in zip(log_dofs, log_times)) - sum(log_dofs)*sum(log_times)) / \
                (n * sum(x**2 for x in log_dofs) - sum(log_dofs)**2)
        ax1.text(0.05, 0.95, f'Complexity: O(n^{slope:.2f})', transform=ax1.transAxes, 
                 fontsize=10, verticalalignment='top', 
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 内存 vs DOF
    ax2 = axes[1]
    ax2.loglog(dofs, mems, 'r-s', linewidth=2, markersize=6, label='Peak Memory')
    ax2.set_xlabel('Degrees of Freedom (DOF)', fontsize=12)
    ax2.set_ylabel('Memory (MB)', fontsize=12)
    ax2.set_title('Memory Usage', fontsize=14)
    ax2.grid(True, which='both', linestyle='--', alpha=0.5)
    ax2.legend()
    
    # 内存复杂度
    if len(dofs) >= 2:
        log_mems = [math.log10(m) for m in mems]
        slope = (n * sum(x*y for x,y in zip(log_dofs, log_mems)) - sum(log_dofs)*sum(log_mems)) / \
                (n * sum(x**2 for x in log_dofs) - sum(log_dofs)**2)
        ax2.text(0.05, 0.95, f'Complexity: O(n^{slope:.2f})', transform=ax2.transAxes,
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.savefig(output_file.replace('.png', '.pdf'), bbox_inches='tight')
    print(f'图表已保存到: {output_file}')


if __name__ == '__main__':
    main()


