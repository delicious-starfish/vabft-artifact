"""
测试 e_max 随矩阵规模的变化规律

理论预测：FP32 的 e_max 应随累加深度增长，拟合对 log(KN) 和 sqrt(KN) 的关系
"""

import os
import torch
import torch_npu
import math
import numpy as np
from datetime import datetime
from itertools import product

device_npu = torch.device("npu:0" if torch_npu.npu.is_available() else "cpu")
print(f"Using device: {device_npu}")


def estimate_emax_relative(M, K, N, dtype, num_trials=100000):
    """
    估计相对误差形式的 e_max
    使用正数矩阵防止正负相消
    """
    max_relative_error = 0.0

    for _ in range(num_trials):
        # 使用正数矩阵 |N(1,1)|
        A = torch.abs(torch.randn(M, K, device=device_npu, dtype=dtype) + 1.0)
        B = torch.abs(torch.randn(K, N, device=device_npu, dtype=dtype) + 1.0)

        # 路径1: A @ B 再求行和
        C = torch.matmul(A, B)
        ones = torch.ones(N, 1, device=device_npu, dtype=dtype)
        path1 = torch.matmul(C, ones).squeeze(-1)

        # 路径2: 先求B行和，再乘
        B_rowsum = torch.matmul(B, ones)
        path2 = torch.matmul(A, B_rowsum).squeeze(-1)

        # 计算相对误差
        error = torch.abs(path1 - path2)
        relative_error = (error / torch.abs(path1)).max().item()
        max_relative_error = max(max_relative_error, relative_error)

    return max_relative_error


def main():
    print("=" * 80)
    print(f"e_max Scaling Test - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # M, K, N 从 {128, 256, 512, ..., 8192} 中选取
    size_choices = [128, 256, 512, 1024, 2048, 4096, 8192]
    
    # 生成组合: (M, K, N)，确保 KN 有较好的分布
    sizes = []
    for K in size_choices:
        for N in size_choices:
            M = 128  # 固定 M，简化实验
            sizes.append((M, K, N))
    
    # 只测试 float32
    dtype = torch.float32
    u = 2**-24
    num_trials = 100000  # 增加实验次数

    print(f"\nTesting {dtype}, unit roundoff u = {u:.2e}")
    print(f"Number of configurations: {len(sizes)}")
    print(f"Trials per config: {num_trials}")
    print(f"{'M':<6} {'K':<6} {'N':<6} {'KN':<12} {'log(KN)':<10} {'sqrt(KN)':<10} {'e_max':<15}")
    print("-" * 90)

    results = []
    for M, K, N in sizes:
        emax = estimate_emax_relative(M, K, N, dtype, num_trials)
        kn = K * N
        log_kn = math.log(kn)
        sqrt_kn = math.sqrt(kn)
        
        results.append({
            'M': M, 'K': K, 'N': N,
            'KN': kn, 'log_KN': log_kn, 'sqrt_KN': sqrt_kn,
            'emax': emax
        })
        
        print(f"{M:<6} {K:<6} {N:<6} {kn:<12} {log_kn:<10.2f} {sqrt_kn:<10.1f} {emax:<15.6e}")

    # 回归分析
    print("\n" + "=" * 80)
    print("Regression Analysis")
    print("=" * 80)

    kn_values = np.array([r['KN'] for r in results])
    log_kn_values = np.array([r['log_KN'] for r in results])
    sqrt_kn_values = np.array([r['sqrt_KN'] for r in results])
    emax_values = np.array([r['emax'] for r in results])

    # 对 log(KN) 的线性回归
    coef_log, intercept_log = np.polyfit(log_kn_values, emax_values, 1)
    
    # 计算 R^2
    y_pred_log = coef_log * log_kn_values + intercept_log
    ss_res_log = np.sum((emax_values - y_pred_log) ** 2)
    ss_tot_log = np.sum((emax_values - np.mean(emax_values)) ** 2)
    r2_log = 1 - (ss_res_log / ss_tot_log)

    # 对 sqrt(KN) 的线性回归
    coef_sqrt, intercept_sqrt = np.polyfit(sqrt_kn_values, emax_values, 1)
    y_pred_sqrt = coef_sqrt * sqrt_kn_values + intercept_sqrt
    ss_res_sqrt = np.sum((emax_values - y_pred_sqrt) ** 2)
    ss_tot_sqrt = np.sum((emax_values - np.mean(emax_values)) ** 2)
    r2_sqrt = 1 - (ss_res_sqrt / ss_tot_sqrt)

    print(f"\n模型: e_max = a * log(KN) + b")
    print(f"  回归系数 a (log): {coef_log:.6e}")
    print(f"  截距 b (log):     {intercept_log:.6e}")
    print(f"  R^2:              {r2_log:.6f}")

    print(f"\n模型: e_max = a * sqrt(KN) + b")
    print(f"  回归系数 a (sqrt): {coef_sqrt:.6e}")
    print(f"  截距 b (sqrt):     {intercept_sqrt:.6e}")
    print(f"  R^2:              {r2_sqrt:.6f}")

    # 比较
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    if r2_log > r2_sqrt:
        print(f"→ log(KN) 模型拟合更好 (R^2 = {r2_log:.6f} vs {r2_sqrt:.6f})")
    else:
        print(f"→ sqrt(KN) 模型拟合更好 (R^2 = {r2_sqrt:.6f} vs {r2_log:.6f})")

    # 保存结果
    output_file = os.path.join(os.environ.get('VABFT_ROOT', '.'), 'emax_scaling_results.txt')
    with open(output_file, 'w') as f:
        f.write(f"e_max Scaling Test Results - {datetime.now()}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Regression Results:\n")
        f.write(f"  e_max = {coef_log:.6e} * log(KN) + {intercept_log:.6e}, R^2 = {r2_log:.6f}\n")
        f.write(f"  e_max = {coef_sqrt:.6e} * sqrt(KN) + {intercept_sqrt:.6e}, R^2 = {r2_sqrt:.6f}\n\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'M':<6} {'K':<6} {'N':<6} {'KN':<12} {'log(KN)':<10} {'sqrt(KN)':<10} {'e_max':<15}\n")
        for r in results:
            f.write(f"{r['M']:<6} {r['K']:<6} {r['N']:<6} {r['KN']:<12} {r['log_KN']:<10.2f} {r['sqrt_KN']:<10.1f} {r['emax']:<15.6e}\n")

    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
