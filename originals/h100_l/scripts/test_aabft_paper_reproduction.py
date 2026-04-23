"""
A-ABFT vs V-ABFT 对比测试 - 复现A-ABFT论文实验设置

根据A-ABFT论文(DSN 2014)的Table II/III/IV:
- 测量的是每个结果元素c_{i,j}的内积舍入误差
- 比较实际舍入误差 vs A-ABFT门限 vs SEA-ABFT门限

A-ABFT内积误差标准差公式 (Equation 44-45):
σ(Δs_n) ≤ sqrt((n(n+1)(n+0.5)+2n)/24) * 2^{-t} * y
其中 n=内积长度, t=尾数位数, y=max|a_k*b_k|
"""

import torch
import math
import numpy as np
from datetime import datetime
import argparse


def get_device(prefer_gpu=True):
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


# ============================================================================
# A-ABFT 内积误差界 (论文原版公式)
# ============================================================================

def aabft_inner_product_bound(n, y, t):
    """
    A-ABFT 内积误差的3σ门限

    参数:
        n: 内积长度 (向量维度)
        y: max|a_k * b_k| 的上界
        t: 尾数位数 (FP64=53, FP32=23)

    返回:
        3σ门限值
    """
    # Equation 44-45 from A-ABFT paper
    # σ² = (n(n+1)(n+0.5)+2n)/24 * 2^{-2t} * y²
    var_coeff = (n * (n + 1) * (n + 0.5) + 2 * n) / 24
    sigma = math.sqrt(var_coeff) * (2 ** (-t)) * y
    return 3 * sigma  # 3σ 置信区间


def sea_abft_inner_product_bound(a_row, b_col, t):
    """
    SEA-ABFT 内积误差界 (简化误差分析)

    基于论文[28] Roy-Chowdhury and Banerjee, FTCS 1993
    """
    n = len(a_row)
    a_norm = torch.norm(a_row.to(torch.float64), p=2).item()
    b_norm = torch.norm(b_col.to(torch.float64), p=2).item()

    # SEA bound: ((n + 2m - 2) * ||b||_2 * Σ||a_i||_2 + n * ||a_{m+1}||_2 * ||b||_2) * ε_M
    # 简化版本 for single inner product:
    bound = (2 * n - 2) * a_norm * b_norm * (2 ** (-t))
    return bound


def vabft_inner_product_bound(a_row, b_col, e_max, c_sigma=2.5):
    """
    V-ABFT 内积误差界

    直接对校验差建模，使用方差估计
    """
    n = len(a_row)

    a_row_f64 = a_row.to(torch.float64)
    b_col_f64 = b_col.to(torch.float64)

    mu_a = torch.mean(a_row_f64).item()
    mu_b = torch.mean(b_col_f64).item()

    # 方差上界估计
    a_max = torch.max(a_row_f64).item()
    a_min = torch.min(a_row_f64).item()
    b_max = torch.max(b_col_f64).item()
    b_min = torch.min(b_col_f64).item()

    sigma_a2 = max((a_max - mu_a) * (mu_a - a_min), 1e-30)
    sigma_b2 = max((b_max - mu_b) * (mu_b - b_min), 1e-30)
    sigma_a = math.sqrt(sigma_a2)
    sigma_b = math.sqrt(sigma_b2)

    # V-ABFT bound (简化版，针对单个内积)
    # 内积 sum_k a_k * b_k 的方差约为 n * σ_a² * σ_b² + n * μ_a² * σ_b² + n * σ_a² * μ_b²
    var_term = n * (sigma_a2 * sigma_b2 + mu_a**2 * sigma_b2 + sigma_a2 * mu_b**2)
    bound = e_max * (n * abs(mu_a * mu_b) + c_sigma * math.sqrt(var_term))

    return bound


# ============================================================================
# 测试函数 - 复现A-ABFT论文Table II实验
# ============================================================================

def compute_inner_product_errors(M, K, N, dtype, device, num_samples=1000):
    """
    计算GEMM中内积的实际舍入误差和各方法的门限

    复现A-ABFT论文Table II的实验设置
    """
    if dtype == torch.float64:
        t = 53
        e_max = 3e-15
    elif dtype == torch.float32:
        t = 23
        e_max = 3e-7
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    actual_errors = []
    aabft_bounds = []
    sea_bounds = []
    vabft_bounds = []

    for _ in range(num_samples):
        # 生成随机矩阵 (A-ABFT论文使用 -1.0 to 1.0)
        A = (torch.rand(M, K, device=device, dtype=dtype) * 2 - 1)  # [-1, 1]
        B = (torch.rand(K, N, device=device, dtype=dtype) * 2 - 1)  # [-1, 1]

        # GPU/低精度计算
        C_computed = torch.matmul(A, B)

        # 高精度参考计算
        A_f64 = A.to(torch.float64)
        B_f64 = B.to(torch.float64)
        C_exact = torch.matmul(A_f64, B_f64)

        # 随机采样一些元素进行误差分析
        for _ in range(min(10, M * N)):
            i = torch.randint(0, M, (1,)).item()
            j = torch.randint(0, N, (1,)).item()

            # 实际舍入误差
            actual_error = abs(C_computed[i, j].to(torch.float64).item() - C_exact[i, j].item())
            actual_errors.append(actual_error)

            # A-ABFT 门限
            a_row = A[i, :]
            b_col = B[:, j]
            y = torch.max(torch.abs(a_row.to(torch.float64) * b_col.to(torch.float64))).item()
            aabft_bound = aabft_inner_product_bound(K, y, t)
            aabft_bounds.append(aabft_bound)

            # SEA-ABFT 门限
            sea_bound = sea_abft_inner_product_bound(a_row, b_col, t)
            sea_bounds.append(sea_bound)

            # V-ABFT 门限
            vabft_bound = vabft_inner_product_bound(a_row, b_col, e_max)
            vabft_bounds.append(vabft_bound)

    return {
        'actual_error': np.mean(actual_errors),
        'aabft': np.mean(aabft_bounds),
        'sea': np.mean(sea_bounds),
        'vabft': np.mean(vabft_bounds),
    }


def compute_inner_product_errors_range(M, K, N, dtype, device, value_range, num_samples=1000):
    """
    指定数值范围的内积误差测试
    """
    if dtype == torch.float64:
        t = 53
        e_max = 3e-15
    elif dtype == torch.float32:
        t = 23
        e_max = 3e-7
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    lo, hi = value_range

    actual_errors = []
    aabft_bounds = []
    sea_bounds = []
    vabft_bounds = []

    for _ in range(num_samples):
        # 生成指定范围的随机矩阵
        A = torch.rand(M, K, device=device, dtype=dtype) * (hi - lo) + lo
        B = torch.rand(K, N, device=device, dtype=dtype) * (hi - lo) + lo

        C_computed = torch.matmul(A, B)

        A_f64 = A.to(torch.float64)
        B_f64 = B.to(torch.float64)
        C_exact = torch.matmul(A_f64, B_f64)

        for _ in range(min(10, M * N)):
            i = torch.randint(0, M, (1,)).item()
            j = torch.randint(0, N, (1,)).item()

            actual_error = abs(C_computed[i, j].to(torch.float64).item() - C_exact[i, j].item())
            actual_errors.append(actual_error)

            a_row = A[i, :]
            b_col = B[:, j]
            y = torch.max(torch.abs(a_row.to(torch.float64) * b_col.to(torch.float64))).item()
            aabft_bound = aabft_inner_product_bound(K, y, t)
            aabft_bounds.append(aabft_bound)

            sea_bound = sea_abft_inner_product_bound(a_row, b_col, t)
            sea_bounds.append(sea_bound)

            vabft_bound = vabft_inner_product_bound(a_row, b_col, e_max)
            vabft_bounds.append(vabft_bound)

    return {
        'actual_error': np.mean(actual_errors),
        'aabft': np.mean(aabft_bounds),
        'sea': np.mean(sea_bounds),
        'vabft': np.mean(vabft_bounds),
    }


# ============================================================================
# 主测试
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='auto', choices=['cpu', 'cuda', 'auto'])
    parser.add_argument('--dtype', type=str, default='float64', choices=['float32', 'float64'])
    parser.add_argument('--samples', type=int, default=1000)
    args = parser.parse_args()

    if args.device == 'auto':
        device = get_device(prefer_gpu=True)
    else:
        device = torch.device(args.device)

    dtype = torch.float64 if args.dtype == 'float64' else torch.float32

    print("=" * 80)
    print(f"A-ABFT Paper Reproduction Test (Table II/III/IV)")
    print(f"Device: {device}, Dtype: {dtype}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # 测试矩阵规模 (与A-ABFT论文Table II一致)
    sizes = [512, 1024, 2048, 3072, 4096]

    # =========================================================================
    # Table II: Input value range -1.0 to 1.0
    # =========================================================================
    print("\n" + "=" * 70)
    print("Table II: Random input value range -1.0 to 1.0")
    print("=" * 70)
    print(f"{'Matrix':<10} {'AVG RND ERR':<15} {'AVG A-ABFT':<15} {'AVG SEA':<15} {'AVG V-ABFT':<15}")
    print("-" * 70)

    for n in sizes:
        results = compute_inner_product_errors_range(
            n, n, n, dtype, device, (-1.0, 1.0), num_samples=args.samples
        )
        print(f"{n:<10} {results['actual_error']:<15.2e} {results['aabft']:<15.2e} "
              f"{results['sea']:<15.2e} {results['vabft']:<15.2e}")

    # =========================================================================
    # Table III: Input value range -100.0 to 100.0
    # =========================================================================
    print("\n" + "=" * 70)
    print("Table III: Random input value range -100.0 to 100.0")
    print("=" * 70)
    print(f"{'Matrix':<10} {'AVG RND ERR':<15} {'AVG A-ABFT':<15} {'AVG SEA':<15} {'AVG V-ABFT':<15}")
    print("-" * 70)

    for n in sizes:
        results = compute_inner_product_errors_range(
            n, n, n, dtype, device, (-100.0, 100.0), num_samples=args.samples
        )
        print(f"{n:<10} {results['actual_error']:<15.2e} {results['aabft']:<15.2e} "
              f"{results['sea']:<15.2e} {results['vabft']:<15.2e}")

    # =========================================================================
    # Tightness Ratio (门限紧致度)
    # =========================================================================
    print("\n" + "=" * 70)
    print("Tightness Ratio: Bound / Actual Error (lower is better)")
    print("=" * 70)
    print(f"{'Matrix':<10} {'A-ABFT':<15} {'SEA-ABFT':<15} {'V-ABFT':<15}")
    print("-" * 70)

    for n in sizes:
        results = compute_inner_product_errors_range(
            n, n, n, dtype, device, (-1.0, 1.0), num_samples=args.samples
        )
        err = results['actual_error'] + 1e-30
        print(f"{n:<10} {results['aabft']/err:<15.0f}x {results['sea']/err:<15.0f}x "
              f"{results['vabft']/err:<15.0f}x")


if __name__ == "__main__":
    main()
