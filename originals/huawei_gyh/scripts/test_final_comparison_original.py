"""
A-ABFT vs V-ABFT 对比测试 - 使用原论文 per-element y 方式

原论文 (Braun et al. DSN'14) 做法:
  - 对 full-checksum matrix C_fc 的每个 checksum 元素单独计算 y
  - 对 row checksum: y_m = max_k |A[m,k] · B_rowsum[k]| (per-row)
  - σ = √((n(n+1)(n+0.5)+2n)/24) × 2^(-t) × y_m
  - threshold = 3σ (per-row)

对比:
  - 方式1: y_global = max|A| × max|B_rowsum| (我们之前用的，最保守)
  - 方式2: y_per_row = max_k |A[m,k]| × max|B_rowsum| (per-row max)
  - 方式3: y_exact = max_k |A[m,k] · B_rowsum[k]| (原论文 per-element, 最紧)
"""

import torch
import math
import numpy as np
from datetime import datetime
import argparse


# ============================================================================
# e_max 拟合公式
# ============================================================================

def get_emax(n: int, dtype, margin: float = 1.5) -> float:
    sqrt_n = math.sqrt(n)
    if dtype == torch.float64:
        return margin * (8.3e-18 * sqrt_n + 2.1e-16)
    elif dtype == torch.float32:
        return margin * (4.1e-9 * sqrt_n + 9.4e-8)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


# ============================================================================
# A-ABFT 三种 y 计算方式
# ============================================================================

def aabft_var_coeff(n: int) -> float:
    """(n(n+1)(n+0.5)+2n)/24"""
    return (n * (n + 1) * (n + 0.5) + 2 * n) / 24


def aabft_threshold_global(A, B, n: int, t: int):
    """方式1: y = max|A| × max|B_rowsum| (全局, 最松)"""
    B_rowsum = torch.sum(B.to(torch.float64), dim=-1)
    max_A = torch.max(torch.abs(A.to(torch.float64))).item()
    max_Brs = torch.max(torch.abs(B_rowsum)).item()
    y = max_A * max_Brs
    sigma = math.sqrt(aabft_var_coeff(n)) * (2 ** (-t)) * y
    return 3 * sigma  # scalar, same for all rows


def aabft_threshold_perrow(A, B, n: int, t: int):
    """方式2: y_m = max_k |A[m,k]| × max|B_rowsum| (per-row A max)"""
    B_rowsum = torch.sum(B.to(torch.float64), dim=-1)
    max_A_perrow = torch.max(torch.abs(A.to(torch.float64)), dim=-1).values  # (M,)
    max_Brs = torch.max(torch.abs(B_rowsum)).item()
    y_perrow = max_A_perrow * max_Brs  # (M,)
    coeff = math.sqrt(aabft_var_coeff(n)) * (2 ** (-t))
    return 3 * coeff * y_perrow  # (M,)


def aabft_threshold_exact(A, B, n: int, t: int):
    """方式3: y_m = max_k |A[m,k] · B_rowsum[k]| (原论文 per-element, 最紧)"""
    B_rowsum = torch.sum(B.to(torch.float64), dim=-1)  # (K,)
    A64 = A.to(torch.float64)
    # 对每行 m: y_m = max_k |A[m,k] * B_rowsum[k]|
    products = torch.abs(A64 * B_rowsum.unsqueeze(0))  # (M, K)
    y_exact = torch.max(products, dim=-1).values  # (M,)
    coeff = math.sqrt(aabft_var_coeff(n)) * (2 ** (-t))
    return 3 * coeff * y_exact  # (M,)


# ============================================================================
# V-ABFT 门限
# ============================================================================

def vabft_threshold(A, B, e_max: float, c_sigma: float = 2.5):
    """V-ABFT 门限: e_max × (T_det + T_var23 + T_var4)"""
    M, K = A.shape
    K2, N = B.shape

    A64 = A.to(torch.float64)
    B64 = B.to(torch.float64)

    mu_A = torch.mean(A64, dim=-1)
    mu_B = torch.mean(B64, dim=-1)

    a_max = torch.max(A64, dim=-1).values
    a_min = torch.min(A64, dim=-1).values
    b_max = torch.max(B64, dim=-1).values
    b_min = torch.min(B64, dim=-1).values

    sigma_A2 = torch.clamp((a_max - mu_A) * (mu_A - a_min), min=1e-20)
    sigma_A = torch.sqrt(sigma_A2)
    sigma_B2 = torch.clamp((b_max - mu_B) * (mu_B - b_min), min=1e-20)

    mu_A_abs = torch.abs(mu_A)
    mu_B_abs = torch.abs(mu_B)
    sum_mu_B = torch.sum(mu_B_abs)
    sum_mu_B2 = torch.sum(mu_B_abs ** 2)
    sum_sigma_B2 = torch.sum(sigma_B2)

    T_det = N * mu_A_abs * sum_mu_B
    T_var23 = c_sigma * torch.sqrt(
        N * mu_A_abs**2 * sum_sigma_B2 + N**2 * sigma_A2 * sum_mu_B2
    )
    T_var4 = c_sigma * math.sqrt(N) * sigma_A * torch.sqrt(sum_sigma_B2)

    threshold = e_max * (T_det + T_var23 + T_var4)
    return threshold


# ============================================================================
# 校验差
# ============================================================================

def compute_checksum_diff(A, B, C):
    B_rowsum = torch.sum(B.to(torch.float64), dim=-1)
    checksum = torch.matmul(A.to(torch.float64), B_rowsum)
    C_rowsum = torch.sum(C.to(torch.float64), dim=-1)
    return torch.abs(checksum - C_rowsum)


# ============================================================================
# 测试
# ============================================================================

def run_test(n: int, dtype, device, num_trials: int):
    t = 53 if dtype == torch.float64 else 23
    e_max = get_emax(n, dtype)

    stats = {k: [] for k in ['diff', 'global', 'perrow', 'exact', 'vabft']}
    fp = {'global': 0, 'perrow': 0, 'exact': 0, 'vabft': 0}

    for _ in range(num_trials):
        A = 2 * torch.rand(n, n, device=device, dtype=dtype) - 1
        B = 2 * torch.rand(n, n, device=device, dtype=dtype) - 1
        C = torch.matmul(A, B)

        diff = compute_checksum_diff(A, B, C)  # (M,)

        th_global = aabft_threshold_global(A, B, n, t)    # scalar
        th_perrow = aabft_threshold_perrow(A, B, n, t)     # (M,)
        th_exact = aabft_threshold_exact(A, B, n, t)       # (M,)
        th_vabft = vabft_threshold(A, B, e_max)             # (M,)

        stats['diff'].append(diff.mean().item())
        stats['global'].append(th_global if isinstance(th_global, float) else th_global.mean().item())
        stats['perrow'].append(th_perrow.mean().item())
        stats['exact'].append(th_exact.mean().item())
        stats['vabft'].append(th_vabft.mean().item())

        if (diff > th_global).any():
            fp['global'] += 1
        if (diff > th_perrow).any():
            fp['perrow'] += 1
        if (diff > th_exact).any():
            fp['exact'] += 1
        if (diff > th_vabft).any():
            fp['vabft'] += 1

    avg_diff = np.mean(stats['diff'])
    results = {'n': n, 'avg_diff': avg_diff, 'e_max': e_max}

    for key in ['global', 'perrow', 'exact', 'vabft']:
        avg_th = np.mean(stats[key])
        results[f'{key}_thresh'] = avg_th
        results[f'{key}_tight'] = avg_th / (avg_diff + 1e-30)
        results[f'{key}_fpr'] = fp[key] / num_trials * 100

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--dtype', type=str, default='float64',
                        choices=['float32', 'float64'])
    parser.add_argument('--trials', type=int, default=100)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.float64 if args.dtype == 'float64' else torch.float32

    print("=" * 130)
    print(f"A-ABFT (3 variants) vs V-ABFT Comparison ({args.dtype})")
    print(f"Device: {device}, Distribution: U(-1, 1), Trials: {args.trials}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 130)
    print(f"  Global:  y = max|A| × max|B_rowsum|  (loosest)")
    print(f"  PerRow:  y_m = max|A[m,:]| × max|B_rowsum|")
    print(f"  Exact:   y_m = max_k |A[m,k]·B_rowsum[k]|  (original paper, tightest)")
    print("=" * 130)

    sizes = [128, 256, 512, 1024, 2048]

    print(f"\n{'Size':<8} {'Diff':<12} "
          f"{'Global':<10} {'PerRow':<10} {'Exact':<10} {'V-ABFT':<10} "
          f"{'G-FPR':<8} {'P-FPR':<8} {'E-FPR':<8} {'V-FPR':<8}")
    print("-" * 120)

    for n in sizes:
        print(f"Testing n={n}...", end="", flush=True)
        res = run_test(n, dtype, device, args.trials)

        print(f"\r{n:<8} {res['avg_diff']:<12.2e} "
              f"{res['global_tight']:<10.0f}× {res['perrow_tight']:<10.0f}× "
              f"{res['exact_tight']:<10.0f}× {res['vabft_tight']:<10.0f}× "
              f"{res['global_fpr']:<8.1f}% {res['perrow_fpr']:<8.1f}% "
              f"{res['exact_fpr']:<8.1f}% {res['vabft_fpr']:<8.1f}%")

    print()


if __name__ == "__main__":
    main()
