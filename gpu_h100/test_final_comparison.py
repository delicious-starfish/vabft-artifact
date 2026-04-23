"""
A-ABFT vs V-ABFT 最终对比测试 (FP64 / FP32)

- A-ABFT: 原始公式 3σ, σ = √((n(n+1)(n+0.5)+2n)/24) × 2^(-t) × y
- V-ABFT: e_max(N) × (T_det + T_var23 + T_var4), e_max 按 √N 缩放
- 分布: U(-1, 1)
"""

import torch
import math
import numpy as np
from datetime import datetime
import argparse


# ============================================================================
# e_max 拟合公式 (GPU H100 实测)
# FP64: e_max = 8.3e-18 × √N + 2.1e-16
# FP32: e_max = 4.1e-9  × √N + 9.4e-8
# CPU 使用相同公式 (保守估计)
# ============================================================================

def get_emax(n: int, dtype, margin: float = 1.0) -> float:
    sqrt_n = math.sqrt(n)
    if dtype == torch.float64:
        return margin * (8.3e-18 * sqrt_n + 2.1e-16)
    elif dtype == torch.float32:
        return margin * (4.1e-9 * sqrt_n + 9.4e-8)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


# ============================================================================
# A-ABFT 门限计算 (原始公式)
# ============================================================================

def compute_y(A, B):
    """y = max|A| × max|B_rowsum|"""
    B_rowsum = torch.sum(B.to(torch.float64), dim=-1)
    max_A = torch.max(torch.abs(A.to(torch.float64))).item()
    max_B_rowsum = torch.max(torch.abs(B_rowsum)).item()
    return max_A * max_B_rowsum


def aabft_threshold(n: int, t: int, y: float) -> float:
    """
    A-ABFT 门限: 3σ
    σ = √((n(n+1)(n+0.5)+2n)/24) × 2^(-t) × y
    """
    var_coeff = (n * (n + 1) * (n + 0.5) + 2 * n) / 24
    sigma = math.sqrt(var_coeff) * (2 ** (-t)) * y
    return 3 * sigma


# ============================================================================
# V-ABFT 门限计算
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
# 校验差计算
# ============================================================================

def compute_checksum_diff(A, B, C):
    """计算校验差 (FP64 精度)"""
    B_rowsum = torch.sum(B.to(torch.float64), dim=-1)
    checksum = torch.matmul(A.to(torch.float64), B_rowsum)
    C_rowsum = torch.sum(C.to(torch.float64), dim=-1)
    diff = torch.abs(checksum - C_rowsum)
    return diff


# ============================================================================
# 测试
# ============================================================================

def run_test(n: int, dtype, device, num_trials: int):
    """运行 A-ABFT vs V-ABFT 对比测试"""
    t = 53 if dtype == torch.float64 else 23
    e_max = get_emax(n, dtype, margin=1.5)

    actual_diffs = []
    aabft_thresholds = []
    vabft_thresholds = []
    aabft_fp = 0
    vabft_fp = 0

    for trial in range(num_trials):
        # U(-1, 1)
        A = 2 * torch.rand(n, n, device=device, dtype=dtype) - 1
        B = 2 * torch.rand(n, n, device=device, dtype=dtype) - 1
        C = torch.matmul(A, B)

        # 校验差
        diff = compute_checksum_diff(A, B, C)

        # A-ABFT
        y = compute_y(A, B)
        aabft_th = aabft_threshold(n, t, y)

        # V-ABFT
        vabft_th = vabft_threshold(A, B, e_max)

        actual_diffs.append(diff.mean().item())
        aabft_thresholds.append(aabft_th)
        vabft_thresholds.append(vabft_th.mean().item())

        # 误报检查
        if (diff > aabft_th).any():
            aabft_fp += 1
        if (diff > vabft_th).any():
            vabft_fp += 1

    avg_diff = np.mean(actual_diffs)
    avg_aabft = np.mean(aabft_thresholds)
    avg_vabft = np.mean(vabft_thresholds)

    return {
        'n': n,
        'e_max': e_max,
        'avg_diff': avg_diff,
        'aabft_thresh': avg_aabft,
        'vabft_thresh': avg_vabft,
        'aabft_tightness': avg_aabft / (avg_diff + 1e-30),
        'vabft_tightness': avg_vabft / (avg_diff + 1e-30),
        'aabft_fpr': aabft_fp / num_trials * 100,
        'vabft_fpr': vabft_fp / num_trials * 100,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--dtype', type=str, default='float64',
                        choices=['float32', 'float64'])
    parser.add_argument('--trials', type=int, default=100)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.float64 if args.dtype == 'float64' else torch.float32
    t = 53 if dtype == torch.float64 else 23
    u = 2 ** (-t)

    print("=" * 110)
    print(f"A-ABFT vs V-ABFT Final Comparison ({args.dtype})")
    print(f"Device: {device}, Dtype: {dtype}")
    print(f"e_max formula: {'8.3e-18×√N + 2.1e-16' if dtype == torch.float64 else '4.1e-9×√N + 9.4e-8'}")
    print(f"Distribution: U(-1, 1)")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 110)

    sizes = [128, 256, 512, 1024, 2048]

    print(f"\n{'Size':<8} {'e_max':<12} {'Actual Diff':<14} {'A-ABFT':<14} {'V-ABFT':<14} "
          f"{'A-Tight':<10} {'V-Tight':<10} {'A-FPR':<8} {'V-FPR':<8}")
    print("-" * 110)

    results = []
    for n in sizes:
        print(f"Testing n={n}...", end="", flush=True)
        res = run_test(n, dtype, device, args.trials)
        results.append(res)

        aabft_t = res['aabft_tightness']
        vabft_t = res['vabft_tightness']
        advantage = aabft_t / (vabft_t + 1e-30)

        print(f"\r{n:<8} {res['e_max']:<12.2e} {res['avg_diff']:<14.2e} "
              f"{res['aabft_thresh']:<14.2e} {res['vabft_thresh']:<14.2e} "
              f"{aabft_t:<10.0f}× {vabft_t:<10.0f}× "
              f"{res['aabft_fpr']:<8.2f}% {res['vabft_fpr']:<8.2f}%")

    # 汇总
    print("\n" + "=" * 110)
    print("Summary for Paper")
    print("=" * 110)
    for res in results:
        advantage = res['aabft_tightness'] / (res['vabft_tightness'] + 1e-30)
        print(f"  n={res['n']:<6} A-ABFT: {res['aabft_tightness']:.0f}×  "
              f"V-ABFT: {res['vabft_tightness']:.0f}×  "
              f"Advantage: {advantage:.1f}×  "
              f"A-FPR: {res['aabft_fpr']:.1f}%  V-FPR: {res['vabft_fpr']:.1f}%")

    # LaTeX
    print("\nLaTeX Table Data:")
    for res in results:
        print(f"  {res['n']}×{res['n']} & "
              f"${res['avg_diff']:.2e}$ & "
              f"${res['aabft_thresh']:.2e}$ & ${res['vabft_thresh']:.2e}$ & "
              f"${int(res['aabft_tightness'])}\\times$ & ${int(res['vabft_tightness'])}\\times$ & "
              f"{res['aabft_fpr']:.0f}\\% & {res['vabft_fpr']:.0f}\\% \\\\")


if __name__ == "__main__":
    main()
