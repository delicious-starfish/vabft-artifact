"""
e_max 测量 - NPU版本

使用公式: e_max = |Δ| / |checksum|
固定N=256，只变化K
"""

import torch
import torch_npu
import math
import numpy as np
from datetime import datetime
import argparse


def compute_theoretical_bound(A, B, c_sigma=2.5):
    """计算V-ABFT理论边界 (不包含e_max)"""
    M, K = A.shape
    K2, N = B.shape

    A_f32 = A.to(torch.float32)
    B_f32 = B.to(torch.float32)

    mu_A = torch.mean(A_f32, dim=-1)
    mu_B = torch.mean(B_f32, dim=-1)

    a_max = torch.max(A_f32, dim=-1).values
    a_min = torch.min(A_f32, dim=-1).values
    b_max = torch.max(B_f32, dim=-1).values
    b_min = torch.min(B_f32, dim=-1).values

    sigma_A2 = torch.clamp((a_max - mu_A) * (mu_A - a_min), min=1e-20)
    sigma_B2 = torch.clamp((b_max - mu_B) * (mu_B - b_min), min=1e-20)
    sigma_A = torch.sqrt(sigma_A2)

    mu_A_abs = torch.abs(mu_A)
    mu_B_abs = torch.abs(mu_B)

    sum_mu_B = torch.sum(mu_B_abs)
    sum_mu_B2 = torch.sum(mu_B_abs ** 2)
    sum_sigma_B2 = torch.sum(sigma_B2)

    T_det = N * mu_A_abs * sum_mu_B
    T_var23 = c_sigma * torch.sqrt(N * mu_A_abs**2 * sum_sigma_B2 + N**2 * sigma_A2 * sum_mu_B2)
    T_var4 = c_sigma * math.sqrt(N) * sigma_A * torch.sqrt(sum_sigma_B2)

    bound = T_det + T_var23 + T_var4
    return bound.cpu()


def measure_emax_npu(M, K, N, dtype, num_trials=1000):
    """在NPU上测量e_max，使用原始公式: e_max = |Δ| / |checksum|"""
    device = torch.device('npu:0')
    emax_values = []

    for _ in range(num_trials):
        A = torch.abs(torch.randn(M, K, dtype=torch.float32)) + 0.5
        B = torch.abs(torch.randn(K, N, dtype=torch.float32)) + 0.5

        A_npu = A.to(dtype).to(device)
        B_npu = B.to(dtype).to(device)
        C_npu = torch.matmul(A_npu, B_npu)

        b_rowsum = torch.sum(B, dim=-1, keepdim=True)
        checksum = torch.matmul(A, b_rowsum).squeeze(-1)
        c_rowsum = torch.sum(C_npu.cpu().to(torch.float32), dim=-1)
        actual_diff = torch.abs(checksum - c_rowsum)

        # 原始公式: e_max = |Δ| / |checksum|
        emax = actual_diff / (torch.abs(checksum) + 1e-30)
        emax_values.extend(emax.tolist())

    return {
        'max': np.max(emax_values),
        'p999': np.percentile(emax_values, 99.9),
        'p99': np.percentile(emax_values, 99),
        'mean': np.mean(emax_values),
    }


def get_unit_roundoff(dtype):
    if dtype == torch.float32:
        return 2**(-24)
    elif dtype == torch.float16:
        return 2**(-11)
    elif dtype == torch.bfloat16:
        return 2**(-8)
    return 1e-7


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dtype', type=str, default='bfloat16',
                        choices=['float32', 'float16', 'bfloat16'])
    parser.add_argument('--trials', type=int, default=2000)
    args = parser.parse_args()

    if args.dtype == 'float32':
        dtype = torch.float32
    elif args.dtype == 'float16':
        dtype = torch.float16
    else:
        dtype = torch.bfloat16

    u = get_unit_roundoff(dtype)

    print("=" * 90)
    print("e_max Measurement on NPU - Formula: e_max = |Δ| / |checksum|")
    print(f"Dtype: {dtype}, u = {u:.2e}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 90)

    M = 128
    N = 256
    K_values = [256, 512, 1024, 2048, 4096, 8192]

    print(f"\n{'K':<8} {'max(e_max)':<14} {'max/u':<10} {'p99.9':<14} {'p99.9/u':<10} {'mean':<14}")
    print("-" * 80)

    results = []
    for K in K_values:
        print(f"Testing K={K}...", end="", flush=True)
        res = measure_emax_npu(M, K, N, dtype, num_trials=args.trials)
        results.append((K, res))
        print(f"\r{K:<8} {res['max']:<14.4e} {res['max']/u:<10.2f} "
              f"{res['p999']:<14.4e} {res['p999']/u:<10.2f} {res['mean']:<14.4e}")

    # 分析
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    K_arr = np.array([r[0] for r in results])
    emax_arr = np.array([r[1]['max'] for r in results])

    print(f"\ne_max range: {emax_arr.min()/u:.2f}u - {emax_arr.max()/u:.2f}u")
    print(f"Variation: {emax_arr.max()/emax_arr.min():.2f}x")

    # 检查是否随sqrt(K)增长
    sqrt_K = np.sqrt(K_arr)
    correlation = np.corrcoef(sqrt_K, emax_arr)[0, 1]
    print(f"Correlation with sqrt(K): {correlation:.4f}")

    if correlation > 0.9:
        coef = np.mean(emax_arr / sqrt_K)
        print(f"e_max ≈ {coef:.2e} * sqrt(K)")
    else:
        print(f"e_max is approximately constant: {np.mean(emax_arr):.2e} ({np.mean(emax_arr)/u:.2f}u)")


if __name__ == "__main__":
    main()
