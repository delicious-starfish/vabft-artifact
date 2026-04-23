"""
A-ABFT vs V-ABFT 对比测试 - FP64版本

使用mpmath高精度库作为基准计算校验差的真实误差
"""

import torch
import math
import numpy as np
from datetime import datetime
import argparse
from mpmath import mp, mpf, fsum

# 设置mpmath精度 (100位十进制精度)
mp.dps = 100


def matmul_high_precision(A, B):
    """使用mpmath高精度计算矩阵乘法"""
    M, K = A.shape
    K2, N = B.shape
    assert K == K2

    # 转换为mpmath高精度
    A_mp = [[mpf(A[i, k].item()) for k in range(K)] for i in range(M)]
    B_mp = [[mpf(B[k, j].item()) for j in range(N)] for k in range(K)]

    # 高精度矩阵乘法
    C_mp = [[fsum([A_mp[i][k] * B_mp[k][j] for k in range(K)]) for j in range(N)] for i in range(M)]

    return C_mp


def rowsum_high_precision(C_mp):
    """高精度行求和"""
    M = len(C_mp)
    N = len(C_mp[0])
    return [fsum([C_mp[i][j] for j in range(N)]) for i in range(M)]


def aabft_threshold(a, b, c):
    """A-ABFT 校验差门限 (FP64版本)"""
    dtype = a.dtype
    M, K = a.shape
    K2, N = b.shape
    assert K == K2

    t = 53
    u_low = 2**(-53)
    u_high = 2**(-53)  # FP64校验

    # 高精度校验和
    b_rowsum = torch.sum(b.to(torch.float64), dim=-1, keepdim=True)
    checksum = torch.matmul(a.to(torch.float64), b_rowsum).squeeze(-1)

    c_max, _ = torch.max(torch.abs(c), dim=-1)
    b_max, _ = torch.max(torch.abs(b), dim=-1, keepdim=True)
    a_max, _ = torch.max(torch.abs(a), dim=-1)

    E1 = math.sqrt(N*(N+1)*(2*N+1)/48) * c_max * u_high
    E2 = c_max * u_low * math.sqrt(N*(N+1)*(2*N+1)/6)
    delta_b = math.sqrt(N*(N+1)*(2*N+1)/48) * b_max.to(torch.float64) * u_high
    E3 = torch.matmul(torch.abs(a.to(torch.float64)), delta_b).squeeze(-1)
    var_coeff = (K * (K + 1) * (K + 0.5) + 2 * K) / 24
    y = a_max.to(torch.float64) * torch.max(b_max)
    E4 = 3 * math.sqrt(var_coeff) * u_high * y

    threshold = E1.to(torch.float64) + E2 + E3 + E4
    return checksum, threshold


def vabft_threshold(a, b, e_max=2e-16, c_sigma=2.5):
    """V-ABFT 校验差门限 (FP64版本)"""
    dtype = a.dtype
    M, K = a.shape
    K2, N = b.shape
    assert K == K2

    # 高精度校验和
    b_rowsum = torch.sum(b.to(torch.float64), dim=-1, keepdim=True)
    checksum = torch.matmul(a.to(torch.float64), b_rowsum).squeeze(-1)

    # 统计量
    mu_A = torch.mean(a.to(torch.float64), dim=-1)
    mu_B = torch.mean(b.to(torch.float64), dim=-1)

    a_max = torch.max(a.to(torch.float64), dim=-1).values
    a_min = torch.min(a.to(torch.float64), dim=-1).values
    b_max = torch.max(b.to(torch.float64), dim=-1).values
    b_min = torch.min(b.to(torch.float64), dim=-1).values

    sigma_A2 = torch.clamp((a_max - mu_A) * (mu_A - a_min), min=1e-20)
    sigma_B2 = torch.clamp((b_max - mu_B) * (mu_B - b_min), min=1e-20)
    sigma_A = torch.sqrt(sigma_A2)

    mu_A_abs = torch.abs(mu_A)
    mu_B_abs = torch.abs(mu_B)

    sum_mu_B = torch.sum(mu_B_abs)
    sum_mu_B2 = torch.sum(mu_B_abs ** 2)
    sum_sigma_B2 = torch.sum(sigma_B2)

    sqrt_N = math.sqrt(N)

    T_det = N * mu_A_abs * sum_mu_B
    T_var23 = c_sigma * torch.sqrt(N * mu_A_abs**2 * sum_sigma_B2 + N**2 * sigma_A2 * sum_mu_B2)
    T_var4 = c_sigma * sqrt_N * sigma_A * torch.sqrt(sum_sigma_B2)

    threshold = e_max * (T_det + T_var23 + T_var4)
    return checksum, threshold


def compute_true_checksum_diff(A, B, C):
    """
    使用mpmath高精度计算真实校验差

    checksum_diff = A @ sum(B, dim=1) - sum(C, dim=1)
    """
    M, K = A.shape
    K2, N = B.shape
    assert K == K2

    # 高精度B行求和
    B_rowsum_mp = [fsum([mpf(B[k, j].item()) for j in range(N)]) for k in range(K)]

    # 高精度 A @ B_rowsum
    checksum_mp = [fsum([mpf(A[i, k].item()) * B_rowsum_mp[k] for k in range(K)]) for i in range(M)]

    # 高精度 C行求和 (C已经是低精度计算结果)
    C_rowsum_mp = [fsum([mpf(C[i, j].item()) for j in range(N)]) for i in range(M)]

    # 真实校验差
    true_diff = [abs(checksum_mp[i] - C_rowsum_mp[i]) for i in range(M)]

    return [float(d) for d in true_diff]


def run_test_fp64(M, K, N, device, num_trials=100):
    """FP64对比测试（使用mpmath基准）"""
    dtype = torch.float64

    actual_diffs = []
    aabft_thresholds = []
    vabft_thresholds = []
    aabft_fp = 0
    vabft_fp = 0

    for trial in range(num_trials):
        if trial % 10 == 0:
            print(f"  Trial {trial}/{num_trials}...", flush=True)

        A = torch.abs(torch.randn(M, K, device=device, dtype=dtype)) + 0.5
        B = torch.abs(torch.randn(K, N, device=device, dtype=dtype)) + 0.5
        C = torch.matmul(A, B)

        # 使用mpmath计算真实校验差
        true_diff = compute_true_checksum_diff(A, B, C)

        # A-ABFT门限
        aabft_ck, aabft_th = aabft_threshold(A, B, C)

        # V-ABFT门限
        vabft_ck, vabft_th = vabft_threshold(A, B)

        actual_diffs.append(np.mean(true_diff))
        aabft_thresholds.append(aabft_th.mean().item())
        vabft_thresholds.append(vabft_th.mean().item())

        # 检查误检
        if any(true_diff[i] > aabft_th[i].item() for i in range(M)):
            aabft_fp += 1
        if any(true_diff[i] > vabft_th[i].item() for i in range(M)):
            vabft_fp += 1

    return {
        'actual_mean': np.mean(actual_diffs),
        'aabft_thresh_mean': np.mean(aabft_thresholds),
        'vabft_thresh_mean': np.mean(vabft_thresholds),
        'aabft_tightness': np.mean(aabft_thresholds) / (np.mean(actual_diffs) + 1e-30),
        'vabft_tightness': np.mean(vabft_thresholds) / (np.mean(actual_diffs) + 1e-30),
        'aabft_fpr': aabft_fp / num_trials,
        'vabft_fpr': vabft_fp / num_trials,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--trials', type=int, default=100)
    args = parser.parse_args()

    device = torch.device(args.device)

    print("=" * 90)
    print("A-ABFT vs V-ABFT Comparison (FP64 with mpmath high-precision baseline)")
    print(f"Device: {device}")
    print(f"mpmath precision: {mp.dps} decimal places")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 90)

    # FP64测试用较小规模（mpmath很慢）
    sizes = [
        (16, 32, 16),
        (16, 64, 16),
        (32, 64, 32),
        (32, 128, 32),
        (64, 128, 64),
    ]

    print("\n" + "-" * 110)
    print(f"{'Size':<18} {'Actual Diff':<14} {'A-ABFT Thresh':<14} {'V-ABFT Thresh':<14} "
          f"{'A-ABFT Tight':<14} {'V-ABFT Tight':<14}")
    print("-" * 110)

    results = []
    for M, K, N in sizes:
        print(f"\nTesting ({M},{K},{N})...")
        res = run_test_fp64(M, K, N, device, num_trials=args.trials)
        results.append((M, K, N, res))
        print(f"({M},{K},{N}){' '*(13-len(f'({M},{K},{N})'))}"
              f"{res['actual_mean']:<14.2e} {res['aabft_thresh_mean']:<14.2e} "
              f"{res['vabft_thresh_mean']:<14.2e} {res['aabft_tightness']:<14.1f}x "
              f"{res['vabft_tightness']:<14.1f}x")

    print("\n" + "=" * 90)
    print("Summary:")
    print("-" * 90)
    print(f"{'Size':<18} {'A-ABFT FPR':<15} {'V-ABFT FPR':<15} {'V-ABFT/A-ABFT Ratio':<20}")
    print("-" * 90)

    for M, K, N, res in results:
        ratio = res['vabft_thresh_mean'] / (res['aabft_thresh_mean'] + 1e-30)
        print(f"({M},{K},{N}){' '*(13-len(f'({M},{K},{N})'))}"
              f"{res['aabft_fpr']*100:<15.4f}% {res['vabft_fpr']*100:<15.4f}% "
              f"{ratio:<20.3f}")


if __name__ == "__main__":
    main()
