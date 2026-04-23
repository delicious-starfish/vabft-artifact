"""
A-ABFT vs V-ABFT 对比测试 - 使用动态 e_max

根据 GPU e_max 测量结果，e_max 随 √N 增长（1.2× 安全余量）：
- GPU FP64: e_max = 1.0e-17 × √N + 2.5e-16
- GPU FP32: e_max = 5.0e-09 × √N + 1.2e-07

这个脚本用于重新测试 Tables 4/5 的数据。
"""

import torch
import math
import numpy as np
from datetime import datetime
import argparse

try:
    from mpmath import mp, mpf, fsum
    mp.dps = 100
    HAS_MPMATH = True
except ImportError:
    HAS_MPMATH = False
    print("Warning: mpmath not available, using higher precision as baseline")


# ============================================================================
# 动态 e_max 计算
# ============================================================================

def get_dynamic_emax(N: int, dtype, device='cuda'):
    """
    根据矩阵大小 N 计算动态 e_max（1.2× 安全余量）

    公式（带常数项，1.2× 安全余量）:
    - GPU FP64: e_max = 1.0×10⁻¹⁷ × √N + 2.5×10⁻¹⁶
    - GPU FP32: e_max = 5.0×10⁻⁹ × √N + 1.2×10⁻⁷

    CPU 测量结果 (近似常数):
    - FP64: ~6e-16 (约 5u)
    - FP32: ~4e-07 (约 6u)
    """
    sqrt_N = math.sqrt(N)

    if 'cuda' in str(device):
        # GPU: e_max = a × √N + b (1.2× 安全余量)
        if dtype == torch.float64:
            e_max = 1.0e-17 * sqrt_N + 2.5e-16
        else:  # float32
            e_max = 5.0e-9 * sqrt_N + 1.2e-7
    else:
        # CPU: e_max 近似常数
        if dtype == torch.float64:
            e_max = 6e-16
        else:  # float32
            e_max = 4e-7

    return e_max


# ============================================================================
# A-ABFT 门限计算
# ============================================================================

def aabft_threshold(n: int, t: int, y: float = 21.0) -> float:
    """
    A-ABFT 门限公式

    根据论文 Section IV-C:
    σ(Δsn) ≤ sqrt((n(n+1)(n+0.5)+2n)/24) * 2^(-t) * y

    门限 = 3σ
    """
    var_coeff = (n * (n + 1) * (n + 0.5) + 2 * n) / 24
    sigma = math.sqrt(var_coeff) * (2 ** (-t)) * y
    return 3 * sigma


# ============================================================================
# V-ABFT 门限计算
# ============================================================================

def vabft_threshold(A, B, e_max: float, c_sigma: float = 2.5):
    """
    V-ABFT 门限计算

    参数:
        A: 矩阵 A (M, K)
        B: 矩阵 B (K, N)
        e_max: 平台相关的误差系数 (动态计算)
        c_sigma: 置信系数 (默认 2.5)
    """
    M, K = A.shape
    K2, N = B.shape
    assert K == K2

    # 转换为 float64 计算统计量
    A_f64 = A.to(torch.float64)
    B_f64 = B.to(torch.float64)

    # A 的行统计量
    mu_A = torch.mean(A_f64, dim=-1)  # (M,)
    a_max = torch.max(A_f64, dim=-1).values
    a_min = torch.min(A_f64, dim=-1).values
    sigma_A2 = torch.clamp((a_max - mu_A) * (mu_A - a_min), min=1e-20)
    sigma_A = torch.sqrt(sigma_A2)

    # B 的行统计量
    mu_B = torch.mean(B_f64, dim=-1)  # (K,)
    b_max = torch.max(B_f64, dim=-1).values
    b_min = torch.min(B_f64, dim=-1).values
    sigma_B2 = torch.clamp((b_max - mu_B) * (mu_B - b_min), min=1e-20)

    # 汇总统计量
    mu_A_abs = torch.abs(mu_A)
    mu_B_abs = torch.abs(mu_B)
    sum_mu_B = torch.sum(mu_B_abs)
    sum_mu_B2 = torch.sum(mu_B_abs ** 2)
    sum_sigma_B2 = torch.sum(sigma_B2)

    sqrt_N = math.sqrt(N)

    # V-ABFT 门限公式
    T_det = N * mu_A_abs * sum_mu_B
    T_var23 = c_sigma * torch.sqrt(N * mu_A_abs**2 * sum_sigma_B2 + N**2 * sigma_A2 * sum_mu_B2)
    T_var4 = c_sigma * sqrt_N * sigma_A * torch.sqrt(sum_sigma_B2)

    threshold = e_max * (T_det + T_var23 + T_var4)
    return threshold


# ============================================================================
# 校验差计算
# ============================================================================

def compute_checksum_diff_mpmath(A, B, C):
    """使用 mpmath 高精度计算真实校验差"""
    M, K = A.shape
    K2, N = B.shape

    # 高精度 B 行求和
    B_rowsum_mp = [fsum([mpf(B[k, j].item()) for j in range(N)]) for k in range(K)]

    # 高精度 A @ B_rowsum
    checksum_mp = [fsum([mpf(A[i, k].item()) * B_rowsum_mp[k] for k in range(K)]) for i in range(M)]

    # 高精度 C 行求和
    C_rowsum_mp = [fsum([mpf(C[i, j].item()) for j in range(N)]) for i in range(M)]

    # 校验差
    diff = [abs(float(checksum_mp[i]) - float(C_rowsum_mp[i])) for i in range(M)]
    return diff


def compute_checksum_diff_fp64(A, B, C):
    """使用 FP64 计算校验差 (作为 FP32 的基准)"""
    A_f64 = A.to(torch.float64)
    B_f64 = B.to(torch.float64)
    C_f64 = C.to(torch.float64)

    B_rowsum = torch.sum(B_f64, dim=-1)
    checksum = torch.matmul(A_f64, B_rowsum)
    C_rowsum = torch.sum(C_f64, dim=-1)

    diff = torch.abs(checksum - C_rowsum)
    return diff.tolist()


# ============================================================================
# 测试函数
# ============================================================================

def run_test(n: int, dtype, device, num_trials: int, use_mpmath: bool = True, aabft_y: float = 21.0):
    """
    运行 A-ABFT vs V-ABFT 对比测试 (使用动态 e_max)
    """
    # 精度参数
    t = 53 if dtype == torch.float64 else 23

    # 动态 e_max
    e_max = get_dynamic_emax(n, dtype, device)

    actual_diffs = []
    aabft_thresholds = []
    vabft_thresholds = []
    aabft_fp = 0
    vabft_fp = 0

    for trial in range(num_trials):
        if trial % max(1, num_trials // 10) == 0:
            print(f"  Trial {trial}/{num_trials}...", flush=True)

        # 生成 [-1, 1] 均匀分布矩阵
        A = 2 * torch.rand(n, n, device=device, dtype=dtype) - 1
        B = 2 * torch.rand(n, n, device=device, dtype=dtype) - 1
        C = torch.matmul(A, B)

        # 计算真实校验差
        if dtype == torch.float64 and use_mpmath and HAS_MPMATH:
            true_diff = compute_checksum_diff_mpmath(A.cpu(), B.cpu(), C.cpu())
        else:
            true_diff = compute_checksum_diff_fp64(A.cpu(), B.cpu(), C.cpu())

        # A-ABFT 门限
        aabft_th = aabft_threshold(n, t, y=aabft_y)

        # V-ABFT 门限 (使用动态 e_max)
        vabft_th = vabft_threshold(A.cpu(), B.cpu(), e_max=e_max)

        actual_diffs.append(np.mean(true_diff))
        aabft_thresholds.append(aabft_th)
        vabft_thresholds.append(vabft_th.mean().item())

        # 检查误报
        if any(d > aabft_th for d in true_diff):
            aabft_fp += 1
        if any(true_diff[i] > vabft_th[i].item() for i in range(n)):
            vabft_fp += 1

    avg_diff = np.mean(actual_diffs)
    avg_aabft = np.mean(aabft_thresholds)
    avg_vabft = np.mean(vabft_thresholds)

    return {
        'n': n,
        'dtype': str(dtype).split('.')[-1],
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
    parser = argparse.ArgumentParser(description='A-ABFT vs V-ABFT with Dynamic e_max')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--dtype', type=str, default='float64', choices=['float64', 'float32'])
    parser.add_argument('--trials', type=int, default=100)
    parser.add_argument('--sizes', type=str, default='128,256,512,1024,2048',
                        help='Comma-separated matrix sizes')
    parser.add_argument('--no-mpmath', action='store_true', help='Disable mpmath for FP64')
    parser.add_argument('--aabft-y', type=float, default=21.0, help='A-ABFT y parameter')
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.float64 if args.dtype == 'float64' else torch.float32
    sizes = [int(s) for s in args.sizes.split(',')]
    use_mpmath = not args.no_mpmath and HAS_MPMATH

    print("=" * 110)
    print("A-ABFT vs V-ABFT Comparison (Dynamic e_max)")
    print("=" * 110)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    print(f"Trials: {args.trials}")
    print(f"Using mpmath: {use_mpmath and dtype == torch.float64}")
    print(f"A-ABFT y: {args.aabft_y}")
    print()
    print("Dynamic e_max formula:")
    if 'cuda' in str(device):
        if dtype == torch.float64:
            print("  e_max(N) = 1.0e-17 × √N + 2.5e-16")
        else:
            print("  e_max(N) = 5.0e-09 × √N + 1.2e-07")
    else:
        if dtype == torch.float64:
            print("  e_max = 6e-16 (constant)")
        else:
            print("  e_max = 4e-07 (constant)")
    print("=" * 110)

    print(f"\n{'Size':<10} {'e_max':<14} {'Actual Diff':<14} {'A-ABFT':<14} {'V-ABFT':<14} "
          f"{'A-Tight':<10} {'V-Tight':<10} {'A-FPR':<8} {'V-FPR':<8}")
    print("-" * 110)

    results = []
    for n in sizes:
        print(f"\nTesting {n}×{n} ({args.dtype})...")
        res = run_test(n, dtype, device, args.trials, use_mpmath, args.aabft_y)
        results.append(res)

        print(f"{n}×{n:<6} {res['e_max']:<14.2e} {res['avg_diff']:<14.2e} {res['aabft_thresh']:<14.2e} "
              f"{res['vabft_thresh']:<14.2e} {res['aabft_tightness']:<10.0f}× "
              f"{res['vabft_tightness']:<10.0f}× {res['aabft_fpr']:<8.2f}% {res['vabft_fpr']:<8.2f}%")

    # 汇总
    print("\n" + "=" * 110)
    print("Summary for Paper Tables")
    print("=" * 110)
    print(f"{'Size':<10} {'Actual Diff':<14} {'A-ABFT Tight':<14} {'V-ABFT Tight':<14} {'V-ABFT Advantage':<18}")
    print("-" * 75)

    for res in results:
        advantage = res['aabft_tightness'] / res['vabft_tightness']
        print(f"{res['n']}×{res['n']:<6} {res['avg_diff']:<14.2e} {res['aabft_tightness']:<14.0f}× "
              f"{res['vabft_tightness']:<14.0f}× {advantage:<18.1f}× tighter")

    # 输出 LaTeX 格式
    print("\n" + "=" * 110)
    print("LaTeX Table Data")
    print("=" * 110)
    print("% For Table 4/5")
    for res in results:
        advantage = res['aabft_tightness'] / res['vabft_tightness']
        print(f"$\\NUMNUM{{{res['n']}}}{{{res['n']}}}$ & "
              f"${res['avg_diff']:.2e}$ & "
              f"${res['aabft_thresh']:.2e}$ & "
              f"${res['vabft_thresh']:.2e}$ & "
              f"${int(res['aabft_tightness'])}\\times$ & "
              f"${int(res['vabft_tightness'])}\\times$ & "
              f"{res['aabft_fpr']:.0f}\\% & "
              f"{res['vabft_fpr']:.0f}\\% \\\\")


if __name__ == "__main__":
    main()
