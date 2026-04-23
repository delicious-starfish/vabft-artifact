"""
A-ABFT vs V-ABFT 对比测试 - GPU 低精度 (BF16/FP16)

BF16/FP16 e_max 测量结果：e_max = 2u（常数）
- BF16: e_max = 7.8e-3 (推荐 9.4e-3 with 1.2× margin)
- FP16: e_max = 9.8e-4 (推荐 1.2e-3 with 1.2× margin)
"""

import torch
import math
import numpy as np
from datetime import datetime
import argparse


# ============================================================================
# e_max 常数
# ============================================================================

EMAX_BF16 = 8e-3   # ~2u
EMAX_FP16 = 1e-3   # ~2u


def get_emax(dtype):
    if dtype == torch.bfloat16:
        return EMAX_BF16
    elif dtype == torch.float16:
        return EMAX_FP16
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def get_unit_roundoff(dtype):
    if dtype == torch.bfloat16:
        return 2 ** (-8)
    elif dtype == torch.float16:
        return 2 ** (-11)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


# ============================================================================
# A-ABFT 门限计算
# ============================================================================

def compute_y(A, B):
    """
    计算 A-ABFT 的 y 参数 (精确计算)

    y = max|A_ik × B_rowsum_k| = max|A| × max|B_rowsum|
    """
    B_rowsum = torch.sum(B, dim=-1)  # (K,)
    max_A = torch.max(torch.abs(A)).item()
    max_B_rowsum = torch.max(torch.abs(B_rowsum)).item()
    return max_A * max_B_rowsum


def aabft_threshold(n: int, t: int, y: float) -> float:
    """
    A-ABFT 门限公式
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
    """V-ABFT 门限计算"""
    M, K = A.shape
    K2, N = B.shape
    assert K == K2

    # 转换为 float32 计算统计量
    A_f32 = A.to(torch.float32)
    B_f32 = B.to(torch.float32)

    # A 的行统计量
    mu_A = torch.mean(A_f32, dim=-1)
    a_max = torch.max(A_f32, dim=-1).values
    a_min = torch.min(A_f32, dim=-1).values
    sigma_A2 = torch.clamp((a_max - mu_A) * (mu_A - a_min), min=1e-20)
    sigma_A = torch.sqrt(sigma_A2)

    # B 的行统计量
    mu_B = torch.mean(B_f32, dim=-1)
    b_max = torch.max(B_f32, dim=-1).values
    b_min = torch.min(B_f32, dim=-1).values
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

def compute_checksum_diff(A, B, C, dtype):
    """使用目标精度计算校验差，然后转换为 FP64"""
    # 所有操作在目标精度
    B_rowsum = torch.sum(B, dim=-1)
    checksum = torch.matmul(A, B_rowsum)
    C_rowsum = torch.sum(C, dim=-1)

    # 转换为 FP64 计算差值
    diff = torch.abs(checksum.to(torch.float64) - C_rowsum.to(torch.float64))
    return diff.tolist()


# ============================================================================
# 测试函数
# ============================================================================

def run_test(n: int, dtype, device, num_trials: int):
    """运行 A-ABFT vs V-ABFT 对比测试 (y 从数据精确计算)"""
    # 精度参数
    t = 8 if dtype == torch.bfloat16 else 11  # 尾数位数
    e_max = get_emax(dtype)

    actual_diffs = []
    aabft_thresholds = []
    vabft_thresholds = []
    y_values = []
    aabft_fp = 0
    vabft_fp = 0
    skipped = 0

    for trial in range(num_trials):
        if trial % max(1, num_trials // 10) == 0:
            print(f"  Trial {trial}/{num_trials}...", flush=True)

        # 生成正数矩阵 [0, 1] 避免数值问题
        A = torch.rand(n, n, device=device, dtype=torch.float32).to(dtype)
        B = torch.rand(n, n, device=device, dtype=torch.float32).to(dtype)
        C = torch.matmul(A, B)

        # 检查 inf/nan
        if torch.isnan(C).any() or torch.isinf(C).any():
            skipped += 1
            continue

        # 计算校验差
        true_diff = compute_checksum_diff(A, B, C, dtype)

        # 精确计算 y
        y = compute_y(A, B)
        y_values.append(y)

        # A-ABFT 门限 (使用精确计算的 y)
        aabft_th = aabft_threshold(n, t, y=y)

        # V-ABFT 门限
        vabft_th = vabft_threshold(A.cpu(), B.cpu(), e_max=e_max)

        actual_diffs.append(np.mean(true_diff))
        aabft_thresholds.append(aabft_th)
        vabft_thresholds.append(vabft_th.mean().item())

        # 检查误报
        if any(d > aabft_th for d in true_diff):
            aabft_fp += 1
        if any(true_diff[i] > vabft_th[i].item() for i in range(n)):
            vabft_fp += 1

    if skipped > 0:
        print(f"  [Warning: {skipped} trials skipped due to inf/nan]")

    valid_trials = num_trials - skipped
    if valid_trials == 0:
        return None

    avg_diff = np.mean(actual_diffs)
    avg_aabft = np.mean(aabft_thresholds)
    avg_vabft = np.mean(vabft_thresholds)
    avg_y = np.mean(y_values)

    return {
        'n': n,
        'dtype': str(dtype).split('.')[-1],
        'e_max': e_max,
        'avg_y': avg_y,
        'avg_diff': avg_diff,
        'aabft_thresh': avg_aabft,
        'vabft_thresh': avg_vabft,
        'aabft_tightness': avg_aabft / (avg_diff + 1e-30),
        'vabft_tightness': avg_vabft / (avg_diff + 1e-30),
        'aabft_fpr': aabft_fp / valid_trials * 100,
        'vabft_fpr': vabft_fp / valid_trials * 100,
        'valid_trials': valid_trials,
    }


def main():
    parser = argparse.ArgumentParser(description='A-ABFT vs V-ABFT Low Precision Comparison')
    parser.add_argument('--dtype', type=str, default='bfloat16',
                        choices=['bfloat16', 'float16'])
    parser.add_argument('--trials', type=int, default=100)
    parser.add_argument('--sizes', type=str, default='128,256,512,1024,2048',
                        help='Comma-separated matrix sizes')
    args = parser.parse_args()

    device = torch.device('cuda')
    dtype = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float16
    sizes = [int(s) for s in args.sizes.split(',')]
    u = get_unit_roundoff(dtype)
    e_max = get_emax(dtype)

    print("=" * 110)
    print(f"A-ABFT vs V-ABFT Comparison - GPU Low Precision ({args.dtype})")
    print("=" * 110)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    print(f"Unit roundoff u: {u:.4e}")
    print(f"e_max: {e_max:.4e} (~{e_max/u:.1f}u)")
    print(f"Trials: {args.trials}")
    print(f"A-ABFT y: computed from data (y = max|A| × max|B_rowsum|)")
    print("=" * 110)

    print(f"\n{'Size':<10} {'avg_y':<10} {'Actual Diff':<14} {'A-ABFT':<14} {'V-ABFT':<14} "
          f"{'A-Tight':<10} {'V-Tight':<10} {'A-FPR':<8} {'V-FPR':<8}")
    print("-" * 110)

    results = []
    for n in sizes:
        print(f"\nTesting {n}×{n} ({args.dtype})...")
        res = run_test(n, dtype, device, args.trials)

        if res is None:
            print(f"{n}×{n:<6} [All trials failed - likely overflow]")
            continue

        results.append(res)

        print(f"{n}×{n:<6} {res['avg_y']:<10.1f} {res['avg_diff']:<14.2e} {res['aabft_thresh']:<14.2e} "
              f"{res['vabft_thresh']:<14.2e} {res['aabft_tightness']:<10.0f}× "
              f"{res['vabft_tightness']:<10.0f}× {res['aabft_fpr']:<8.2f}% {res['vabft_fpr']:<8.2f}%")

    if not results:
        print("\nNo valid results!")
        return

    # 汇总
    print("\n" + "=" * 110)
    print("Summary for Paper")
    print("=" * 110)
    print(f"{'Size':<10} {'Actual Diff':<14} {'A-ABFT Tight':<14} {'V-ABFT Tight':<14} {'V-ABFT Advantage':<18}")
    print("-" * 75)

    for res in results:
        advantage = res['aabft_tightness'] / res['vabft_tightness']
        print(f"{res['n']}×{res['n']:<6} {res['avg_diff']:<14.2e} {res['aabft_tightness']:<14.0f}× "
              f"{res['vabft_tightness']:<14.0f}× {advantage:<18.1f}× tighter")

    # 输出 LaTeX
    print("\n" + "=" * 110)
    print("LaTeX Table Data")
    print("=" * 110)
    for res in results:
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
