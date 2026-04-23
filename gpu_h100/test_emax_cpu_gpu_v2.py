"""
e_max 测量 - CPU/GPU 统一版本 (v2)

固定 M=256, N=256，扫描 K ∈ {256, 512, 1024, 2048, 4096, 8192, 16384}
与 NPU 测量方法对齐，消除方阵 N×N 的维度耦合问题

公式: e_max = max_{m,trials} |D1[m]| / |checksum[m]|
其中 D1 = checksum - rowsum(C)
"""

import torch
import math
import numpy as np
from datetime import datetime
import argparse


def measure_emax(M: int, K: int, N: int, dtype, device, num_trials: int = 2000):
    """
    测量 (M, K, N) 矩阵的 e_max

    路径 1 (pre-computed): checksum[m] = sum_k A[m,k] * (sum_n B[k,n])
    路径 2 (post-hoc):     rowsum[m]   = sum_n C[m,n] = sum_n (sum_k A[m,k]*B[k,n])

    e_max = |checksum - rowsum| / |checksum|

    对于低精度格式 (FP16/BF16):
      - GEMM 用低精度执行（模拟硬件行为）
      - checksum/rowsum 用 FP32 计算（与实际部署一致：Tensor Core 内部 FP32 累加）
      - 输入缩放以防止 FP16 溢出
    """
    is_low_prec = dtype in (torch.float16, torch.bfloat16)
    compute_dtype = torch.float32  # checksum/rowsum 始终用 FP32
    emax_values = []

    for _ in range(num_trials):
        # 生成 FP32 正数矩阵
        A_f32 = torch.abs(torch.randn(M, K, dtype=torch.float32, device=device)) + 0.5
        B_f32 = torch.abs(torch.randn(K, N, dtype=torch.float32, device=device)) + 0.5

        if is_low_prec:
            # FP16 动态范围小 (max=65504)，缩放输入防止溢出
            # C 元素量级 ~ scale² * K，checksum ~ scale² * K * N
            # 要求 < 65504 → scale = sqrt(65504 / (K * N * 4))
            if dtype == torch.float16:
                scale = min(1.0, (65504 / (K * N * 4)) ** 0.5)
                A_f32 = A_f32 * scale
                B_f32 = B_f32 * scale

            # GEMM 用低精度（模拟 Tensor Core 行为）
            A_lp = A_f32.to(dtype)
            B_lp = B_f32.to(dtype)
            C = torch.matmul(A_lp, B_lp).to(torch.float32)

            # checksum/rowsum 用 FP32（与实际部署一致）
            B_rowsum = torch.sum(B_f32, dim=1)
            checksum = torch.matmul(A_f32, B_rowsum)
            C_rowsum = torch.sum(C, dim=1)
        else:
            # 高精度 (FP32/FP64): 全部用同一精度
            A = A_f32.to(dtype)
            B = B_f32.to(dtype)
            C = torch.matmul(A, B)
            B_rowsum = torch.sum(B, dim=1)
            checksum = torch.matmul(A, B_rowsum)
            C_rowsum = torch.sum(C, dim=1)

        # ── 最后一步：用更高精度做减法，避免灾难性抵消 ──
        # FP32 GEMM → FP64 减法；FP64 GEMM → numpy float128/mpmath
        # BF16/FP16 → FP32 减法（误差 >> FP32 精度，无抵消问题）
        if dtype == torch.float32:
            # FP32: 提升到 FP64 再做减法
            diff = torch.abs(checksum.to(torch.float64) - C_rowsum.to(torch.float64))
            emax = diff / (torch.abs(checksum.to(torch.float64)) + 1e-30)
        elif dtype == torch.float64:
            # FP64: PyTorch 无 FP128，用 numpy longdouble (x86 上是 80-bit extended)
            cs_np = checksum.cpu().numpy().astype(np.longdouble)
            cr_np = C_rowsum.cpu().numpy().astype(np.longdouble)
            diff_np = np.abs(cs_np - cr_np)
            emax_np = diff_np / (np.abs(cs_np) + 1e-30)
            emax_values.extend(emax_np.tolist())
            continue
        else:
            # BF16/FP16: checksum/C_rowsum 已经是 FP32，精度足够
            diff = torch.abs(checksum - C_rowsum)
            emax = diff / (torch.abs(checksum) + 1e-30)

        emax_values.extend(emax.cpu().tolist())

    return {
        'max': np.max(emax_values),
        'p999': np.percentile(emax_values, 99.9),
        'p99': np.percentile(emax_values, 99),
        'mean': np.mean(emax_values),
        'std': np.std(emax_values),
    }


def get_unit_roundoff(dtype):
    """获取单位舍入误差 u"""
    if dtype == torch.float64:
        return 2**(-53)
    elif dtype == torch.float32:
        return 2**(-24)
    elif dtype == torch.float16:
        return 2**(-11)
    elif dtype == torch.bfloat16:
        return 2**(-8)
    return 1e-7


def fit_and_analyze(K_values, emax_values):
    """对 e_max vs K 做拟合分析，返回各模型的 R²"""
    from scipy.optimize import curve_fit

    K_arr = np.array(K_values, dtype=float)
    emax_arr = np.array(emax_values, dtype=float)
    results = {}

    # 1. 常数模型
    const_val = np.mean(emax_arr)
    ss_res = np.sum((emax_arr - const_val) ** 2)
    ss_tot = np.sum((emax_arr - np.mean(emax_arr)) ** 2)
    results['constant'] = {
        'value': const_val,
        'r2': 0.0,  # 常数模型 R² 定义为 0（baseline）
        'formula': f'e_max = {const_val:.4e}',
    }

    # 2. sqrt(K) 模型: e_max = a * sqrt(K) + b
    def sqrt_model(x, a, b):
        return a * np.sqrt(x) + b
    try:
        popt, _ = curve_fit(sqrt_model, K_arr, emax_arr)
        pred = sqrt_model(K_arr, *popt)
        r2 = 1 - np.sum((emax_arr - pred) ** 2) / ss_tot if ss_tot > 0 else 0
        results['sqrt_K'] = {
            'a': popt[0], 'b': popt[1], 'r2': r2,
            'formula': f'e_max = {popt[0]:.4e} * sqrt(K) + {popt[1]:.4e}',
        }
    except Exception:
        results['sqrt_K'] = {'r2': -1, 'formula': 'fit failed'}

    # 3. 线性模型: e_max = a * K + b
    def linear_model(x, a, b):
        return a * x + b
    try:
        popt, _ = curve_fit(linear_model, K_arr, emax_arr)
        pred = linear_model(K_arr, *popt)
        r2 = 1 - np.sum((emax_arr - pred) ** 2) / ss_tot if ss_tot > 0 else 0
        results['linear_K'] = {
            'a': popt[0], 'b': popt[1], 'r2': r2,
            'formula': f'e_max = {popt[0]:.4e} * K + {popt[1]:.4e}',
        }
    except Exception:
        results['linear_K'] = {'r2': -1, 'formula': 'fit failed'}

    return results


# ── PLACEHOLDER_MAIN ──


def main():
    parser = argparse.ArgumentParser(description='Measure e_max on CPU/GPU (v2: fixed M,N, sweep K)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--dtype', type=str, default='float32',
                        choices=['float64', 'float32', 'float16', 'bfloat16'])
    parser.add_argument('--trials', type=int, default=5000, help='Number of trials per K')
    parser.add_argument('--M', type=int, default=1024)
    parser.add_argument('--N', type=int, default=256)
    parser.add_argument('--K-values', type=str, default='256,512,1024,2048,4096,8192,16384',
                        help='Comma-separated K values to sweep')
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype_map = {
        'float64': torch.float64,
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]
    u = get_unit_roundoff(dtype)
    M = args.M
    N = args.N
    K_values = [int(s) for s in args.K_values.split(',')]

    print("=" * 100)
    print(f"e_max Measurement v2 — fixed M={M}, N={N}, sweep K")
    print("=" * 100)
    print(f"Date:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device:  {device}")
    print(f"Dtype:   {dtype}  (u = {u:.4e})")
    print(f"Trials:  {args.trials} per K value")
    print(f"K sweep: {K_values}")
    print("=" * 100)

    header = f"{'K':<8} {'max(e_max)':<14} {'max/u':<10} {'p99.9':<14} {'p99.9/u':<10} {'mean':<14} {'mean/u':<10}"
    print(f"\n{header}")
    print("-" * len(header))

    emax_max_values = []
    all_results = []

    for K in K_values:
        print(f"Testing K={K}...", end="", flush=True)
        res = measure_emax(M, K, N, dtype, device, num_trials=args.trials)
        all_results.append((K, res))
        emax_max_values.append(res['max'])
        print(f"\r{K:<8} {res['max']:<14.4e} {res['max']/u:<10.2f} "
              f"{res['p999']:<14.4e} {res['p999']/u:<10.2f} "
              f"{res['mean']:<14.4e} {res['mean']/u:<10.2f}")

    # ── 分析 ──
    emax_arr = np.array(emax_max_values)
    cv = np.std(emax_arr) / np.mean(emax_arr) * 100
    variation = emax_arr.max() / emax_arr.min()

    print(f"\n{'='*100}")
    print("ANALYSIS")
    print(f"{'='*100}")
    print(f"e_max/u range: {emax_arr.min()/u:.2f} — {emax_arr.max()/u:.2f}")
    print(f"Variation (max/min): {variation:.2f}x")
    print(f"CV (across K values): {cv:.1f}%")

    # 拟合
    try:
        fit = fit_and_analyze(K_values, emax_max_values)
        print("\nModel comparison (sorted by R²):")
        sorted_models = sorted(fit.items(), key=lambda x: x[1].get('r2', -1), reverse=True)
        for i, (name, res) in enumerate(sorted_models):
            tag = "  <-- BEST" if i == 0 else ""
            print(f"  {name:12s}: R² = {res.get('r2', -1):.4f}  {res.get('formula', '')}{tag}")

        best_name, best = sorted_models[0]
        print(f"\nRecommended: {best_name}")
        print(f"Formula:     {best['formula']}")

        # 输出 R²(sqrt_K) 用于论文表格
        sqrt_r2 = fit.get('sqrt_K', {}).get('r2', -1)
        print(f"\nR²(sqrt_K) for paper table: {sqrt_r2:.2f}")
    except ImportError:
        print("\nscipy not available — skipping fit. Raw data:")
        for K, val in zip(K_values, emax_max_values):
            print(f"  K={K}: e_max = {val:.4e} ({val/u:.2f}u)")

    # 判定 scaling 类型
    if variation < 2.0 and cv < 15:
        scaling = "constant"
        rec = f"{emax_arr.max():.4e} ({emax_arr.max()/u:.1f}u)"
        print(f"\nConclusion: e_max ≈ constant across K.  Recommended value = {rec}")
    else:
        scaling = "sqrt_K"
        print(f"\nConclusion: e_max scales with sqrt(K). Use fitted formula.")

    # ── 论文用 Markdown 表格 ──
    print(f"\n{'='*100}")
    print("DATA FOR PAPER (Markdown)")
    print(f"{'='*100}")
    print(f"Config: M={M}, N={N}, dtype={args.dtype}, device={args.device}, {args.trials} trials/K\n")
    print(f"| K | e_max | e_max/u |")
    print(f"|---:|------:|--------:|")
    for K, val in zip(K_values, emax_max_values):
        print(f"| {K} | {val:.2e} | {val/u:.1f} |")
    print(f"\nCV = {cv:.1f}%,  scaling = {scaling}")


if __name__ == "__main__":
    main()
