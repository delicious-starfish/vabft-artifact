"""
GPU 低精度 (BF16/FP16) e_max 测量脚本 - 修正版 v2

使用正数矩阵避免抵消问题，并检查 inf/nan
"""

import torch
import math
import numpy as np
from datetime import datetime
import argparse


def measure_emax(N: int, dtype, device, num_trials: int = 2000):
    """
    测量 e_max = max |checksum - C_rowsum| / |checksum|

    使用正数矩阵 [0, 1] 避免抵消问题
    """
    emax_list = []
    nan_count = 0
    inf_count = 0

    for trial in range(num_trials):
        # 生成 [0, 1] 均匀分布正数矩阵
        A = torch.rand(N, N, device=device, dtype=torch.float32).to(dtype)
        B = torch.rand(N, N, device=device, dtype=torch.float32).to(dtype)

        # 低精度矩阵乘法
        C = torch.matmul(A, B)

        # 检查 inf/nan
        if torch.isnan(C).any() or torch.isinf(C).any():
            nan_count += 1
            continue

        # 低精度校验和计算 (与 GEMM 使用相同精度)
        B_rowsum = torch.sum(B, dim=-1)  # (N,) in dtype
        checksum = torch.matmul(A, B_rowsum)  # (N,) in dtype
        C_rowsum = torch.sum(C, dim=-1)  # (N,) in dtype

        # 检查 inf/nan
        if torch.isnan(checksum).any() or torch.isinf(checksum).any():
            inf_count += 1
            continue

        # 转换为 FP64 计算相对误差
        checksum_f64 = checksum.to(torch.float64)
        C_rowsum_f64 = C_rowsum.to(torch.float64)

        diff = torch.abs(checksum_f64 - C_rowsum_f64)
        rel_err = diff / (torch.abs(checksum_f64) + 1e-30)

        emax = rel_err.max().item()

        # 过滤异常值
        if not (math.isnan(emax) or math.isinf(emax) or emax > 1):
            emax_list.append(emax)

    if nan_count > 0 or inf_count > 0:
        print(f"  [Warning: {nan_count} nan, {inf_count} inf trials skipped]", end='')

    return np.array(emax_list) if emax_list else np.array([0])


def get_unit_roundoff(dtype):
    """获取单位舍入误差 u"""
    if dtype == torch.bfloat16:
        return 2 ** (-8)  # BF16: 7 位尾数 + 1 隐含位
    elif dtype == torch.float16:
        return 2 ** (-11)  # FP16: 10 位尾数 + 1 隐含位
    elif dtype == torch.float32:
        return 2 ** (-24)  # FP32: 23 位尾数 + 1 隐含位
    else:
        return 2 ** (-53)  # FP64


def main():
    parser = argparse.ArgumentParser(description='GPU Low Precision e_max Measurement')
    parser.add_argument('--dtype', type=str, default='bfloat16',
                        choices=['bfloat16', 'float16'])
    parser.add_argument('--trials', type=int, default=2000)
    parser.add_argument('--sizes', type=str, default='128,256,512,1024,2048,4096',
                        help='Comma-separated matrix sizes')
    args = parser.parse_args()

    device = torch.device('cuda')
    dtype = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float16
    sizes = [int(s) for s in args.sizes.split(',')]
    u = get_unit_roundoff(dtype)

    print("=" * 100)
    print(f"GPU Low Precision e_max Measurement (v2 - positive matrices)")
    print("=" * 100)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    print(f"Unit roundoff u: {u:.4e}")
    print(f"Trials per size: {args.trials}")
    print(f"Matrix sizes (N×N): {sizes}")
    print("=" * 100)
    print()

    print(f"{'N':<10} {'max(e_max)':<15} {'max/u':<10} {'p99.9':<15} {'p99.9/u':<10} {'mean':<15} {'mean/u':<10}")
    print("-" * 90)

    results = []
    for N in sizes:
        print(f"Testing N={N}...", end='\r', flush=True)

        emax_arr = measure_emax(N, dtype, device, args.trials)

        if len(emax_arr) == 0:
            print(f"{N:<10} [All trials failed]")
            continue

        max_emax = np.max(emax_arr)
        p999_emax = np.percentile(emax_arr, 99.9)
        mean_emax = np.mean(emax_arr)

        print(f"{N:<10} {max_emax:<15.4e} {max_emax/u:<10.2f} {p999_emax:<15.4e} {p999_emax/u:<10.2f} {mean_emax:<15.4e} {mean_emax/u:<10.2f}")

        results.append({
            'N': N,
            'max_emax': max_emax,
            'max_emax_over_u': max_emax / u,
            'p999_emax': p999_emax,
            'mean_emax': mean_emax,
            'valid_trials': len(emax_arr),
        })

    if not results:
        print("No valid results!")
        return

    # 分析是否是常数还是随 √N 增长
    print("\n" + "=" * 100)
    print("Analysis")
    print("=" * 100)

    max_values = [r['max_emax'] for r in results]
    max_over_u = [r['max_emax_over_u'] for r in results]

    # 计算变异系数
    cv = np.std(max_over_u) / np.mean(max_over_u) * 100
    print(f"Coefficient of Variation (CV) of max/u: {cv:.1f}%")

    if cv < 30:
        recommended = np.max(max_values) * 1.2
        print(f"Conclusion: e_max is approximately CONSTANT (~{np.mean(max_over_u):.1f}u)")
        print(f"Recommended e_max = {recommended:.2e} (with 1.2× safety margin)")
    else:
        # 尝试 √N 拟合
        sqrt_N = np.array([math.sqrt(r['N']) for r in results])
        try:
            from scipy import stats
            slope, intercept, r_value, _, _ = stats.linregress(sqrt_N, max_values)
            print(f"√N linear fit: e_max = {slope:.2e} × √N + {intercept:.2e}, R² = {r_value**2:.3f}")
            if r_value**2 > 0.7:
                print(f"Conclusion: e_max grows with √N")
                print(f"Recommended formula: e_max = {slope*1.2:.2e} × √N + {intercept*1.2:.2e}")
            else:
                print(f"Conclusion: e_max pattern unclear (CV={cv:.1f}%, R²={r_value**2:.3f})")
                recommended = np.max(max_values) * 1.5
                print(f"Using conservative constant: e_max = {recommended:.2e}")
        except ImportError:
            print("scipy not available for fitting")

    # 输出表格格式
    print("\n" + "=" * 100)
    print("Table for Paper")
    print("=" * 100)
    print(f"| N | e_max | e_max/u | valid_trials |")
    print("|---|-------|---------|--------------|")
    for r in results:
        print(f"| {r['N']} | {r['max_emax']:.2e} | {r['max_emax_over_u']:.1f} | {r['valid_trials']} |")


if __name__ == "__main__":
    main()
