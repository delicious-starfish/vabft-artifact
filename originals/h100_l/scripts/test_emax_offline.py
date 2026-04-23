"""
Offline detection e_max measurement — both paths use low precision
误差来源: checksum = A_lp @ (B_lp @ ones), rowsum = (A_lp @ B_lp) @ ones
两条路径都经过低精度运算，模拟 offline 检测场景
"""
import torch
import numpy as np
from datetime import datetime
import argparse


def measure_emax_offline(M, K, N, dtype, device, num_trials=10000, batch_size=200):
    """Offline: checksum 和 rowsum 都从低精度矩阵计算（batch 版本）"""
    emax_values = []
    ones = torch.ones(N, 1, device=device, dtype=dtype)

    for start in range(0, num_trials, batch_size):
        bs = min(batch_size, num_trials - start)

        # 批量生成: (bs, M, K) 和 (bs, K, N)
        A_f32 = torch.abs(torch.randn(bs, M, K, dtype=torch.float32)) + 0.5
        B_f32 = torch.abs(torch.randn(bs, K, N, dtype=torch.float32)) + 0.5

        if dtype == torch.float16:
            scale = min(1.0, (65504 / (K * N * 4)) ** 0.5)
            A_f32 = A_f32 * scale
            B_f32 = B_f32 * scale

        A_lp = A_f32.to(dtype).to(device)
        B_lp = B_f32.to(dtype).to(device)

        # 路径1: checksum = A @ (B @ ones), shape (bs, M)
        checksum = torch.matmul(A_lp, torch.matmul(B_lp, ones)).squeeze(-1)

        # 路径2: rowsum = (A @ B) @ ones, shape (bs, M)
        C = torch.matmul(A_lp, B_lp)
        rowsum = torch.matmul(C, ones).squeeze(-1)

        diff = torch.abs(checksum.to(torch.float64) - rowsum.to(torch.float64))
        emax = diff / (torch.abs(checksum.to(torch.float64)) + 1e-30)
        emax_values.extend(emax.cpu().flatten().tolist())

    return {
        'max': np.max(emax_values),
        'p999': np.percentile(emax_values, 99.9),
        'mean': np.mean(emax_values),
    }


def get_unit_roundoff(dtype):
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
    from scipy.optimize import curve_fit
    K_arr = np.array(K_values, dtype=float)
    emax_arr = np.array(emax_values, dtype=float)
    results = {}

    const_val = np.mean(emax_arr)
    ss_tot = np.sum((emax_arr - np.mean(emax_arr)) ** 2)
    results['constant'] = {'value': const_val, 'r2': 0.0}

    def sqrt_model(x, a, b):
        return a * np.sqrt(x) + b
    try:
        popt, _ = curve_fit(sqrt_model, K_arr, emax_arr)
        pred = sqrt_model(K_arr, *popt)
        r2 = 1 - np.sum((emax_arr - pred) ** 2) / ss_tot if ss_tot > 0 else 0
        results['sqrt_K'] = {'a': popt[0], 'b': popt[1], 'r2': r2,
                             'formula': f'e_max = {popt[0]:.4e} * sqrt(K) + {popt[1]:.4e}'}
    except Exception:
        results['sqrt_K'] = {'r2': -1}

    def linear_model(x, a, b):
        return a * x + b
    try:
        popt, _ = curve_fit(linear_model, K_arr, emax_arr)
        pred = linear_model(K_arr, *popt)
        r2 = 1 - np.sum((emax_arr - pred) ** 2) / ss_tot if ss_tot > 0 else 0
        results['linear_K'] = {'a': popt[0], 'b': popt[1], 'r2': r2,
                               'formula': f'e_max = {popt[0]:.4e} * K + {popt[1]:.4e}'}
    except Exception:
        results['linear_K'] = {'r2': -1}

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--dtype', type=str, default='bfloat16',
                        choices=['float64', 'float32', 'float16', 'bfloat16'])
    parser.add_argument('--trials', type=int, default=50000)
    parser.add_argument('--M', type=int, default=1024)
    parser.add_argument('--N', type=int, default=256)
    parser.add_argument('--K-values', type=str, default='256,512,1024,2048,4096,8192,16384')
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype_map = {'float64': torch.float64, 'float32': torch.float32,
                 'float16': torch.float16, 'bfloat16': torch.bfloat16}
    dtype = dtype_map[args.dtype]
    u = get_unit_roundoff(dtype)
    M, N = args.M, args.N
    K_values = [int(s) for s in args.K_values.split(',')]

    print("=" * 100)
    print(f"e_max OFFLINE Measurement — fixed M={M}, N={N}, sweep K (both paths use low precision)")
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

    for K in K_values:
        print(f"Testing K={K}...", end="", flush=True)
        res = measure_emax_offline(M, K, N, dtype, device, num_trials=args.trials)
        emax_max_values.append(res['max'])
        print(f"\r{K:<8} {res['max']:<14.4e} {res['max']/u:<10.2f} "
              f"{res['p999']:<14.4e} {res['p999']/u:<10.2f} "
              f"{res['mean']:<14.4e} {res['mean']/u:<10.2f}")

    # 分析
    emax_arr = np.array(emax_max_values)
    cv = np.std(emax_arr) / np.mean(emax_arr) * 100
    variation = emax_arr.max() / emax_arr.min()

    print(f"\n{'='*100}")
    print("ANALYSIS")
    print(f"{'='*100}")
    print(f"e_max/u range: {emax_arr.min()/u:.2f} — {emax_arr.max()/u:.2f}")
    print(f"Variation (max/min): {variation:.2f}x")
    print(f"CV (across K values): {cv:.1f}%")

    try:
        fit = fit_and_analyze(K_values, emax_max_values)
        sorted_models = sorted(fit.items(), key=lambda x: x[1].get('r2', -1), reverse=True)
        print("\nModel comparison (sorted by R²):")
        for i, (name, res) in enumerate(sorted_models):
            tag = "  <-- BEST" if i == 0 else ""
            print(f"  {name:12s}: R² = {res.get('r2', -1):.4f}  {res.get('formula', '')}{tag}")

        sqrt_r2 = fit.get('sqrt_K', {}).get('r2', -1)
        print(f"\nR²(sqrt_K) for paper table: {sqrt_r2:.2f}")
    except ImportError:
        print("\nscipy not available — skipping fit.")

    if variation < 2.0 and cv < 15:
        scaling = "constant"
    else:
        scaling = "sqrt_K"
    print(f"\nConclusion: scaling = {scaling}")

    # Markdown 表格
    print(f"\n{'='*100}")
    print("DATA FOR PAPER (Markdown)")
    print(f"{'='*100}")
    print(f"Config: M={M}, N={N}, dtype={args.dtype}, device={args.device}, {args.trials} trials/K (OFFLINE)")
    print(f"\n| K | e_max | e_max/u |")
    print(f"|---:|------:|--------:|")
    for K, val in zip(K_values, emax_max_values):
        print(f"| {K} | {val:.2e} | {val/u:.1f} |")
    print(f"\nCV = {cv:.1f}%,  scaling = {scaling}")


if __name__ == "__main__":
    main()
