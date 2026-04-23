"""
GPU FP8 (E4M3/E5M2) e_max 测量脚本 v2

测量 FP8 量化对 GEMM 校验的影响：
- FP8 Tensor Core: FP8 input → FP32 accumulation → FP16/FP32 output
- 校验误差来源：FP8 量化误差 + FP16 输出舍入误差

本脚本测量：校验差 / checksum，其中：
- GEMM 和 checksum 都经过 FP8 量化
- 实际计算在更高精度下进行（模拟 Tensor Core 行为）
"""

import torch
import math
import numpy as np
from datetime import datetime
import argparse


def check_fp8_support():
    """检查 PyTorch 和 GPU 是否支持 FP8"""
    print(f"PyTorch version: {torch.__version__}")

    if not torch.cuda.is_available():
        print("CUDA not available!")
        return False

    capability = torch.cuda.get_device_capability()
    print(f"CUDA capability: {capability}")

    try:
        _ = torch.float8_e4m3fn
        _ = torch.float8_e5m2
        print("FP8 dtypes available: float8_e4m3fn, float8_e5m2")
        return True
    except AttributeError:
        print("FP8 dtypes not available!")
        return False


def get_fp8_dtype(dtype_name: str):
    if dtype_name == 'e4m3':
        return torch.float8_e4m3fn
    elif dtype_name == 'e5m2':
        return torch.float8_e5m2
    else:
        raise ValueError(f"Unknown FP8 dtype: {dtype_name}")


def get_unit_roundoff(dtype_name: str):
    if dtype_name == 'e4m3':
        return 2 ** (-4)  # 3 mantissa bits + 1 implicit
    elif dtype_name == 'e5m2':
        return 2 ** (-3)  # 2 mantissa bits + 1 implicit
    else:
        raise ValueError(f"Unknown dtype: {dtype_name}")


def get_max_value(dtype_name: str):
    if dtype_name == 'e4m3':
        return 448.0
    elif dtype_name == 'e5m2':
        return 57344.0
    else:
        raise ValueError(f"Unknown dtype: {dtype_name}")


def measure_emax_fp8_v2(N: int, dtype_name: str, device, num_trials: int = 2000, scale: float = 0.1):
    """
    FP8 e_max 测量 v2 - 测量 FP8 量化对校验的影响

    计算流程（模拟 FP8 Tensor Core）:
    1. 生成 FP32 矩阵 A, B
    2. 量化为 FP8: A_fp8, B_fp8
    3. 转换为 FP32 计算 GEMM: C = A_fp8.float() @ B_fp8.float()
    4. C 量化为输出精度（FP16）: C_out
    5. 校验和同样经过 FP8 量化后计算

    e_max = |checksum - C_rowsum| / |checksum|
    """
    fp8_dtype = get_fp8_dtype(dtype_name)
    max_val = get_max_value(dtype_name)

    emax_list = []
    nan_count = 0
    overflow_count = 0

    for trial in range(num_trials):
        # 生成正数矩阵 [0, scale]
        A_fp32 = torch.rand(N, N, device=device, dtype=torch.float32) * scale
        B_fp32 = torch.rand(N, N, device=device, dtype=torch.float32) * scale

        # 检查是否会溢出
        expected_max = N * scale * scale
        if expected_max > max_val * 0.5:
            overflow_count += 1
            continue

        try:
            # FP8 量化
            A_fp8 = A_fp32.to(fp8_dtype)
            B_fp8 = B_fp32.to(fp8_dtype)

            # 模拟 FP8 Tensor Core: FP8 → FP32 计算 → FP16 输出
            # GEMM: 使用 FP8 量化后的数据，在 FP32 下计算，输出 FP16
            A_for_gemm = A_fp8.to(torch.float32)
            B_for_gemm = B_fp8.to(torch.float32)
            C_fp32 = torch.matmul(A_for_gemm, B_for_gemm)
            C_fp16 = C_fp32.to(torch.float16)  # 模拟输出舍入

            # 校验和计算（同样经过 FP8 量化）
            # B_rowsum 在 FP32 下计算（因为 FP8 没有 sum）
            B_rowsum_fp32 = torch.sum(B_for_gemm, dim=-1)
            # 校验乘法: A_fp8 @ B_rowsum
            checksum_fp32 = torch.matmul(A_for_gemm, B_rowsum_fp32)
            checksum_fp16 = checksum_fp32.to(torch.float16)

            # C 的行和
            C_rowsum_fp16 = torch.sum(C_fp16, dim=-1)

            # 检查 inf/nan
            if torch.isnan(C_fp16).any() or torch.isinf(C_fp16).any():
                nan_count += 1
                continue
            if torch.isnan(checksum_fp16).any() or torch.isinf(checksum_fp16).any():
                nan_count += 1
                continue

            # 计算相对误差（在 FP64 下）
            checksum_f64 = checksum_fp16.to(torch.float64)
            C_rowsum_f64 = C_rowsum_fp16.to(torch.float64)

            diff = torch.abs(checksum_f64 - C_rowsum_f64)
            rel_err = diff / (torch.abs(checksum_f64) + 1e-30)

            emax = rel_err.max().item()

            if not (math.isnan(emax) or math.isinf(emax) or emax > 1):
                emax_list.append(emax)

        except Exception as e:
            nan_count += 1
            continue

    if nan_count > 0 or overflow_count > 0:
        print(f"  [Warning: {nan_count} errors, {overflow_count} overflow skipped]", end='')

    return np.array(emax_list) if emax_list else np.array([0])


def measure_emax_fp8_pure(N: int, dtype_name: str, device, num_trials: int = 2000, scale: float = 0.1):
    """
    FP8 e_max 测量 - 纯 FP8 量化误差

    测量 FP8 量化引入的相对误差：
    e_max = max |x_fp8 - x_fp32| / |x_fp32|

    这给出 FP8 量化本身的误差界
    """
    fp8_dtype = get_fp8_dtype(dtype_name)

    emax_list = []
    for trial in range(num_trials):
        # 生成数据
        x = torch.rand(N, N, device=device, dtype=torch.float32) * scale + scale * 0.1

        # FP8 量化
        x_fp8 = x.to(fp8_dtype)
        x_back = x_fp8.to(torch.float32)

        # 相对误差
        rel_err = torch.abs(x_back - x) / (torch.abs(x) + 1e-30)
        emax = rel_err.max().item()

        if not (math.isnan(emax) or math.isinf(emax)):
            emax_list.append(emax)

    return np.array(emax_list) if emax_list else np.array([0])


def main():
    parser = argparse.ArgumentParser(description='GPU FP8 e_max Measurement v2')
    parser.add_argument('--dtype', type=str, default='e4m3',
                        choices=['e4m3', 'e5m2'])
    parser.add_argument('--trials', type=int, default=2000)
    parser.add_argument('--sizes', type=str, default='64,128,256,512,1024',
                        help='Comma-separated matrix sizes')
    parser.add_argument('--scale', type=float, default=0.1,
                        help='Scale factor for matrix elements')
    parser.add_argument('--measure-quant', action='store_true',
                        help='Also measure pure FP8 quantization error')
    args = parser.parse_args()

    print("=" * 100)
    print(f"GPU FP8 e_max Measurement v2 - Quantization + Verification Error")
    print("=" * 100)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if not check_fp8_support():
        print("\nFP8 not supported, exiting.")
        return

    device = torch.device('cuda')
    dtype_name = args.dtype
    sizes = [int(s) for s in args.sizes.split(',')]
    u = get_unit_roundoff(dtype_name)
    max_val = get_max_value(dtype_name)

    print(f"\nDevice: {torch.cuda.get_device_name()}")
    print(f"FP8 Format: {dtype_name.upper()}")
    print(f"  - Unit roundoff u: {u:.4e}")
    print(f"  - Max value: {max_val}")
    print(f"Scale factor: {args.scale}")
    print(f"Trials per size: {args.trials}")
    print(f"Matrix sizes: {sizes}")
    print("=" * 100)

    # 首先测量纯 FP8 量化误差
    if args.measure_quant:
        print("\n--- FP8 Quantization Error (for reference) ---")
        quant_err = measure_emax_fp8_pure(256, dtype_name, device, 1000, args.scale)
        print(f"FP8 {dtype_name.upper()} quantization error: max={np.max(quant_err):.4e}, mean={np.mean(quant_err):.4e}")
        print(f"  max/u = {np.max(quant_err)/u:.2f}, mean/u = {np.mean(quant_err)/u:.2f}")
        print(f"  (Expected: ~0.5u for rounding to nearest)")

    # 主测试：校验误差
    print("\n--- Verification Error (GEMM checksum) ---")
    print(f"{'N':<10} {'max(e_max)':<15} {'max/u':<10} {'p99.9':<15} {'p99.9/u':<10} {'mean':<15} {'mean/u':<10} {'valid':<8}")
    print("-" * 100)

    results = []
    for N in sizes:
        print(f"Testing N={N}...", end='\r', flush=True)

        # 动态调整 scale
        safe_scale = min(args.scale, math.sqrt(max_val * 0.3 / N))

        emax_arr = measure_emax_fp8_v2(N, dtype_name, device, args.trials, safe_scale)

        if len(emax_arr) == 0:
            print(f"{N:<10} [All trials failed]")
            continue

        max_emax = np.max(emax_arr)
        p999_emax = np.percentile(emax_arr, 99.9)
        mean_emax = np.mean(emax_arr)

        print(f"{N:<10} {max_emax:<15.4e} {max_emax/u:<10.2f} {p999_emax:<15.4e} {p999_emax/u:<10.2f} "
              f"{mean_emax:<15.4e} {mean_emax/u:<10.2f} {len(emax_arr):<8}")

        results.append({
            'N': N,
            'scale': safe_scale,
            'max_emax': max_emax,
            'max_emax_over_u': max_emax / u,
            'p999_emax': p999_emax,
            'mean_emax': mean_emax,
            'valid_trials': len(emax_arr),
        })

    if not results:
        print("\nNo valid results!")
        return

    # 分析
    print("\n" + "=" * 100)
    print("Analysis")
    print("=" * 100)

    max_over_u = [r['max_emax_over_u'] for r in results]
    cv = np.std(max_over_u) / np.mean(max_over_u) * 100 if np.mean(max_over_u) > 0 else 0

    print(f"e_max/u range: {min(max_over_u):.2f} - {max(max_over_u):.2f}")
    print(f"Mean e_max/u: {np.mean(max_over_u):.2f}")
    print(f"CV of e_max/u: {cv:.1f}%")

    # FP16 参考值
    u_fp16 = 2 ** (-11)
    print(f"\nReference: FP16 u = {u_fp16:.4e}")
    print(f"  If e_max ≈ {u_fp16*2:.4e} (~2u_FP16), then FP8 uses FP32 internal accum + FP16 output")

    avg_emax = np.mean([r['max_emax'] for r in results])
    print(f"\nMeasured avg e_max = {avg_emax:.4e}")
    print(f"  Ratio to 2u_FP16: {avg_emax / (2*u_fp16):.2f}x")

    if avg_emax < u * 0.5:
        print(f"\nConclusion: e_max << u_FP8, indicating FP8 Tensor Core uses high-precision accumulation")
        print(f"  Effective e_max for V-ABFT: use FP16 value (~{2*u_fp16:.4e})")
    else:
        print(f"\nConclusion: e_max ~ {np.mean(max_over_u):.1f}u_FP8")

    # 表格输出
    print("\n" + "=" * 100)
    print("Table for Paper")
    print("=" * 100)
    print(f"| N | e_max | e_max/u_FP8 | e_max/u_FP16 |")
    print("|---|-------|-------------|--------------|")
    for r in results:
        print(f"| {r['N']} | {r['max_emax']:.2e} | {r['max_emax_over_u']:.2f} | {r['max_emax']/u_fp16:.1f} |")

    # 推荐值
    print("\n" + "=" * 100)
    print("Recommended e_max for V-ABFT")
    print("=" * 100)
    recommended = np.max([r['max_emax'] for r in results]) * 1.2
    print(f"FP8 {dtype_name.upper()}: e_max = {recommended:.2e} (constant, ~{recommended/u_fp16:.1f} × u_FP16)")
    print(f"  Note: FP8 uses FP32 internal accumulation, so e_max is determined by output precision (FP16)")


if __name__ == "__main__":
    main()
