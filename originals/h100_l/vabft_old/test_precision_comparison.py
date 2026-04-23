"""
实验：低精度 GEMM vs 单精度 GEMM 精度对比

设计：
1. 生成 FP16 矩阵 A_fp16, B_fp16
2. 转换为 FP32：A_fp32 = A_fp16.float(), B_fp32 = B_fp16.float()
3. 计算：
   - C_lowprec = A_fp16 × B_fp16（低精度 GEMM）
   - C_fp32 = A_fp32 × B_fp32（单精度 GEMM）
   - C_ref = A_fp64 × B_fp64（FP64 基准）
4. 比较两者相对于 FP64 基准的误差

注：使用 FP16 代替 BF16（cupy 不直接支持 bfloat16），实验逻辑相同
"""

import numpy as np
import cupy as cp
from cupy.cuda import cublas
import ctypes

# cuBLAS 常量
CUBLAS_OP_N = 0
CUDA_R_16F = 2    # FP16
CUDA_R_32F = 0    # FP32
CUBLAS_COMPUTE_32F = 68
CUBLAS_COMPUTE_16F = 64
CUBLAS_GEMM_DEFAULT_TENSOR_OP = 99


class PrecisionExperiment:
    """精度对比实验"""

    def __init__(self):
        self.handle = cublas.create()
        cublas.setMathMode(self.handle, cublas.CUBLAS_TENSOR_OP_MATH)
        # 预分配 alpha, beta
        self.alpha = np.array([1.0], dtype=np.float32)
        self.beta = np.array([0.0], dtype=np.float32)

    def __del__(self):
        cublas.destroy(self.handle)

    def gemm_fp16(self, A_fp16, B_fp16, output_fp32=False):
        """FP16 × FP16 GEMM"""
        M, K = A_fp16.shape
        N = B_fp16.shape[1]

        out_type = CUDA_R_32F if output_fp32 else CUDA_R_16F
        C = cp.empty((M, N), dtype=cp.float32 if output_fp32 else cp.float16)

        cublas.gemmEx(
            self.handle, CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K,
            self.alpha.ctypes.data,
            B_fp16.data.ptr, CUDA_R_16F, N,
            A_fp16.data.ptr, CUDA_R_16F, K,
            self.beta.ctypes.data,
            C.data.ptr, out_type, N,
            CUBLAS_COMPUTE_32F,  # FP32 累加
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        )
        return C

    def gemm_fp32(self, A_fp32, B_fp32):
        """FP32 × FP32 GEMM"""
        M, K = A_fp32.shape
        N = B_fp32.shape[1]

        C = cp.empty((M, N), dtype=cp.float32)

        cublas.gemmEx(
            self.handle, CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K,
            self.alpha.ctypes.data,
            B_fp32.data.ptr, CUDA_R_32F, N,
            A_fp32.data.ptr, CUDA_R_32F, K,
            self.beta.ctypes.data,
            C.data.ptr, CUDA_R_32F, N,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        )
        return C

    def gemm_fp64_reference(self, A, B):
        """FP64 基准"""
        A_f64 = A.astype(cp.float64)
        B_f64 = B.astype(cp.float64)
        return A_f64 @ B_f64


def run_experiment(M, N, K, num_trials=100):
    """运行单组实验"""

    exp = PrecisionExperiment()

    results = {
        # GEMM 结果的误差（相对于 FP64）
        'gemm_err_fp16_out_fp16': [],    # FP16×FP16 → FP16
        'gemm_err_fp16_out_fp32': [],    # FP16×FP16 → FP32
        'gemm_err_fp32': [],              # FP32×FP32 → FP32

        # 校验差（验证差）
        'verif_diff_fp16_out_fp16': [],
        'verif_diff_fp16_out_fp32': [],
        'verif_diff_fp32': [],

        # e_max
        'e_max_fp16_out_fp16': [],
        'e_max_fp16_out_fp32': [],
        'e_max_fp32': [],
    }

    for trial in range(num_trials):
        # 1. 生成 FP16 矩阵（正数，避免抵消）
        A_fp16 = cp.abs(cp.random.randn(M, K).astype(cp.float32)).astype(cp.float16)
        B_fp16 = cp.abs(cp.random.randn(K, N).astype(cp.float32)).astype(cp.float16)

        # 2. 转换为 FP32（相同数值）
        A_fp32 = A_fp16.astype(cp.float32)
        B_fp32 = B_fp16.astype(cp.float32)

        # 3. 计算 GEMM
        C_fp16_out_fp16 = exp.gemm_fp16(A_fp16, B_fp16, output_fp32=False)  # FP16→FP16
        C_fp16_out_fp32 = exp.gemm_fp16(A_fp16, B_fp16, output_fp32=True)   # FP16→FP32
        C_fp32 = exp.gemm_fp32(A_fp32, B_fp32)                              # FP32→FP32

        # 4. FP64 基准
        C_ref = exp.gemm_fp64_reference(A_fp16, B_fp16)

        # 5. 计算 GEMM 误差（相对于 FP64）
        err_fp16_out_fp16 = float(cp.abs(C_fp16_out_fp16.astype(cp.float64) - C_ref).max())
        err_fp16_out_fp32 = float(cp.abs(C_fp16_out_fp32.astype(cp.float64) - C_ref).max())
        err_fp32 = float(cp.abs(C_fp32.astype(cp.float64) - C_ref).max())

        results['gemm_err_fp16_out_fp16'].append(err_fp16_out_fp16)
        results['gemm_err_fp16_out_fp32'].append(err_fp16_out_fp32)
        results['gemm_err_fp32'].append(err_fp32)

        # 6. 计算校验差 |Ce - ABe|
        e_f64 = cp.ones(N, dtype=cp.float64)
        e_f32 = cp.ones(N, dtype=cp.float32)

        # FP64 参考值
        Be_ref = B_fp16.astype(cp.float64) @ e_f64
        ABe_ref = A_fp16.astype(cp.float64) @ Be_ref

        # 各方案的 Ce
        Ce_fp16_fp16 = C_fp16_out_fp16.astype(cp.float64) @ e_f64
        Ce_fp16_fp32 = C_fp16_out_fp32.astype(cp.float64) @ e_f64
        Ce_fp32 = C_fp32.astype(cp.float64) @ e_f64

        # 使用 FP32 精度计算 ABe（模拟实际校验）
        Be_f32 = B_fp32 @ e_f32
        ABe_f32 = (A_fp32 @ Be_f32).astype(cp.float64)

        # 校验差
        verif_fp16_fp16 = float(cp.abs(Ce_fp16_fp16 - ABe_f32).max())
        verif_fp16_fp32 = float(cp.abs(Ce_fp16_fp32 - ABe_f32).max())
        verif_fp32 = float(cp.abs(Ce_fp32 - ABe_f32).max())

        results['verif_diff_fp16_out_fp16'].append(verif_fp16_fp16)
        results['verif_diff_fp16_out_fp32'].append(verif_fp16_fp32)
        results['verif_diff_fp32'].append(verif_fp32)

        # 计算 e_max = |Ce - ABe| / |ABe|
        checksum_mag = float(cp.abs(ABe_ref).max())
        if checksum_mag > 1e-10:
            results['e_max_fp16_out_fp16'].append(verif_fp16_fp16 / checksum_mag)
            results['e_max_fp16_out_fp32'].append(verif_fp16_fp32 / checksum_mag)
            results['e_max_fp32'].append(verif_fp32 / checksum_mag)

    return results


def main():
    print("=" * 95)
    print("实验：低精度 GEMM vs 单精度 GEMM 精度对比")
    print("=" * 95)
    print()
    print("实验设计：")
    print("  1. 生成 FP16 矩阵 A_fp16, B_fp16")
    print("  2. 转换为 FP32：A_fp32 = A_fp16.float(), B_fp32 = B_fp16.float()")
    print("  3. 三种 GEMM：")
    print("     - FP16×FP16 → FP16（低精度输入，低精度输出）")
    print("     - FP16×FP16 → FP32（低精度输入，单精度输出）")
    print("     - FP32×FP32 → FP32（单精度输入，单精度输出）")
    print("  4. 基准：FP64 计算")
    print()
    print("  注：使用 FP16 代替 BF16（cupy 限制），实验逻辑相同")
    print()

    sizes = [
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
    ]

    all_results = []

    for M, N, K in sizes:
        print(f"\n{'='*80}")
        print(f"Size: {M}×{N}×{K}")
        print(f"{'='*80}")

        results = run_experiment(M, N, K, num_trials=100)

        # 统计（取最大值，最保守）
        summary = {
            'size': f"{M}×{N}×{K}",
            'gemm_err_fp16_fp16': max(results['gemm_err_fp16_out_fp16']),
            'gemm_err_fp16_fp32': max(results['gemm_err_fp16_out_fp32']),
            'gemm_err_fp32': max(results['gemm_err_fp32']),
            'verif_fp16_fp16': max(results['verif_diff_fp16_out_fp16']),
            'verif_fp16_fp32': max(results['verif_diff_fp16_out_fp32']),
            'verif_fp32': max(results['verif_diff_fp32']),
            'e_max_fp16_fp16': max(results['e_max_fp16_out_fp16']) if results['e_max_fp16_out_fp16'] else 0,
            'e_max_fp16_fp32': max(results['e_max_fp16_out_fp32']) if results['e_max_fp16_out_fp32'] else 0,
            'e_max_fp32': max(results['e_max_fp32']) if results['e_max_fp32'] else 0,
        }
        all_results.append(summary)

        print(f"\n【GEMM 结果误差】（相对于 FP64 基准）")
        print(f"  FP16×FP16 → FP16: {summary['gemm_err_fp16_fp16']:.4e}")
        print(f"  FP16×FP16 → FP32: {summary['gemm_err_fp16_fp32']:.4e}")
        print(f"  FP32×FP32 → FP32: {summary['gemm_err_fp32']:.4e}")

        ratio1 = summary['gemm_err_fp16_fp16'] / summary['gemm_err_fp16_fp32'] if summary['gemm_err_fp16_fp32'] > 0 else 0
        ratio2 = summary['gemm_err_fp16_fp32'] / summary['gemm_err_fp32'] if summary['gemm_err_fp32'] > 0 else 0
        print(f"\n  FP16输出 vs FP32输出 (低精度输入): {ratio1:.1f}×")
        print(f"  低精度输入 vs 单精度输入 (FP32输出): {ratio2:.2f}×")

        print(f"\n【校验差 |Ce - ABe|】")
        print(f"  FP16×FP16 → FP16: {summary['verif_fp16_fp16']:.4e}")
        print(f"  FP16×FP16 → FP32: {summary['verif_fp16_fp32']:.4e}")
        print(f"  FP32×FP32 → FP32: {summary['verif_fp32']:.4e}")

        ratio3 = summary['verif_fp16_fp16'] / summary['verif_fp16_fp32'] if summary['verif_fp16_fp32'] > 0 else 0
        ratio4 = summary['verif_fp16_fp32'] / summary['verif_fp32'] if summary['verif_fp32'] > 0 else 0
        print(f"\n  FP16输出 vs FP32输出 (低精度输入): {ratio3:.1f}×")
        print(f"  低精度输入 vs 单精度输入 (FP32输出): {ratio4:.2f}×")

        print(f"\n【e_max】")
        print(f"  FP16×FP16 → FP16: {summary['e_max_fp16_fp16']:.4e}")
        print(f"  FP16×FP16 → FP32: {summary['e_max_fp16_fp32']:.4e}")
        print(f"  FP32×FP32 → FP32: {summary['e_max_fp32']:.4e}")

    # 汇总表格
    print("\n" + "=" * 95)
    print("汇总表格")
    print("=" * 95)

    print("\n【GEMM 误差】")
    print(f"{'Size':<16} {'FP16→FP16':<14} {'FP16→FP32':<14} {'FP32→FP32':<14} {'比值(低精度/单精度输入)':<24}")
    print("-" * 85)
    for r in all_results:
        ratio = r['gemm_err_fp16_fp32'] / r['gemm_err_fp32'] if r['gemm_err_fp32'] > 0 else 0
        print(f"{r['size']:<16} {r['gemm_err_fp16_fp16']:<14.4e} {r['gemm_err_fp16_fp32']:<14.4e} {r['gemm_err_fp32']:<14.4e} {ratio:<24.2f}×")

    print("\n【校验差 |Ce - ABe|】")
    print(f"{'Size':<16} {'FP16→FP16':<14} {'FP16→FP32':<14} {'FP32→FP32':<14} {'比值(低精度/单精度输入)':<24}")
    print("-" * 85)
    for r in all_results:
        ratio = r['verif_fp16_fp32'] / r['verif_fp32'] if r['verif_fp32'] > 0 else 0
        print(f"{r['size']:<16} {r['verif_fp16_fp16']:<14.4e} {r['verif_fp16_fp32']:<14.4e} {r['verif_fp32']:<14.4e} {ratio:<24.2f}×")

    print("\n【e_max】")
    print(f"{'Size':<16} {'FP16→FP16':<14} {'FP16→FP32':<14} {'FP32→FP32':<14} {'比值(低精度/单精度输入)':<24}")
    print("-" * 85)
    for r in all_results:
        ratio = r['e_max_fp16_fp32'] / r['e_max_fp32'] if r['e_max_fp32'] > 0 else 0
        print(f"{r['size']:<16} {r['e_max_fp16_fp16']:<14.4e} {r['e_max_fp16_fp32']:<14.4e} {r['e_max_fp32']:<14.4e} {ratio:<24.2f}×")

    # 结论
    print("\n" + "=" * 95)
    print("结论")
    print("=" * 95)

    # 计算平均比值
    avg_ratio_verif = np.mean([r['verif_fp16_fp32'] / r['verif_fp32'] for r in all_results if r['verif_fp32'] > 0])
    avg_ratio_emax = np.mean([r['e_max_fp16_fp32'] / r['e_max_fp32'] for r in all_results if r['e_max_fp32'] > 0])

    print(f"""
核心发现：
  - 校验差比值 (FP16输入+FP32输出 / FP32输入+FP32输出): {avg_ratio_verif:.2f}×
  - e_max 比值: {avg_ratio_emax:.2f}×

结论：""")

    if avg_ratio_verif < 2.0:
        print(f"""
  ✓ 两者基本等价（比值 < 2×）
  → 低精度 GEMM（输出 FP32）可以使用 FP32 的 e_max
  → 门限可以从 ~10⁻³ (FP16输出) 收紧到 ~10⁻⁶ (FP32)
  → 检测精度提升约 1000 倍！
""")
    elif avg_ratio_verif < 10.0:
        print(f"""
  △ 有一定差异（比值 {avg_ratio_verif:.1f}×）
  → 低精度输入+FP32输出 的 e_max 约为 FP32 的 {avg_ratio_verif:.1f} 倍
  → 仍然比 FP16 输出（~10⁻³）好很多
  → 门限仍可收紧约 {1000/avg_ratio_verif:.0f} 倍
""")
    else:
        print(f"""
  ✗ 差异较大（比值 {avg_ratio_verif:.1f}×）
  → 需要为低精度输入单独标定 e_max
""")


if __name__ == "__main__":
    main()
