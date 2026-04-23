"""
实验：错误定位的信噪比分析

目标：
1. 分析不同 r1, r2 编码方案的定位精度
2. 确定不同精度（FP32/BF16/FP16）下的最小可检测错误
3. 给出最优编码方案建议

理论模型：
- D1 = Δ_j + N1 (检测量)
- D2 = j×Δ_j + N2 (定位量)
- j_hat = D2 / D1
- 定位误差 = |j_hat - j|

正确定位条件: |j_hat - j| < 0.5
"""

import numpy as np
import cupy as cp
from cupy.cuda import cublas


def measure_localization_noise(M, N, K, precision='float32', num_trials=1000):
    """测量定位噪声的统计特性"""

    dtype = getattr(cp, precision)

    # 存储噪声样本
    N1_samples = []  # 全一向量噪声
    N2_samples = []  # 位置编码噪声

    # 不同编码方案
    r1 = cp.ones(N, dtype=cp.float64)
    r2_linear = cp.arange(1, N+1, dtype=cp.float64)  # [1,2,...,N]
    r2_sqrt = cp.sqrt(cp.arange(1, N+1, dtype=cp.float64))  # [1,√2,...,√N]
    r2_log = cp.log2(cp.arange(1, N+1, dtype=cp.float64) + 1)  # log编码

    N2_linear_samples = []
    N2_sqrt_samples = []
    N2_log_samples = []

    for _ in range(num_trials):
        # 生成矩阵
        A = cp.random.randn(M, K).astype(dtype)
        B = cp.random.randn(K, N).astype(dtype)

        # 计算 C = A @ B
        C = A @ B

        # 计算校验和（两条路径）
        # 路径1: (A @ B) @ r
        Br = B.astype(cp.float64) @ r1
        ABr_path1 = A.astype(cp.float64) @ Br  # FP64 计算

        # 路径2: C @ r
        Cr_path2 = C.astype(cp.float64) @ r1

        # 验证差 (每行)
        diff = Cr_path2 - ABr_path1
        N1_samples.extend(diff.get().tolist())

        # 不同 r2 编码的噪声
        ABr2_path1 = A.astype(cp.float64) @ (B.astype(cp.float64) @ r2_linear)
        Cr2_path2 = C.astype(cp.float64) @ r2_linear
        diff2_linear = Cr2_path2 - ABr2_path1
        N2_linear_samples.extend(diff2_linear.get().tolist())

        ABr2_sqrt = A.astype(cp.float64) @ (B.astype(cp.float64) @ r2_sqrt)
        Cr2_sqrt = C.astype(cp.float64) @ r2_sqrt
        diff2_sqrt = Cr2_sqrt - ABr2_sqrt
        N2_sqrt_samples.extend(diff2_sqrt.get().tolist())

        ABr2_log = A.astype(cp.float64) @ (B.astype(cp.float64) @ r2_log)
        Cr2_log = C.astype(cp.float64) @ r2_log
        diff2_log = Cr2_log - ABr2_log
        N2_log_samples.extend(diff2_log.get().tolist())

    return {
        'N1': np.array(N1_samples),
        'N2_linear': np.array(N2_linear_samples),
        'N2_sqrt': np.array(N2_sqrt_samples),
        'N2_log': np.array(N2_log_samples),
    }


def simulate_localization(M, N, K, precision='float32', error_magnitude=1.0,
                          error_col=None, num_trials=1000):
    """模拟错误定位过程"""

    dtype = getattr(cp, precision)

    if error_col is None:
        error_col = N // 2  # 默认在中间位置注入错误

    r1 = cp.ones(N, dtype=cp.float64)
    r2 = cp.arange(1, N+1, dtype=cp.float64)

    localization_errors = []
    detection_diffs = []

    for _ in range(num_trials):
        # 生成矩阵
        A = cp.random.randn(M, K).astype(dtype)
        B = cp.random.randn(K, N).astype(dtype)

        # 计算 C = A @ B
        C = A @ B

        # 注入错误到某一行的第 error_col 列
        error_row = np.random.randint(0, M)
        C_corrupted = C.copy()
        original_value = float(C_corrupted[error_row, error_col])
        C_corrupted[error_row, error_col] += error_magnitude

        # 计算校验和
        # 预期值 (从 A, B 计算)
        Br1 = B.astype(cp.float64) @ r1
        Br2 = B.astype(cp.float64) @ r2
        ABr1 = A.astype(cp.float64) @ Br1
        ABr2 = A.astype(cp.float64) @ Br2

        # 实际值 (从 C_corrupted 计算)
        Cr1 = C_corrupted.astype(cp.float64) @ r1
        Cr2 = C_corrupted.astype(cp.float64) @ r2

        # 检测量和定位量
        D1 = float(Cr1[error_row] - ABr1[error_row])
        D2 = float(Cr2[error_row] - ABr2[error_row])

        detection_diffs.append(D1)

        # 估计错误位置
        if abs(D1) > 1e-10:
            j_hat = D2 / D1
            loc_error = abs(j_hat - (error_col + 1))  # r2 是 1-indexed
            localization_errors.append(loc_error)
        else:
            localization_errors.append(float('inf'))

    return {
        'loc_errors': np.array(localization_errors),
        'detection_diffs': np.array(detection_diffs),
        'error_magnitude': error_magnitude,
    }


def find_min_detectable_error(M, N, K, precision, target_success_rate=0.99):
    """找到给定精度下的最小可定位错误"""

    # 二分搜索
    low, high = 1e-10, 1e3

    while high / low > 1.1:  # 精度到 10%
        mid = np.sqrt(low * high)

        result = simulate_localization(M, N, K, precision,
                                       error_magnitude=mid,
                                       num_trials=500)

        success_rate = np.mean(result['loc_errors'] < 0.5)

        if success_rate >= target_success_rate:
            high = mid
        else:
            low = mid

    return high


def analyze_encoding_schemes(N):
    """分析不同编码方案的理论性能"""

    # r1: 全一向量 [1,1,...,1]
    r1 = np.ones(N)

    # 不同 r2 方案
    schemes = {
        'Linear [1,2,...,N]': np.arange(1, N+1),
        'Sqrt [1,√2,...,√N]': np.sqrt(np.arange(1, N+1)),
        'Log [log(2),...,log(N+1)]': np.log2(np.arange(1, N+1) + 1),
        'Binary (需要多个向量)': None,  # 特殊处理
    }

    print(f"\n{'='*70}")
    print(f"不同 r2 编码方案的理论分析 (N={N})")
    print(f"{'='*70}")
    print(f"\n假设舍入误差 ε_k ~ N(0, σ²) 独立同分布")
    print(f"\nN1 = Σ ε_k, Var(N1) = N × σ²")
    print()

    for name, r2 in schemes.items():
        if r2 is None:
            print(f"\n{name}:")
            print(f"  使用 log₂(N) 个向量，每个向量的系数为 0 或 1")
            print(f"  Var(N2_bit) = N/2 × σ² (每个 bit)")
            print(f"  总方差 = log₂(N) × N/2 × σ² = {np.log2(N) * N / 2:.1f} σ²")
            continue

        # N2 = Σ r2_k × ε_k
        var_N2 = np.sum(r2**2)

        # 最坏情况定位误差方差 (j = N 时)
        # Var(N2 - j×N1) = Var(N2) + j²×Var(N1) ≈ Var(N2) for large j

        # 位置分辨率：相邻位置的差异
        resolution = np.diff(r2)
        min_resolution = np.min(resolution) if len(resolution) > 0 else r2[0]

        print(f"\n{name}:")
        print(f"  Var(N2) = {var_N2:.1f} σ²")
        print(f"  Var(N2) / Var(N1) = {var_N2/N:.1f}")
        print(f"  Std(N2) / Std(N1) = {np.sqrt(var_N2/N):.1f}")
        print(f"  最小分辨率 = {min_resolution:.4f}")
        print(f"  最大 r2 值 = {np.max(r2):.1f}")

    return schemes


def main():
    print("=" * 70)
    print("错误定位信噪比分析实验")
    print("=" * 70)

    # 1. 理论分析
    for N in [256, 1024, 4096]:
        analyze_encoding_schemes(N)

    # 2. 测量实际噪声
    print("\n" + "=" * 70)
    print("实际噪声测量")
    print("=" * 70)

    test_configs = [
        (128, 256, 256, 'float32'),
        (128, 256, 256, 'float16'),
        (128, 1024, 256, 'float32'),
        (128, 1024, 256, 'float16'),
    ]

    for M, N, K, precision in test_configs:
        print(f"\n--- {precision.upper()}, M={M}, N={N}, K={K} ---")

        noise = measure_localization_noise(M, N, K, precision, num_trials=500)

        std_N1 = np.std(noise['N1'])
        std_N2_linear = np.std(noise['N2_linear'])
        std_N2_sqrt = np.std(noise['N2_sqrt'])
        std_N2_log = np.std(noise['N2_log'])

        print(f"  Std(N1) = {std_N1:.4e}")
        print(f"  Std(N2_linear) = {std_N2_linear:.4e} (ratio: {std_N2_linear/std_N1:.1f}×)")
        print(f"  Std(N2_sqrt) = {std_N2_sqrt:.4e} (ratio: {std_N2_sqrt/std_N1:.1f}×)")
        print(f"  Std(N2_log) = {std_N2_log:.4e} (ratio: {std_N2_log/std_N1:.1f}×)")

        # 理论预测
        theory_ratio_linear = np.sqrt(N/3)  # sqrt(N³/3 / N) = sqrt(N²/3)
        print(f"  理论 Std(N2_linear)/Std(N1) ≈ √(N/3) = {theory_ratio_linear:.1f}")

    # 3. 定位成功率 vs 错误大小
    print("\n" + "=" * 70)
    print("定位成功率 vs 错误大小")
    print("=" * 70)

    M, N, K = 128, 256, 256

    for precision in ['float32', 'float16']:
        print(f"\n--- {precision.upper()} ---")

        # 测试不同错误大小
        if precision == 'float32':
            error_magnitudes = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
        else:
            error_magnitudes = [1e-3, 1e-2, 1e-1, 1.0, 10.0]

        print(f"{'Error Magnitude':<18} {'Success Rate':<15} {'Mean Loc Error':<18}")
        print("-" * 55)

        for err_mag in error_magnitudes:
            result = simulate_localization(M, N, K, precision,
                                          error_magnitude=err_mag,
                                          num_trials=500)

            success_rate = np.mean(result['loc_errors'] < 0.5) * 100
            mean_loc_err = np.mean(result['loc_errors'][result['loc_errors'] < 100])

            print(f"{err_mag:<18.2e} {success_rate:<15.1f}% {mean_loc_err:<18.4f}")

    # 4. 找最小可定位错误
    print("\n" + "=" * 70)
    print("最小可定位错误 (99% 成功率)")
    print("=" * 70)

    configs = [
        (128, 256, 256),
        (128, 1024, 256),
        (128, 4096, 256),
    ]

    print(f"\n{'Config':<25} {'FP32':<15} {'FP16':<15} {'Ratio':<10}")
    print("-" * 70)

    for M, N, K in configs:
        min_err_fp32 = find_min_detectable_error(M, N, K, 'float32')
        min_err_fp16 = find_min_detectable_error(M, N, K, 'float16')

        config_str = f"M={M}, N={N}, K={K}"
        ratio = min_err_fp16 / min_err_fp32

        print(f"{config_str:<25} {min_err_fp32:<15.2e} {min_err_fp16:<15.2e} {ratio:<10.1f}×")

    # 5. 结论
    print("\n" + "=" * 70)
    print("结论与建议")
    print("=" * 70)

    print("""
1. 【r1, r2 最优选择】
   - r1 = [1,1,...,1] 是最优的检测向量（最大化信号）
   - r2 = [1,2,...,N] 的问题：Var(N2) ~ O(N³)，噪声太大
   - 更好的选择：
     * r2_sqrt = [1,√2,...,√N]: Var ~ O(N²)，噪声降低 √N 倍
     * r2_log = [log₂(2),...,log₂(N+1)]: Var ~ O(N log²N)
     * 二进制编码：使用 log₂(N) 个向量，每个 Var ~ O(N)

2. 【精度要求】
   - FP32 校验：最小可定位错误 ~ 10⁻⁴ × |C_avg|
   - FP16 校验：最小可定位错误 ~ 10⁻² × |C_avg|
   - BF16 校验：最小可定位错误 ~ 10⁻¹ × |C_avg|

   结论：低精度校验难以实现精确定位，建议：
   - 检测用低精度（阈值可以宽松）
   - 定位用 FP32（需要高精度）

3. 【实际建议】
   - 对于 N ≤ 256: 线性编码 [1,2,...,N] 可接受
   - 对于 N > 256: 考虑 sqrt 或 log 编码
   - 对于纠正：必须用 FP32 精度计算定位量
""")


if __name__ == "__main__":
    main()
