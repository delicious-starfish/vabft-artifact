"""
A-ABFT vs V-ABFT 对比测试 - 支持 FP32/FP64 on CPU/GPU

修正A-ABFT中的问题:
原来: c_ele_round_error_accum = c_max * u * sqrt(N)
修正: c_ele_round_error_accum = c_max * u * sqrt(N*(N+1)*(2N+1)/6)  (约 N^1.5)

原因: C行求和可看作 C_i · 1 的内积，累加误差应按A-ABFT内积误差框架处理
"""

import torch
import math
import numpy as np
from datetime import datetime
import argparse

# 检测可用设备
def get_device(prefer_gpu=True):
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


# ============================================================================
# A-ABFT 实现 (原版 + 修正版)
# ============================================================================

def aabft_original(a, b, c, use_high_precision_check=True):
    """
    原版A-ABFT门限计算 (有bug的版本)

    参数:
        a: (M, K) 输入矩阵A
        b: (K, N) 输入矩阵B
        c: (M, N) 输出矩阵C = A @ B
        use_high_precision_check: 是否用高精度计算校验和

    返回:
        checksum: (M,) 校验值 A @ sum(B, dim=1)
        threshold: (M,) 门限向量
    """
    dtype = a.dtype
    device = a.device

    # 精度参数
    if dtype == torch.float64:
        t = 53  # FP64 尾数位
        u_low = 2**(-53)
    elif dtype == torch.float32:
        t = 23  # FP32 尾数位
        u_low = 2**(-23)
    elif dtype == torch.bfloat16:
        t = 7
        u_low = 2**(-8)
    elif dtype == torch.float16:
        t = 10
        u_low = 2**(-11)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    u_high = 2**(-t) if use_high_precision_check else u_low

    M, K = a.shape
    K2, N = b.shape
    assert K == K2

    # 计算校验和
    if use_high_precision_check:
        b_rowsum = torch.sum(b.to(torch.float64), dim=-1, keepdim=True)
        checksum = torch.matmul(a.to(torch.float64), b_rowsum).squeeze(-1)
    else:
        b_rowsum = torch.sum(b, dim=-1, keepdim=True)
        checksum = torch.matmul(a, b_rowsum).squeeze(-1)

    c_max, _ = torch.max(torch.abs(c), dim=-1)
    b_max, _ = torch.max(torch.abs(b), dim=-1, keepdim=True)
    a_max, _ = torch.max(torch.abs(a), dim=-1)

    # E1: C行求和的累加误差 (高精度)
    # sqrt(sum(i^2)/8) * max|C| * u_high
    E1 = math.sqrt(N*(N+1)*(2*N+1)/48) * c_max * u_high

    # E2: C元素量化误差累积 (原版bug: 用sqrt(N))
    E2 = c_max * u_low * math.sqrt(N)  # <-- 这里是问题所在

    # E3: B行求和的累加误差传播到 A @ B_rowsum
    delta_b = math.sqrt(N*(N+1)*(2*N+1)/48) * b_max * u_high
    E3 = torch.matmul(torch.abs(a.to(torch.float64)), delta_b).squeeze(-1)

    # E4: A @ B_rowsum 的累加误差
    E4 = math.sqrt((K*(K+1)*(K+0.5)+2*K)/24) * a_max.to(torch.float64) * torch.max(b_max) * u_high

    threshold = (E1.to(torch.float64) + E2 + E3 + E4).to(dtype)

    return checksum.to(dtype), threshold


def aabft_corrected(a, b, c, use_high_precision_check=True):
    """
    修正版A-ABFT门限计算

    修正: C行求和可看作内积 C_i · 1，应用内积误差框架
    c_ele_round_error_accum = c_max * u * sqrt(N*(N+1)*(2N+1)/6)
    """
    dtype = a.dtype
    device = a.device

    # 精度参数
    if dtype == torch.float64:
        t = 53
        u_low = 2**(-53)
    elif dtype == torch.float32:
        t = 23
        u_low = 2**(-23)
    elif dtype == torch.bfloat16:
        t = 7
        u_low = 2**(-8)
    elif dtype == torch.float16:
        t = 10
        u_low = 2**(-11)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    u_high = 2**(-t) if use_high_precision_check else u_low

    M, K = a.shape
    K2, N = b.shape
    assert K == K2

    # 计算校验和
    if use_high_precision_check:
        b_rowsum = torch.sum(b.to(torch.float64), dim=-1, keepdim=True)
        checksum = torch.matmul(a.to(torch.float64), b_rowsum).squeeze(-1)
    else:
        b_rowsum = torch.sum(b, dim=-1, keepdim=True)
        checksum = torch.matmul(a, b_rowsum).squeeze(-1)

    c_max, _ = torch.max(torch.abs(c), dim=-1)
    b_max, _ = torch.max(torch.abs(b), dim=-1, keepdim=True)
    a_max, _ = torch.max(torch.abs(a), dim=-1)

    # E1: C行求和的累加误差 (高精度)
    E1 = math.sqrt(N*(N+1)*(2*N+1)/48) * c_max * u_high

    # E2: C元素量化误差累积 (修正版: 用 N^1.5)
    # C行求和 = C_i · 1 的内积，按内积误差框架
    E2 = c_max * u_low * math.sqrt(N*(N+1)*(2*N+1)/6)  # <-- 修正后

    # E3: B行求和的累加误差传播
    delta_b = math.sqrt(N*(N+1)*(2*N+1)/48) * b_max * u_high
    E3 = torch.matmul(torch.abs(a.to(torch.float64)), delta_b).squeeze(-1)

    # E4: A @ B_rowsum 的累加误差
    E4 = math.sqrt((K*(K+1)*(K+0.5)+2*K)/24) * a_max.to(torch.float64) * torch.max(b_max) * u_high

    threshold = (E1.to(torch.float64) + E2 + E3 + E4).to(dtype)

    return checksum.to(dtype), threshold


# ============================================================================
# V-ABFT 实现 (支持所有精度)
# ============================================================================

def vabft(a, b, e_max=None, c_sigma=2.5):
    """
    V-ABFT 门限计算

    参数:
        a: (M, K) 输入矩阵A
        b: (K, N) 输入矩阵B
        e_max: 误差系数，None则根据精度自动选择
        c_sigma: 置信系数

    返回:
        checksum: (M,) 校验值
        threshold: (M,) 门限向量
    """
    dtype = a.dtype
    device = a.device

    # 默认e_max值
    if e_max is None:
        e_max_dict = {
            torch.float64: 3e-15,  # FP64: ~2u
            torch.float32: 3e-7,   # FP32: 随K增长，取典型值
            torch.bfloat16: 8e-3,
            torch.float16: 1e-3,
        }
        e_max = e_max_dict.get(dtype, 1e-6)

    M, K = a.shape
    K2, N = b.shape
    assert K == K2

    # 计算校验和 (用高精度)
    b_rowsum = torch.sum(b.to(torch.float64), dim=-1, keepdim=True)
    checksum = torch.matmul(a.to(torch.float64), b_rowsum).squeeze(-1)

    # 统计量计算
    mu_A = torch.mean(a.to(torch.float64), dim=-1)  # (M,)
    mu_B = torch.mean(b.to(torch.float64), dim=-1)  # (K,)

    # 方差上界估计
    a_max = torch.max(a.to(torch.float64), dim=-1).values
    a_min = torch.min(a.to(torch.float64), dim=-1).values
    b_max = torch.max(b.to(torch.float64), dim=-1).values
    b_min = torch.min(b.to(torch.float64), dim=-1).values

    sigma_A2 = torch.clamp((a_max - mu_A) * (mu_A - a_min), min=1e-20)
    sigma_B2 = torch.clamp((b_max - mu_B) * (mu_B - b_min), min=1e-20)
    sigma_A = torch.sqrt(sigma_A2)
    sigma_B = torch.sqrt(sigma_B2)

    # 门限公式
    mu_A_abs = torch.abs(mu_A)
    mu_B_abs = torch.abs(mu_B)

    sum_mu_B = torch.sum(mu_B_abs)
    sum_mu_B2 = torch.sum(mu_B_abs ** 2)
    sum_sigma_B2 = torch.sum(sigma_B2)

    sqrt_N = math.sqrt(N)

    # 三项
    T_det = N * mu_A_abs * sum_mu_B
    T_var23 = c_sigma * torch.sqrt(N * mu_A_abs**2 * sum_sigma_B2 + N**2 * sigma_A2 * sum_mu_B2)
    T_var4 = c_sigma * sqrt_N * sigma_A * torch.sqrt(sum_sigma_B2)

    threshold = e_max * (T_det + T_var23 + T_var4)

    return checksum.to(dtype), threshold.to(dtype)


# ============================================================================
# 测试函数
# ============================================================================

def test_false_positive_rate(M, K, N, dtype, device, algorithm, num_trials=10000, **kwargs):
    """测试误检率"""
    false_positives = 0

    for _ in range(num_trials):
        # 生成正数矩阵 (A-ABFT原文设置)
        A = torch.abs(torch.randn(M, K, device=device, dtype=dtype) + 1.0)
        B = torch.abs(torch.randn(K, N, device=device, dtype=dtype) + 1.0)
        C = torch.matmul(A, B)

        # 计算校验
        if algorithm == 'aabft_original':
            checksum, threshold = aabft_original(A, B, C, **kwargs)
        elif algorithm == 'aabft_corrected':
            checksum, threshold = aabft_corrected(A, B, C, **kwargs)
        elif algorithm == 'vabft':
            checksum, threshold = vabft(A, B, **kwargs)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        # C的行和
        C_rowsum = torch.sum(C, dim=-1)

        # 检验
        diff = torch.abs(checksum.to(torch.float64) - C_rowsum.to(torch.float64))
        if (diff > threshold.to(torch.float64)).any():
            false_positives += 1

    return false_positives / num_trials


def test_detection_rate(M, K, N, dtype, device, algorithm, bit_position, num_trials=1000, **kwargs):
    """测试检出率 (注入单bit翻转)"""
    detected = 0

    for _ in range(num_trials):
        A = torch.abs(torch.randn(M, K, device=device, dtype=dtype) + 1.0)
        B = torch.abs(torch.randn(K, N, device=device, dtype=dtype) + 1.0)
        C = torch.matmul(A, B)

        # 注入错误
        row_idx = torch.randint(0, M, (1,)).item()
        col_idx = torch.randint(0, N, (1,)).item()

        C_corrupted = C.clone()
        val = C_corrupted[row_idx, col_idx]

        # bit flip
        if dtype == torch.float64:
            int_val = val.view(torch.int64)
            mask = 1 << bit_position
            flipped = int_val ^ mask
            C_corrupted[row_idx, col_idx] = flipped.view(torch.float64)
        elif dtype == torch.float32:
            int_val = val.view(torch.int32)
            mask = 1 << bit_position
            flipped = int_val ^ mask
            C_corrupted[row_idx, col_idx] = flipped.view(torch.float32)

        # 计算校验
        if algorithm == 'aabft_original':
            checksum, threshold = aabft_original(A, B, C_corrupted, **kwargs)
        elif algorithm == 'aabft_corrected':
            checksum, threshold = aabft_corrected(A, B, C_corrupted, **kwargs)
        elif algorithm == 'vabft':
            checksum, threshold = vabft(A, B, **kwargs)

        C_rowsum = torch.sum(C_corrupted, dim=-1)
        diff = torch.abs(checksum.to(torch.float64) - C_rowsum.to(torch.float64))

        if (diff > threshold.to(torch.float64)).any():
            detected += 1

    return detected / num_trials


def test_bound_tightness(M, K, N, dtype, device, algorithm, num_trials=10000, **kwargs):
    """测试门限紧致度: 返回 actual_error / threshold 的统计"""
    ratios = []

    for _ in range(num_trials):
        A = torch.abs(torch.randn(M, K, device=device, dtype=dtype) + 1.0)
        B = torch.abs(torch.randn(K, N, device=device, dtype=dtype) + 1.0)
        C = torch.matmul(A, B)

        if algorithm == 'aabft_original':
            checksum, threshold = aabft_original(A, B, C, **kwargs)
        elif algorithm == 'aabft_corrected':
            checksum, threshold = aabft_corrected(A, B, C, **kwargs)
        elif algorithm == 'vabft':
            checksum, threshold = vabft(A, B, **kwargs)

        C_rowsum = torch.sum(C, dim=-1)
        actual_error = torch.abs(checksum.to(torch.float64) - C_rowsum.to(torch.float64))

        ratio = (threshold.to(torch.float64) / (actual_error + 1e-30)).mean().item()
        ratios.append(ratio)

    return np.mean(ratios), np.std(ratios), np.min(ratios)


# ============================================================================
# 主测试
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='auto', choices=['cpu', 'cuda', 'auto'])
    parser.add_argument('--dtype', type=str, default='float64', choices=['float32', 'float64'])
    parser.add_argument('--trials', type=int, default=10000)
    args = parser.parse_args()

    # 设备选择
    if args.device == 'auto':
        device = get_device(prefer_gpu=True)
    else:
        device = torch.device(args.device)

    dtype = torch.float64 if args.dtype == 'float64' else torch.float32

    print("=" * 80)
    print(f"A-ABFT vs V-ABFT Comparison Test")
    print(f"Device: {device}, Dtype: {dtype}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # 测试不同规模
    sizes = [
        (32, 64, 32),
        (32, 128, 32),
        (32, 256, 32),
        (32, 512, 32),
        (32, 1024, 32),
        (128, 1024, 256),
    ]

    algorithms = ['aabft_original', 'aabft_corrected', 'vabft']

    # 1. 误检率测试
    print("\n" + "=" * 70)
    print("1. FALSE POSITIVE RATE TEST")
    print("=" * 70)

    for M, K, N in sizes:
        print(f"\nSize: ({M}, {K}, {N})")
        print("-" * 50)
        for algo in algorithms:
            fpr = test_false_positive_rate(M, K, N, dtype, device, algo, num_trials=args.trials)
            print(f"  {algo:20s}: FPR = {fpr*100:.4f}%")

    # 2. 门限紧致度测试
    print("\n" + "=" * 70)
    print("2. BOUND TIGHTNESS TEST (threshold / actual_error)")
    print("=" * 70)

    for M, K, N in sizes:
        print(f"\nSize: ({M}, {K}, {N})")
        print("-" * 50)
        for algo in algorithms:
            mean_ratio, std_ratio, min_ratio = test_bound_tightness(
                M, K, N, dtype, device, algo, num_trials=args.trials
            )
            print(f"  {algo:20s}: {mean_ratio:.2f}x ± {std_ratio:.2f} (min: {min_ratio:.2f}x)")

    # 3. 检出率测试 (FP64指数位: 52-62)
    if dtype == torch.float64:
        print("\n" + "=" * 70)
        print("3. DETECTION RATE TEST (FP64 exponent bits 52-62)")
        print("=" * 70)

        M, K, N = 128, 1024, 256
        bit_positions = [52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62]

        print(f"\nSize: ({M}, {K}, {N})")
        print("-" * 60)
        print(f"{'Bit':<6}", end='')
        for algo in algorithms:
            print(f"{algo:>20s}", end='')
        print()

        for bit in bit_positions:
            print(f"{bit:<6}", end='')
            for algo in algorithms:
                dr = test_detection_rate(M, K, N, dtype, device, algo, bit, num_trials=1000)
                print(f"{dr*100:>19.2f}%", end='')
            print()

    # 4. e_max 验证 (不同规模)
    print("\n" + "=" * 70)
    print("4. e_max SCALING VERIFICATION")
    print("=" * 70)

    u = 2**(-53) if dtype == torch.float64 else 2**(-23)
    print(f"Unit roundoff u = {u:.2e}")

    for M, K, N in sizes:
        max_rel_error = 0
        for _ in range(args.trials):
            A = torch.abs(torch.randn(M, K, device=device, dtype=dtype) + 1.0)
            B = torch.abs(torch.randn(K, N, device=device, dtype=dtype) + 1.0)
            C = torch.matmul(A, B)

            b_rowsum = torch.sum(B, dim=-1, keepdim=True)
            checksum = torch.matmul(A, b_rowsum).squeeze(-1)
            c_rowsum = torch.sum(C, dim=-1)

            rel_error = (torch.abs(checksum - c_rowsum) / torch.abs(c_rowsum)).max().item()
            max_rel_error = max(max_rel_error, rel_error)

        print(f"Size ({M:4d}, {K:4d}, {N:4d}): e_max = {max_rel_error:.6e}, e_max/u = {max_rel_error/u:.2f}")


if __name__ == "__main__":
    main()
