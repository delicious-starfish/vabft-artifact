"""
检查两个可能的问题：
1. 矩阵分布是否正确（均匀分布 [-1, 1]）
2. A-ABFT 的 y 值计算是否与论文一致

论文 Section IV-E 描述了 y 的计算方法：
- 找每行/列的 p 个最大绝对值（论文用 p=2）
- 根据索引是否重叠选择不同的计算方式
"""

import torch
import math
import numpy as np
from datetime import datetime

try:
    from mpmath import mp, mpf, fsum
    mp.dps = 100
    HAS_MPMATH = True
except ImportError:
    HAS_MPMATH = False


def check_distribution(n: int = 1000):
    """检查 torch.rand 生成的分布"""
    print("=" * 60)
    print("1. 检查矩阵分布")
    print("=" * 60)

    # 生成矩阵
    A = 2 * torch.rand(n, n, dtype=torch.float64) - 1

    print(f"矩阵大小: {n}×{n}")
    print(f"最小值: {A.min().item():.6f} (期望: -1)")
    print(f"最大值: {A.max().item():.6f} (期望: 1)")
    print(f"均值: {A.mean().item():.6f} (期望: 0)")
    print(f"标准差: {A.std().item():.6f} (期望: {1/math.sqrt(3):.6f} = 1/√3)")

    # 检查是否均匀分布（分10个区间统计）
    bins = 10
    hist = torch.histc(A.flatten(), bins=bins, min=-1, max=1)
    expected = n * n / bins
    print(f"\n直方图 (期望每区间 {expected:.0f} 个):")
    for i, count in enumerate(hist):
        bar = '█' * int(count / expected * 20)
        print(f"  [{-1 + i*0.2:.1f}, {-1 + (i+1)*0.2:.1f}): {int(count):6d} {bar}")

    return A


def compute_y_paper_method(A, B, p: int = 2):
    """
    按论文 Section IV-E 的方法计算 y

    对于每个 c_{i,j} = sum_k a_{i,k} * b_{k,j}:
    - 找 A 第 i 行的 p 个最大绝对值及索引
    - 找 B 第 j 列的 p 个最大绝对值及索引
    - 根据索引是否重叠计算 y
    """
    n = A.shape[0]

    # 找每行的 p 个最大绝对值
    A_abs = torch.abs(A)
    A_topk_vals, A_topk_idx = torch.topk(A_abs, p, dim=1)

    # 找每列的 p 个最大绝对值
    B_abs = torch.abs(B)
    B_topk_vals, B_topk_idx = torch.topk(B_abs, p, dim=0)

    # 对于校验和计算，我们关心的是 A @ B_rowsum
    # B_rowsum[k] = sum_j B[k,j]
    B_rowsum = torch.sum(B, dim=1)
    B_rowsum_abs = torch.abs(B_rowsum)

    # 找 B_rowsum 的 p 个最大绝对值
    B_rowsum_topk_vals, B_rowsum_topk_idx = torch.topk(B_rowsum_abs, p)

    # 对于 A 的每一行，计算 y
    y_values = []
    for i in range(n):
        A_idx_set = set(A_topk_idx[i].tolist())
        B_idx_set = set(B_rowsum_topk_idx.tolist())

        intersection = A_idx_set & B_idx_set

        if intersection:
            # Case 1: 有重叠，y = max(|a_s * b_s|)
            y = max(abs(A[i, s].item() * B_rowsum[s].item()) for s in intersection)
        else:
            # Case 2 & 3: 无重叠
            # y = max(|a_r|) * min(|b_s|) 或 max(|b_s|) * min(|a_r|)
            a_max = A_topk_vals[i, 0].item()
            a_min = A_topk_vals[i, -1].item()
            b_max = B_rowsum_topk_vals[0].item()
            b_min = B_rowsum_topk_vals[-1].item()

            y = max(a_max * b_min, b_max * a_min)

        y_values.append(y)

    return y_values


def aabft_threshold_with_y(n: int, y: float, t: int = 53) -> float:
    """A-ABFT 门限公式"""
    var_coeff = (n * (n + 1) * (n + 0.5) + 2 * n) / 24
    sigma = math.sqrt(var_coeff) * (2 ** (-t)) * y
    return 3 * sigma


def compute_checksum_diff(A, B, C):
    """计算校验差"""
    if HAS_MPMATH:
        n = A.shape[0]
        B_rowsum_mp = [fsum([mpf(B[k, j].item()) for j in range(n)]) for k in range(n)]
        checksum_mp = [fsum([mpf(A[i, k].item()) * B_rowsum_mp[k] for k in range(n)]) for i in range(n)]
        C_rowsum_mp = [fsum([mpf(C[i, j].item()) for j in range(n)]) for i in range(n)]
        diff = [abs(float(checksum_mp[i]) - float(C_rowsum_mp[i])) for i in range(n)]
    else:
        B_rowsum = torch.sum(B, dim=-1)
        checksum = torch.matmul(A, B_rowsum)
        C_rowsum = torch.sum(C, dim=-1)
        diff = torch.abs(checksum - C_rowsum).tolist()
    return diff


def test_aabft_with_paper_y(n: int, device, num_trials: int = 5):
    """使用论文方法计算 y 的测试"""
    print(f"\n{'='*60}")
    print(f"2. 测试 A-ABFT (使用论文的 y 计算方法, n={n})")
    print(f"{'='*60}")

    dtype = torch.float64

    actual_diffs = []
    thresholds_simple = []  # y = 1
    thresholds_paper = []   # 论文方法计算 y

    for trial in range(num_trials):
        A = 2 * torch.rand(n, n, device=device, dtype=dtype) - 1
        B = 2 * torch.rand(n, n, device=device, dtype=dtype) - 1
        C = torch.matmul(A, B)

        # 真实校验差
        diff = compute_checksum_diff(A.cpu(), B.cpu(), C.cpu())
        actual_diffs.append(np.mean(diff))

        # 方法1: y = 1 (简单假设)
        th_simple = aabft_threshold_with_y(n, y=1.0)
        thresholds_simple.append(th_simple)

        # 方法2: 论文方法计算 y
        y_values = compute_y_paper_method(A.cpu(), B.cpu(), p=2)
        avg_y = np.mean(y_values)
        th_paper = aabft_threshold_with_y(n, y=avg_y)
        thresholds_paper.append(th_paper)

        if trial == 0:
            print(f"\n第一轮详细信息:")
            print(f"  B_rowsum 范围: [{torch.sum(B, dim=1).min():.2f}, {torch.sum(B, dim=1).max():.2f}]")
            print(f"  B_rowsum 期望范围: ~[-{math.sqrt(n)*3:.1f}, {math.sqrt(n)*3:.1f}] (3σ)")
            print(f"  y 值范围: [{min(y_values):.2f}, {max(y_values):.2f}]")
            print(f"  y 平均值: {avg_y:.2f}")

    avg_diff = np.mean(actual_diffs)
    avg_th_simple = np.mean(thresholds_simple)
    avg_th_paper = np.mean(thresholds_paper)

    print(f"\n结果:")
    print(f"  实际误差: {avg_diff:.2e}")
    print(f"  门限 (y=1): {avg_th_simple:.2e}, tightness = {avg_th_simple/avg_diff:.0f}×")
    print(f"  门限 (论文y): {avg_th_paper:.2e}, tightness = {avg_th_paper/avg_diff:.0f}×")

    # 论文参考值
    paper_data = {512: (2.25e-14, 1.68e-11), 1024: (4.53e-14, 4.88e-11), 2048: (9.09e-14, 1.46e-10)}
    if n in paper_data:
        paper_err, paper_th = paper_data[n]
        print(f"\n论文参考:")
        print(f"  论文误差: {paper_err:.2e}, 我们/论文 = {avg_diff/paper_err:.2f}×")
        print(f"  论文门限: {paper_th:.2e}, 我们(论文y)/论文 = {avg_th_paper/paper_th:.2f}×")
        # 反推论文的 y 值
        var_coeff = (n * (n + 1) * (n + 0.5) + 2 * n) / 24
        implied_y = paper_th / (3 * math.sqrt(var_coeff) * 2**(-53))
        print(f"  论文隐含的 y 值: {implied_y:.2f}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--sizes', type=str, default='512,1024')
    args = parser.parse_args()

    device = torch.device(args.device)
    sizes = [int(s) for s in args.sizes.split(',')]

    # 1. 检查分布
    check_distribution(1000)

    # 2. 测试 A-ABFT
    for n in sizes:
        test_aabft_with_paper_y(n, device, num_trials=5)


if __name__ == "__main__":
    main()
