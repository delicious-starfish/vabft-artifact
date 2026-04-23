"""
验证 A-ABFT 论文的分区编码 (Partitioned Encoding)

论文 Section II 提到:
"The A-ABFT matrix multiplication uses a partitioned encoding scheme,
which encodes the sub-matrices of A and B with a block size of BS × BS."

这意味着:
- 大矩阵被分成 BS×BS 的子矩阵
- 每个子矩阵单独计算校验和
- y 值是针对子矩阵计算的，所以会小很多
"""

import torch
import math
import numpy as np

try:
    from mpmath import mp, mpf, fsum
    mp.dps = 100
    HAS_MPMATH = True
except ImportError:
    HAS_MPMATH = False


def aabft_threshold(n: int, y: float, t: int = 53) -> float:
    """A-ABFT 门限"""
    var_coeff = (n * (n + 1) * (n + 0.5) + 2 * n) / 24
    sigma = math.sqrt(var_coeff) * (2 ** (-t)) * y
    return 3 * sigma


def compute_y_for_block(A_block, B_block, p: int = 2):
    """计算子块的 y 值"""
    n = A_block.shape[0]

    A_abs = torch.abs(A_block)
    B_rowsum = torch.sum(B_block, dim=1)
    B_rowsum_abs = torch.abs(B_rowsum)

    A_topk_vals, A_topk_idx = torch.topk(A_abs, min(p, n), dim=1)
    B_topk_vals, B_topk_idx = torch.topk(B_rowsum_abs, min(p, n))

    y_values = []
    for i in range(n):
        A_idx_set = set(A_topk_idx[i].tolist())
        B_idx_set = set(B_topk_idx.tolist())

        intersection = A_idx_set & B_idx_set
        if intersection:
            y = max(abs(A_block[i, s].item() * B_rowsum[s].item()) for s in intersection)
        else:
            a_max = A_topk_vals[i, 0].item()
            a_min = A_topk_vals[i, -1].item()
            b_max = B_topk_vals[0].item()
            b_min = B_topk_vals[-1].item()
            y = max(a_max * b_min, b_max * a_min)
        y_values.append(y)

    return np.mean(y_values)


def test_partitioned_encoding(n: int, block_sizes: list, num_trials: int = 5):
    """测试分区编码对 y 值的影响"""
    print(f"\n{'='*70}")
    print(f"测试分区编码 (矩阵大小 {n}×{n})")
    print(f"{'='*70}")

    dtype = torch.float64

    for bs in block_sizes:
        if n % bs != 0:
            continue

        y_values = []
        for trial in range(num_trials):
            A = 2 * torch.rand(n, n, dtype=dtype) - 1
            B = 2 * torch.rand(n, n, dtype=dtype) - 1

            # 分块计算 y
            block_y_list = []
            num_blocks = n // bs
            for bi in range(num_blocks):
                for bj in range(num_blocks):
                    # 子矩阵
                    A_block = A[bi*bs:(bi+1)*bs, :]
                    B_block = B[:, bj*bs:(bj+1)*bs]

                    # 对于行校验和，需要 A_block @ sum(B_block, dim=1)
                    # 但分区编码下，是 A_sub @ B_sub 的子块
                    # 这里简化：只看 A_block 的一行与 B 整行的内积

            # 简化：计算整个矩阵的 y，然后按块大小估计
            # 对于分区编码，内积长度变成 bs 而不是 n
            # y 也应该相应减小

            # 论文方法：对每个 bs×bs 的子块单独计算
            for bi in range(num_blocks):
                A_rows = A[bi*bs:(bi+1)*bs, :]
                for bk in range(num_blocks):
                    A_sub = A_rows[:, bk*bs:(bk+1)*bs]  # bs×bs
                    B_sub = B[bk*bs:(bk+1)*bs, :]

                    # B_sub 的行求和 (bs 个元素)
                    B_sub_rowsum = torch.sum(B_sub, dim=1)  # shape: (bs,)

                    # y = max|A_sub[i,k] * B_sub_rowsum[k]|
                    for i in range(bs):
                        for k in range(bs):
                            block_y_list.append(abs(A_sub[i, k].item() * B_sub_rowsum[k].item()))

            y_values.append(np.mean(block_y_list))

        avg_y = np.mean(y_values)

        # 计算对应的门限
        # 分区编码下，内积长度是 bs，不是 n
        threshold = aabft_threshold(bs, avg_y)

        print(f"\n块大小 BS={bs}:")
        print(f"  平均 y 值: {avg_y:.2f}")
        print(f"  内积长度: {bs}")
        print(f"  门限 (BS={bs}): {threshold:.2e}")

        # 对比全矩阵
        if bs == n:
            print(f"  (这是全矩阵情况)")


def test_paper_implied_y(n: int, num_trials: int = 5):
    """测试论文隐含的 y 值"""
    print(f"\n{'='*70}")
    print(f"分析论文隐含的 y 值 (n={n})")
    print(f"{'='*70}")

    # 论文数据
    paper_data = {512: (2.25e-14, 1.68e-11), 1024: (4.53e-14, 4.88e-11),
                  2048: (9.09e-14, 1.46e-10), 4096: (1.81e-13, 4.27e-10)}

    if n not in paper_data:
        print(f"  没有 n={n} 的论文数据")
        return

    paper_err, paper_th = paper_data[n]

    # 反推 y
    var_coeff = (n * (n + 1) * (n + 0.5) + 2 * n) / 24
    implied_y = paper_th / (3 * math.sqrt(var_coeff) * 2**(-53))

    print(f"  论文门限: {paper_th:.2e}")
    print(f"  方差系数: {var_coeff:.2e}")
    print(f"  隐含 y 值: {implied_y:.2f}")

    # 如果使用分区编码 BS=16
    bs = 16
    var_coeff_block = (bs * (bs + 1) * (bs + 0.5) + 2 * bs) / 24

    # 假设 y ≈ sqrt(bs) (因为 B_rowsum 的量级是 sqrt(bs))
    y_block = math.sqrt(bs)
    threshold_block = 3 * math.sqrt(var_coeff_block) * 2**(-53) * y_block

    # 整个矩阵有 (n/bs)^2 个子块，误差累积...
    num_blocks_per_row = n // bs
    # 每行校验和涉及 num_blocks_per_row 个子块的累加
    total_threshold = threshold_block * num_blocks_per_row

    print(f"\n  假设分区编码 BS={bs}:")
    print(f"    子块方差系数: {var_coeff_block:.2e}")
    print(f"    子块 y 估计 (√BS): {y_block:.2f}")
    print(f"    子块门限: {threshold_block:.2e}")
    print(f"    累积门限 (×{num_blocks_per_row}): {total_threshold:.2e}")
    print(f"    与论文比值: {total_threshold/paper_th:.2f}×")


def main():
    print("分析 A-ABFT 论文的分区编码")

    for n in [512, 1024, 2048]:
        test_paper_implied_y(n)

    # 直接测试不同块大小
    print("\n" + "="*70)
    print("测试: 对于 [-1,1] 均匀分布，不同内积长度下的 y 值")
    print("="*70)

    for bs in [16, 32, 64, 128, 256, 512]:
        dtype = torch.float64
        y_samples = []
        for _ in range(100):
            A = 2 * torch.rand(bs, bs, dtype=dtype) - 1
            B = 2 * torch.rand(bs, bs, dtype=dtype) - 1
            B_rowsum = torch.sum(B, dim=1)

            # y = max|A[i,k] * B_rowsum[k]|
            products = torch.abs(A) * torch.abs(B_rowsum).unsqueeze(0)
            y = products.max().item()
            y_samples.append(y)

        avg_y = np.mean(y_samples)
        expected_y = math.sqrt(bs / 3) * 3  # 约 3σ 的 B_rowsum
        print(f"  BS={bs:4d}: avg_y = {avg_y:.2f}, expected ≈ √(BS/3)×3 = {expected_y:.2f}, ratio = {avg_y/expected_y:.2f}")


if __name__ == "__main__":
    main()
