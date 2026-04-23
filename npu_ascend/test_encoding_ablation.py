#!/usr/bin/env python3
"""Encoding Ablation: Linear vs Sinh (复现 Table 3).

复现论文 Table 3 (encoding-comparison):
- 配置: (M, K, N) = (256, 1024, 256)
- 分布: N(0, 1) 标准正态
- 累加精度: FP32
- Trials: 3000
- 错误幅度 |Δ| ∈ {0.05, 0.07, 0.10}
- 5 个错误列位置 (均匀分布)
- 指标: 各位置正确定位率，输出最坏情况 (worst) 和范围 (range)

预期结果 (论文 Table 3):
| |Δ|  | Linear worst | Linear range | Sinh worst | Sinh range |
|------|-------------:|-------------:|-----------:|-----------:|
| 0.05 | 58.1%        | 28.8%        | 78.7%      | 2.2%       |
| 0.07 | 74.3%        | 21.0%        | 90.7%      | 1.7%       |
| 0.10 | 88.2%        | 11.0%        | 98.2%      | 0.3%       |
"""

import torch
import torch_npu
import math
import json
from datetime import datetime

import encoding_utils as eu

device = torch.device('npu:0' if torch_npu.npu.is_available() else 'cpu')
dtype = torch.float32

M, K, N = 128, 1024, 256  # 论文 Table 3: (128, 1024, 256)
TRIALS = 3000
ERROR_MAGS = [0.05, 0.07, 0.10]
# 论文 Sec.4 line 480: "boundary columns (j=0, N-1) are excluded"
TEST_POSITIONS = [1, N // 4, N // 2, 3 * N // 4, N - 2]

print(f"[{datetime.now()}] Encoding Ablation Test")
print(f"Device: {device}, Dtype: {dtype}")
print(f"Config: M={M}, K={K}, N={N}, Trials={TRIALS}")
print(f"Error magnitudes: {ERROR_MAGS}")
print(f"Test positions: {TEST_POSITIONS}")
print("=" * 90)


def run_one_trial(encoding_type: str, error_col: int, error_mag: float) -> int:
    """单次 trial: 注入指定列指定幅度错误，返回是否定位成功 (0/1)."""
    A = torch.randn(M, K, device=device, dtype=dtype)
    B = torch.randn(K, N, device=device, dtype=dtype)
    C = torch.matmul(A, B)

    # 编码 (论文 Sec.4 line 480: zero-centered, equal variance)
    if encoding_type == 'linear':
        w = eu.make_linear_encoding(N, dtype=dtype).to(device)
    else:  # sinh
        w = eu.make_sinh_encoding(N, dtype=dtype).to(device)

    r1 = torch.ones(N, device=device, dtype=dtype)
    Br1 = torch.matmul(B, r1)         # (K,)
    Br2 = torch.matmul(B, w)          # (K,)
    Cr1_pre = torch.matmul(A, Br1)    # (M,) pre-computed checksum
    Cr2_pre = torch.matmul(A, Br2)    # (M,)

    # 注入错误：在 (row=0, col=error_col) 加 error_mag
    C_err = C.clone()
    C_err[0, error_col] += error_mag

    # 计算 D1, D2
    Cr1_post = C_err.sum(dim=1)
    Cr2_post = (C_err * w.unsqueeze(0)).sum(dim=1)
    D1 = Cr1_post - Cr1_pre
    D2 = Cr2_post - Cr2_pre

    # 仅看 row 0 (注入位置)
    if encoding_type == 'linear':
        j_est = eu.localize_error_linear(D1[0:1], D2[0:1], N)
    else:
        j_est = eu.localize_error_sinh(D1[0:1], D2[0:1], w, N)

    return 1 if j_est.item() == error_col else 0


def evaluate(encoding_type: str, error_mag: float):
    """对每个测试位置跑 TRIALS 次，返回每个位置的成功率."""
    pos_accuracies = []
    for pos in TEST_POSITIONS:
        successes = 0
        for _ in range(TRIALS):
            successes += run_one_trial(encoding_type, pos, error_mag)
        acc = 100.0 * successes / TRIALS
        pos_accuracies.append(acc)
    return pos_accuracies


results = {}
for mag in ERROR_MAGS:
    print(f"\n--- |Δ| = {mag} ---")
    for enc in ['linear', 'sinh']:
        accs = evaluate(enc, mag)
        worst = min(accs)
        rng = max(accs) - min(accs)
        results[f"{enc}_{mag}"] = {
            'positions': TEST_POSITIONS,
            'accuracies': accs,
            'worst': worst,
            'range': rng,
        }
        acc_str = ', '.join(f"{a:.1f}" for a in accs)
        print(f"  {enc:<7}: positions=[{acc_str}]%  worst={worst:.1f}%  range={rng:.1f}%")

# 输出 Table 3 格式
print("\n" + "=" * 90)
print("Table 3 (Reproduced)")
print("=" * 90)
print(f"{'|Δ|':<6} {'Lin worst':>10} {'Lin range':>10} {'Sinh worst':>11} {'Sinh range':>11}")
for mag in ERROR_MAGS:
    lin = results[f"linear_{mag}"]
    sinh = results[f"sinh_{mag}"]
    print(f"{mag:<6} {lin['worst']:>9.1f}% {lin['range']:>9.1f}% {sinh['worst']:>10.1f}% {sinh['range']:>10.1f}%")

# 保存 JSON
out = '/home/gyh/V-ABFT_TEST/results_phase2/F1_encoding_ablation.json'
with open(out, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {out}")
