"""把 recalib_detection.json 画成论文格式：检出率热图(V-ABFT vs A-ABFT) + DR/FPR 表。

生成:
  fig_recalib_heatmap.png  4 分布 × 逐 bit 的 DR 热图，行=[K1024,K4096]，列=[V-ABFT(new), A-ABFT]
  recalib_table.md         全谱 DR(new/old/aabft) + FPR 表
用法: python3 plot_recalib.py
"""

from __future__ import annotations

import json
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

JSON = "recalib_detection.json"
BITS: List[str] = [str(b) for b in range(31, 15, -1)]  # 31→16


def bit_label(b: int) -> str:
    """fp32 位类型标注。"""
    if b == 31:
        return "S"
    if b >= 23:
        return f"E{b}"
    return f"M{b}"


def heatmap(ax, mat: np.ndarray, dists: List[str], title: str):
    """画一张 DR 热图。mat: (n_dist, n_bit)。负值(=无有效样本)显示为灰色 n/a。"""
    masked = np.where(mat < 0, np.nan, mat)
    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad("#cccccc")
    im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=0, vmax=100)
    ax.set_xticks(range(len(BITS)))
    ax.set_xticklabels([bit_label(int(b)) for b in BITS], fontsize=6, rotation=90)
    ax.set_yticks(range(len(dists)))
    ax.set_yticklabels(dists, fontsize=8)
    ax.set_title(title, fontsize=10)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if v < 0:
                ax.text(j, i, "–", ha="center", va="center", fontsize=6, color="#666666")
            else:
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=5,
                        color="black" if 25 < v < 90 else "white")
    return im


def main() -> None:
    d = json.load(open(JSON))
    configs = list(d["DR"].keys())
    dists = list(d["DR"][configs[0]].keys())

    fig, axes = plt.subplots(len(configs), 2, figsize=(11, 6), constrained_layout=True)
    for r, cfg in enumerate(configs):
        mat_new = np.array([[d["DR"][cfg][dist]["new"][b] for b in BITS] for dist in dists])
        mat_a = np.array([[d["DR"][cfg][dist]["aabft"][b] for b in BITS] for dist in dists])
        heatmap(axes[r][0], mat_new, dists, f"V-ABFT (recalib)  {cfg}")
        im = heatmap(axes[r][1], mat_a, dists, f"A-ABFT  {cfg}")
    fig.colorbar(im, ax=axes, shrink=0.6, label="Detection rate (%)")
    fig.suptitle("Detection rate: recalibrated V-ABFT (e_max=1.377e-6, c_σ=5.39) vs A-ABFT\n"
                 "FP32, single bit-flip, bits 31→16 (S/E/M), 200 trials/bit", fontsize=11)
    fig.savefig("fig_recalib_heatmap.png", dpi=150, bbox_inches="tight")
    print("saved fig_recalib_heatmap.png")

    # ---- 表：new/old DR 对比 + FPR ----
    lines: List[str] = []
    lines.append("## 重标定门限 检出率(DR) 与误检率(FPR)\n")
    lines.append(f"- new: `{d['meta']['new']}`  |  old: `{d['meta']['old']}`")
    lines.append(f"- DR trials/bit = {d['meta']['dr_trials']}, FPR trials = {d['meta']['fpr_trials']} "
                 f"(~{d['meta']['fpr_trials']*128//1000}k rows/分布)\n")

    # FPR 表
    lines.append("### FPR（clean，越低越安全）")
    lines.append("| config | dist | new | old | aabft |")
    lines.append("|---|---|---|---|---|")
    for cfg in configs:
        for dist in dists:
            f = d["FPR"][cfg][dist]
            lines.append(f"| {cfg} | {dist} | {f['new']:.2e} | {f['old']:.2e} | {f['aabft']:.2e} |")

    # DR 分化区表（bit 16-20，new vs old vs aabft）
    lines.append("\n### DR 分化区（bit 16-20，%）：new / old / aabft")
    lines.append("| config | dist | b16 | b17 | b18 | b19 | b20 |")
    lines.append("|---|---|---|---|---|---|---|")
    for cfg in configs:
        for dist in dists:
            cells = []
            for b in ["16", "17", "18", "19", "20"]:
                dr = d["DR"][cfg][dist]
                cells.append(f"{dr['new'][b]:.0f}/{dr['old'][b]:.0f}/{dr['aabft'][b]:.0f}")
            lines.append(f"| {cfg} | {dist} | " + " | ".join(cells) + " |")

    # new vs old 最大偏差
    max_dev = 0.0
    for cfg in configs:
        for dist in dists:
            for b in BITS:
                dr = d["DR"][cfg][dist]
                max_dev = max(max_dev, abs(dr["new"][b] - dr["old"][b]))
    lines.append(f"\n**new vs old 全谱最大 DR 偏差 = {max_dev:.1f}%**（≈重合，说明去余量+严格 c_σ 不损失检出）")

    open("recalib_table.md", "w").write("\n".join(lines) + "\n")
    print("saved recalib_table.md")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
