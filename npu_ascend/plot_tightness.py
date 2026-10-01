"""tightness_recalib.json -> 论文 tightness 图 + 表。

图: 每分布一个子图, x=size(log2), y=tightness(log10), 三线 A-ABFT / V-new / V-old。
用法: python3 plot_tightness.py
"""

from __future__ import annotations

import json
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

JSON = "tightness_recalib.json"


def main() -> None:
    d = json.load(open(JSON))
    dists: List[str] = list(d["data"].keys())
    fig, axes = plt.subplots(1, len(dists), figsize=(4.4 * len(dists), 3.8))
    if len(dists) == 1:
        axes = [axes]
    for ax, dist in zip(axes, dists):
        sizes = sorted(int(s) for s in d["data"][dist])
        a = [d["data"][dist][str(n)]["Atight"] for n in sizes]
        vn = [d["data"][dist][str(n)]["Vtight_new"] for n in sizes]
        vo = [d["data"][dist][str(n)]["Vtight_old"] for n in sizes]
        ax.plot(sizes, a, "s-", color="#d62728", lw=1.8, label="A-ABFT")
        ax.plot(sizes, vn, "o-", color="#1f77b4", lw=2.0, label="V-ABFT (recalib, c_σ=5.39)")
        ax.plot(sizes, vo, "o--", color="#7fb0d6", lw=1.5, label="V-ABFT (old, c_σ=2.5)")
        # 标注 new 优势倍数
        for n in sizes:
            adv = d["data"][dist][str(n)]["adv_new"]
            ax.annotate(f"{adv:.0f}×", (n, d["data"][dist][str(n)]["Vtight_new"]),
                        textcoords="offset points", xytext=(0, -12), fontsize=6,
                        color="#1f77b4", ha="center")
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xticks(sizes); ax.set_xticklabels(sizes, fontsize=7)
        ax.set_title(dist, fontsize=10)
        ax.set_xlabel("matrix size (N=K=M)")
        ax.grid(alpha=0.3, which="both")
        ax.axhspan(1, 25, color="#e8f5e9", alpha=0.5, zorder=0)  # 摘要声称 FP32 8-25× 带
    axes[0].set_ylabel("tightness = threshold / actual  (↓ tighter)")
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Threshold tightness after recalibration (FP32, δ=1e-6→c_σ=5.39, 100 trials)\n"
                 "V-new vs A-ABFT advantage annotated; green band = abstract's 8–25× claim",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig("fig_tightness_recalib.png", dpi=150, bbox_inches="tight")
    print("saved fig_tightness_recalib.png")

    # ---- 表 ----
    lines = ["## 重标定紧度 (threshold/actual, 100 trials)\n",
             f"- new: `{d['meta']['new']}` | old: `{d['meta']['old']}`\n"]
    for dist in dists:
        lines.append(f"### {dist}")
        lines.append("| size | actual | A-tight | V-tight(new) | V-tight(old) | adv(new) | adv(old) |")
        lines.append("|---|---|---|---|---|---|---|")
        for n in sorted(int(s) for s in d["data"][dist]):
            r = d["data"][dist][str(n)]
            lines.append(f"| {n} | {r['actual']:.2e} | {r['Atight']:.0f}× | "
                         f"{r['Vtight_new']:.1f}× | {r['Vtight_old']:.1f}× | "
                         f"{r['adv_new']:.0f}× | {r['adv_old']:.0f}× |")
        lines.append("")
    open("tightness_recalib_table.md", "w").write("\n".join(lines) + "\n")
    print("saved tightness_recalib_table.md")


if __name__ == "__main__":
    main()
