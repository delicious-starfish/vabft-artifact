"""重标定后的门限评测（论文格式）：多分布 × 逐 bit 注入的检出率(DR) + clean 误检率(FPR)。

对比三套门限口径（隔离"e_max 去余量 + c_σ 由 Azuma–Hoeffding 定"的效果）：
  new   : V-ABFT 重标定  e_max=1.377e-6(去×1.45 余量), c_σ=5.39 (=sqrt(2 ln(2/δ)), δ=1e-6)
  old   : V-ABFT 原口径  e_max=2e-6,               c_σ=2.5  (Chebyshev 84%)
  aabft : A-ABFT worst-case 门限（utils.aabft_corrected）

检出率(DR): 单 bit-flip 注入到随机 (i,j)，看错误行 i 是否被判超阈。逐 bit(31→16) × 4 分布。
误检率(FPR): clean(无注入) 大样本下 |D1[i]|>T_i 的行比例，逐分布，反映各口径的安全裕度。

用法(远程 NPU):
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  ASCEND_RT_VISIBLE_DEVICES=<die> <PY> recalib_detection.py --out /path/recalib.json
"""

from __future__ import annotations

import argparse
import json
from typing import Callable, Dict, List, Tuple

import torch
import torch_npu  # noqa: F401  # 注册 npu 后端

import utils

DEV = "npu:0"
DTYPE = torch.float32
# fp32 输出的 bit 位: 31=符号, 30-23=指数, 22-16=高尾数（论文 heatmap 的注入位置维度）
BITS: List[int] = list(range(31, 15, -1))

# 四种输入分布（对齐论文 fig:e2e-heatmap 的分布集合）
DISTS: List[Dict] = [
    {"name": "N(1e-6,1)", "func": utils.generate_matrice, "kw": {"std": 1, "mean": 1e-6}},
    {"name": "N(1,1)", "func": utils.generate_matrice, "kw": {"std": 1, "mean": 1}},
    {"name": "U(-1,1)", "func": utils.generate_matrice_uniform, "kw": {"lower": -1, "upper": 1}},
    {"name": "TruncN", "func": utils.generate_matrice_clamp, "kw": {"std": 1, "mean": 0}},
]

CONFIGS: Dict[str, Tuple[int, int, int]] = {
    "128x1024x256": (128, 1024, 256),
    "128x4096x256": (128, 4096, 256),
}


def thresholds(a: torch.Tensor, b: torch.Tensor,
               c: torch.Tensor) -> Dict[str, torch.Tensor]:
    """三套逐行门限。a:(M,K), b:(K,N), c:(M,N) -> {name: (M,)}。"""
    new = utils.my_bound_improve_robust(a, b, dtype=DTYPE)                       # 默认新口径
    old = utils.my_bound_improve_robust(a, b, dtype=DTYPE, e_max=2e-6, c_sigma=2.5)
    _, aabft = utils.aabft_corrected(a, b, c)
    return {"new": new, "old": old, "aabft": aabft}


def d1_vector(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """校验差 D1 = A·(Σ_col B) - Σ_col C。(M,)。"""
    return torch.matmul(a, torch.sum(b, dim=1)) - torch.sum(c, dim=1)


def make_matrices(dist: Dict, m: int, k: int, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """按分布生成 A(m,k), B(k,n)。"""
    a = dist["func"](m, k, device=DEV, dtype=DTYPE, **dist["kw"])
    b = dist["func"](k, n, device=DEV, dtype=DTYPE, **dist["kw"])
    return a, b


def measure_dr(dist: Dict, m: int, k: int, n: int, trials: int) -> Dict[str, Dict[int, float]]:
    """逐 bit 检出率：注入单 bit-flip，统计错误行被判超阈的比例(%)。"""
    out: Dict[str, Dict[int, float]] = {s: {} for s in ("new", "old", "aabft")}
    for bit in BITS:
        det = {"new": 0, "old": 0, "aabft": 0}
        valid = 0
        attempts = 0
        while valid < trials and attempts < trials * 5:
            attempts += 1
            a, b = make_matrices(dist, m, k, n)
            c = torch.matmul(a, b)
            i = int(torch.randint(0, m, (1,)).item())
            j = int(torch.randint(0, n, (1,)).item())
            flipped, ok = utils.flip_infuse(c[i, j].cpu(), bit)
            if not ok or not torch.isfinite(flipped):
                continue
            c[i, j] = flipped.to(DEV)
            valid += 1
            thr = thresholds(a, b, c)
            d1a = torch.abs(d1_vector(a, b, c)[i])
            for s in det:
                if bool(d1a > thr[s][i]):
                    det[s] += 1
        for s in out:
            out[s][bit] = round(det[s] / valid * 100, 2) if valid else -1.0
    return out


def measure_fpr(dist: Dict, m: int, k: int, n: int, trials: int) -> Dict[str, float]:
    """clean 误检率：无注入下 |D1[i]|>T_i 的行比例。返回各口径 FPR。"""
    fp = {"new": 0, "old": 0, "aabft": 0}
    rows = 0
    for _ in range(trials):
        a, b = make_matrices(dist, m, k, n)
        c = torch.matmul(a, b)
        d1a = torch.abs(d1_vector(a, b, c))
        thr = thresholds(a, b, c)
        rows += m
        for s in fp:
            fp[s] += int(torch.sum(d1a > thr[s]).item())
    return {s: fp[s] / rows if rows else 0.0 for s in fp}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="recalib_detection.json")
    ap.add_argument("--dr_trials", type=int, default=200, help="每 bit 检出率试验数")
    ap.add_argument("--fpr_trials", type=int, default=2000, help="每分布 FPR 试验数(大样本)")
    args = ap.parse_args()

    result: Dict = {"meta": {"dtype": "fp32", "bits": BITS,
                             "new": "e_max=1.377e-6,c_sigma=5.39",
                             "old": "e_max=2e-6,c_sigma=2.5",
                             "dr_trials": args.dr_trials, "fpr_trials": args.fpr_trials},
                    "DR": {}, "FPR": {}}

    for cname, (m, k, n) in CONFIGS.items():
        print(f"\n===== {cname} (M={m},K={k},N={n}) =====", flush=True)
        result["DR"][cname] = {}
        result["FPR"][cname] = {}
        for dist in DISTS:
            dn = dist["name"]
            dr = measure_dr(dist, m, k, n, args.dr_trials)
            result["DR"][cname][dn] = dr
            fpr = measure_fpr(dist, m, k, n, args.fpr_trials)
            result["FPR"][cname][dn] = fpr
            # 打印: bit0 附近(16) 检出 + FPR，快速看重标定效果
            print(f"  [{dn:10s}] DR bit16 new/old/aabft = "
                  f"{dr['new'][16]:.1f}/{dr['old'][16]:.1f}/{dr['aabft'][16]:.1f}%  "
                  f"| FPR new/old/aabft = {fpr['new']:.2e}/{fpr['old']:.2e}/{fpr['aabft']:.2e}",
                  flush=True)
            with open(args.out, "w") as f:      # 增量落盘，防中断丢数据
                json.dump(result, f, indent=2)

    print(f"\nsaved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
