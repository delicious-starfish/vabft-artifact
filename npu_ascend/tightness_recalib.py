"""重标定门限的紧度(tightness)测量：threshold / 实际校验差，对比论文声称的 8-300×。

tightness = 门限 / 实际校验差(|D1|)，越接近 1 越紧；优势倍数 = A-Tight / V-Tight。
重标定后 U(-1,1) 等零均值分布 var 主导，V-new 门限较 V-old ×1.48（更松），本脚本量化其代价。

对每个方阵 size(n,n,n) × 分布，clean(无注入) T trials：
  actual = max_行 |D1|（最坏实际校验差，门限必须覆盖它以保 FPR=0）
  三门限取 actual 所在行的值 -> tightness = thr/actual
用法(远程 NPU): ASCEND_RT_VISIBLE_DEVICES=<die> <PY> tightness_recalib.py --out tightness.json
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List, Tuple

import torch
import torch_npu  # noqa: F401

import utils

DEV = "npu:0"
DTYPE = torch.float32
_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp64": torch.float64}
# fp64 用 mpmath 高精度 base 算真实校验差（匹配论文 gpu_h100 compute_checksum_diff_mpmath）。
try:
    from mpmath import mpf, fsum as _fsum
    HAS_MPMATH = True
except ImportError:
    HAS_MPMATH = False
SIZES: List[int] = [128, 256, 512, 1024, 2048]
DISTS: List[Dict] = [
    {"name": "U(-1,1)", "func": utils.generate_matrice_uniform, "kw": {"lower": -1, "upper": 1}},
    {"name": "N(0,1)", "func": utils.generate_matrice, "kw": {"std": 1, "mean": 0}},
    {"name": "N(1,1)", "func": utils.generate_matrice, "kw": {"std": 1, "mean": 1}},
]


def mpmath_d1(a, b, c):
    """Exact per-row checksum diff |A@sum(B) - sum(C)| via mpmath (high-precision base)."""
    M, K = a.shape
    N = b.shape[1]
    a, b, c = a.cpu(), b.cpu(), c.cpu()
    brow = [_fsum(mpf(b[k, j].item()) for j in range(N)) for k in range(K)]
    csum = [_fsum(mpf(a[i, k].item()) * brow[k] for k in range(K)) for i in range(M)]
    crow = [_fsum(mpf(c[i, j].item()) for j in range(N)) for i in range(M)]
    return [abs(float(csum[i]) - float(crow[i])) for i in range(M)]


def measure(dist: Dict, n: int, trials: int) -> Dict[str, float]:
    """clean 下的紧度。返回 actual/三门限/三 tightness/两优势倍数（跨 trial 取 actual 最大者）。"""
    best_actual = -1.0
    best: Dict[str, float] = {}
    for _ in range(trials):
        a = dist["func"](n, n, device=DEV, dtype=DTYPE, **dist["kw"])
        b = dist["func"](n, n, device=DEV, dtype=DTYPE, **dist["kw"])
        c = torch.matmul(a, b)
        if DTYPE == torch.float64 and HAS_MPMATH:
            d1_list = mpmath_d1(a, b, c)
            i = max(range(len(d1_list)), key=lambda r: d1_list[r])
            actual = d1_list[i]
        else:
            d1 = torch.abs(torch.matmul(a, torch.sum(b, dim=1)) - torch.sum(c, dim=1))  # (n,)
            i = int(torch.argmax(d1).item())
            actual = float(d1[i].item())
        if actual <= best_actual:
            continue
        v_new = utils.my_bound_improve_robust(a, b, dtype=DTYPE)
        v_old = utils.my_bound_improve_robust(a, b, dtype=DTYPE, e_max=2e-6, c_sigma=2.5)
        _, a_thr = utils.aabft_corrected(a, b, c)
        best_actual = actual
        best = {"actual": actual,
                "V_new": float(v_new[i].item()), "V_old": float(v_old[i].item()),
                "A": float(a_thr[i].item())}
    # tightness = 门限/实际；优势 = A_tight / V_tight
    tv_new = best["V_new"] / best["actual"]
    tv_old = best["V_old"] / best["actual"]
    ta = best["A"] / best["actual"]
    return {**best,
            "Vtight_new": tv_new, "Vtight_old": tv_old, "Atight": ta,
            "adv_new": ta / tv_new, "adv_old": ta / tv_old}


def main() -> None:
    global DTYPE, DEV
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tightness_recalib.json")
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--dtype", default="fp32", choices=list(_DTYPES), help="compute dtype")
    args = ap.parse_args()
    DTYPE = _DTYPES[args.dtype]
    # Ascend NPU matmul only supports fp32/fp16/bf16; fp64 must run on CPU (fp64 BLAS
    # as the high-precision base).
    if DTYPE == torch.float64:
        DEV = "cpu"

    result: Dict = {"meta": {"dtype": args.dtype,
                             "new": "e_max=1.377e-6,c_sigma=5.39",
                             "old": "e_max=2e-6,c_sigma=2.5", "trials": args.trials}, "data": {}}
    for dist in DISTS:
        dn = dist["name"]
        result["data"][dn] = {}
        print(f"\n=== {dn} ===  (tightness = 门限/实际; 优势 = A/V)")
        print(f"  {'size':>5s} {'actual':>10s} {'A-tight':>9s} "
              f"{'V-tight(new)':>13s} {'V-tight(old)':>13s} {'adv_new':>8s} {'adv_old':>8s}")
        for n in SIZES:
            r = measure(dist, n, args.trials)
            result["data"][dn][str(n)] = r
            print(f"  {n:>5d} {r['actual']:>10.2e} {r['Atight']:>8.0f}x "
                  f"{r['Vtight_new']:>12.1f}x {r['Vtight_old']:>12.1f}x "
                  f"{r['adv_new']:>7.0f}x {r['adv_old']:>7.0f}x", flush=True)
            with open(args.out, "w") as f:
                json.dump(result, f, indent=2)
    print(f"\nsaved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
