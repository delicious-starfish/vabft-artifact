"""Unified-threshold comparison (CPU): Higham & SEA-ABFT vs A-ABFT & V-ABFT.

Computes, on the SAME row checksum difference
    D1[i] = A[i,:] . (sum_col B) - sum_col C[i,:]
the tightness (threshold/actual), per-bit detection rate (DR) and clean
false-positive rate (FPR) for four threshold methods:

  Higham : forward-error bound  gamma_K * sum_k |A[i,k]| * (sum_j |B[k,j]|)   (utils.higham_bound)
  SEA    : SEA-ABFT (Roy-Chowdhury & Banerjee) 2-norm checksum tolerance      (utils.sea_bound)
  A-ABFT : 4-component worst-case bound                                       (utils.aabft_corrected)
  V-ABFT : variance bound, recalibrated e_max/c_sigma (this paper)            (utils.my_bound_improve_robust)

This is a CPU port of tightness_recalib.py + recalib_detection.py (DEV="cpu",
no torch_npu). It stubs the torch_npu module so utils.py imports on a CPU host.
It is meant purely for offline table generation; it does not touch any .tex file.

Usage:
    python3 baseline_compare.py --smoke            # tiny run to validate wiring
    python3 baseline_compare.py                    # full run (fp32 required rows)
    python3 baseline_compare.py --dtypes fp32,fp64,bf16
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from typing import Dict, List, Tuple

# --- stub torch_npu so `import utils` (which does `import torch_npu`) works on CPU ---
if "torch_npu" not in sys.modules:
    _fake = types.ModuleType("torch_npu")
    _fake.npu = types.SimpleNamespace(is_available=lambda: False)
    sys.modules["torch_npu"] = _fake

import torch  # noqa: E402

import utils  # noqa: E402

DEV = "cpu"

try:
    from mpmath import mpf, fsum as _fsum
    HAS_MPMATH = True
except ImportError:  # pragma: no cover
    HAS_MPMATH = False

_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp64": torch.float64}

# tightness harness (square M=N=K), mirrors tightness_recalib.py
SIZES: List[int] = [128, 256, 512, 1024, 2048]
SIZES_FP64: List[int] = [128, 256, 512]  # mpmath base is expensive -> cap sizes
SIZES_BF16: List[int] = [128, 256, 512, 1024]  # CPU bf16 matmul is very slow -> cap sizes
DISTS: List[Dict] = [
    {"name": "U(-1,1)", "func": utils.generate_matrice_uniform, "kw": {"lower": -1, "upper": 1}},
    {"name": "N(0,1)", "func": utils.generate_matrice, "kw": {"std": 1, "mean": 0}},
    {"name": "N(1,1)", "func": utils.generate_matrice, "kw": {"std": 1, "mean": 1}},
]

# detection harness (rectangular), mirrors recalib_detection.py
BITS: List[int] = list(range(31, 15, -1))  # fp32 output bits 31..16
DR_TABLE_BITS: List[int] = [16, 17, 18, 19, 20]
DET_DISTS: List[Dict] = [
    {"name": "N(1e-6,1)", "func": utils.generate_matrice, "kw": {"std": 1, "mean": 1e-6}},
    {"name": "N(1,1)", "func": utils.generate_matrice, "kw": {"std": 1, "mean": 1}},
    {"name": "U(-1,1)", "func": utils.generate_matrice_uniform, "kw": {"lower": -1, "upper": 1}},
    {"name": "TruncN", "func": utils.generate_matrice_clamp, "kw": {"std": 1, "mean": 0}},
]
CONFIGS: Dict[str, Tuple[int, int, int]] = {
    "128x1024x256": (128, 1024, 256),
    "128x4096x256": (128, 4096, 256),
}
METHODS: List[str] = ["Higham", "SEA", "A-ABFT", "V-ABFT"]


def all_thresholds(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor,
                   dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    """All four per-row (M,) thresholds for operands (a, b) and product c."""
    _, aabft = utils.aabft_corrected(a, b, c)
    return {
        "Higham": utils.higham_bound(a, b),
        "SEA": utils.sea_bound(a, b),
        "A-ABFT": aabft,
        "V-ABFT": utils.my_bound_improve_robust(a, b, dtype=dtype),
    }


def d1_vector(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor,
              base_fp32: bool = False) -> torch.Tensor:
    """Row checksum diff D1 = A.(sum_col B) - sum_col C.

    base_fp32=True upcasts to fp32 for the reductions/checksum (used for bf16
    compute, so D1 isolates the matmul rounding error rather than adding
    reduction noise).
    """
    if base_fp32:
        af, bf, cf = a.float(), b.float(), c.float()
        return torch.matmul(af, torch.sum(bf, dim=1)) - torch.sum(cf, dim=1)
    return torch.matmul(a, torch.sum(b, dim=1)) - torch.sum(c, dim=1)


def mpmath_d1(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> List[float]:
    """Exact per-row |A@sum(B) - sum(C)| via mpmath (high-precision base for fp64)."""
    al = a.cpu().tolist()
    bl = b.cpu().tolist()
    cl = c.cpu().tolist()
    m, k = a.shape
    n = b.shape[1]
    brow = [_fsum(mpf(bl[kk][j]) for j in range(n)) for kk in range(k)]
    csum = [_fsum(mpf(al[i][kk]) * brow[kk] for kk in range(k)) for i in range(m)]
    crow = [_fsum(mpf(cl[i][j]) for j in range(n)) for i in range(m)]
    return [abs(float(csum[i]) - float(crow[i])) for i in range(m)]


# ----------------------------------------------------------------------------
# Tightness
# ----------------------------------------------------------------------------

def measure_tightness(dist: Dict, n: int, trials: int, dtype: torch.dtype) -> Dict[str, float]:
    """clean tightness for one (dist, size, dtype): keeps the worst-case (max actual) trial."""
    base_fp32 = dtype == torch.bfloat16
    use_mpmath = dtype == torch.float64 and HAS_MPMATH
    best_actual = -1.0
    best: Dict[str, float] = {}
    for _ in range(trials):
        a = dist["func"](n, n, device=DEV, dtype=dtype, **dist["kw"])
        b = dist["func"](n, n, device=DEV, dtype=dtype, **dist["kw"])
        c = torch.matmul(a, b)
        if use_mpmath:
            d1_list = mpmath_d1(a, b, c)
            i = max(range(len(d1_list)), key=lambda r: d1_list[r])
            actual = d1_list[i]
        else:
            d1 = torch.abs(d1_vector(a, b, c, base_fp32=base_fp32))
            i = int(torch.argmax(d1).item())
            actual = float(d1[i].item())
        if actual <= best_actual:
            continue
        thr = all_thresholds(a, b, c, dtype)
        best_actual = actual
        best = {"actual": actual}
        for name in METHODS:
            best[name] = float(thr[name].flatten()[i].item())
    out = dict(best)
    for name in METHODS:
        out[f"t_{name}"] = (best[name] / best["actual"]) if best["actual"] > 0 else float("inf")
    return out


def run_tightness(dtypes: List[str], trials: int, fp64_trials: int,
                  bf16_trials: int) -> Dict:
    result: Dict = {}
    for dk in dtypes:
        dtype = _DTYPES[dk]
        if dtype == torch.float64:
            sizes, t = SIZES_FP64, fp64_trials
        elif dtype == torch.bfloat16:
            sizes, t = SIZES_BF16, bf16_trials
        else:
            sizes, t = SIZES, trials
        result[dk] = {}
        print(f"\n===== TIGHTNESS {dk} (trials={t}) =====", flush=True)
        for dist in DISTS:
            dn = dist["name"]
            result[dk][dn] = {}
            print(f"  --- {dn} ---   (tightness = thr/actual, lower=tighter)")
            print(f"    {'size':>5s} {'actual':>10s} "
                  f"{'Higham':>11s} {'SEA':>11s} {'A-ABFT':>10s} {'V-ABFT':>9s}")
            for n in sizes:
                r = measure_tightness(dist, n, t, dtype)
                result[dk][dn][str(n)] = r
                print(f"    {n:>5d} {r['actual']:>10.2e} "
                      f"{r['t_Higham']:>10.0f}x {r['t_SEA']:>10.0f}x "
                      f"{r['t_A-ABFT']:>9.0f}x {r['t_V-ABFT']:>8.1f}x", flush=True)
    return result


# ----------------------------------------------------------------------------
# Detection rate (DR) + false positive rate (FPR), fp32 only
# ----------------------------------------------------------------------------

def measure_dr(dist: Dict, m: int, k: int, n: int, trials: int) -> Dict[str, Dict[int, float]]:
    """Per-bit detection rate: single bit-flip into C[i,j], is row i flagged?"""
    out: Dict[str, Dict[int, float]] = {s: {} for s in METHODS}
    for bit in BITS:
        det = {s: 0 for s in METHODS}
        valid = 0
        attempts = 0
        while valid < trials and attempts < trials * 5:
            attempts += 1
            a = dist["func"](m, k, device=DEV, dtype=torch.float32, **dist["kw"])
            b = dist["func"](k, n, device=DEV, dtype=torch.float32, **dist["kw"])
            c = torch.matmul(a, b)
            i = int(torch.randint(0, m, (1,)).item())
            j = int(torch.randint(0, n, (1,)).item())
            flipped, ok = utils.flip_infuse(c[i, j].cpu(), bit)
            if not ok or not torch.isfinite(flipped):
                continue
            c[i, j] = flipped
            valid += 1
            thr = all_thresholds(a, b, c, torch.float32)
            d1a = float(torch.abs(d1_vector(a, b, c)[i]).item())
            for s in METHODS:
                if d1a > float(thr[s].flatten()[i].item()):
                    det[s] += 1
        for s in METHODS:
            out[s][bit] = round(det[s] / valid * 100, 2) if valid else -1.0
    return out


def measure_fpr(dist: Dict, m: int, k: int, n: int, trials: int) -> Dict[str, float]:
    """clean FPR: fraction of rows with |D1[i]| > T_i (no injection)."""
    fp = {s: 0 for s in METHODS}
    rows = 0
    for _ in range(trials):
        a = dist["func"](m, k, device=DEV, dtype=torch.float32, **dist["kw"])
        b = dist["func"](k, n, device=DEV, dtype=torch.float32, **dist["kw"])
        c = torch.matmul(a, b)
        d1a = torch.abs(d1_vector(a, b, c))
        thr = all_thresholds(a, b, c, torch.float32)
        rows += m
        for s in METHODS:
            fp[s] += int(torch.sum(d1a > thr[s].flatten()).item())
    return {s: fp[s] / rows if rows else 0.0 for s in METHODS}


def run_detection(dr_trials: int, fpr_trials: int) -> Tuple[Dict, Dict]:
    dr_res: Dict = {}
    fpr_res: Dict = {}
    for cname, (m, k, n) in CONFIGS.items():
        print(f"\n===== DETECTION {cname} (M={m},K={k},N={n}) =====", flush=True)
        dr_res[cname] = {}
        fpr_res[cname] = {}
        for dist in DET_DISTS:
            dn = dist["name"]
            dr = measure_dr(dist, m, k, n, dr_trials)
            fpr = measure_fpr(dist, m, k, n, fpr_trials)
            dr_res[cname][dn] = dr
            fpr_res[cname][dn] = fpr
            print(f"  [{dn:10s}] DR@b16 H/S/A/V = "
                  f"{dr['Higham'][16]:.0f}/{dr['SEA'][16]:.0f}/"
                  f"{dr['A-ABFT'][16]:.0f}/{dr['V-ABFT'][16]:.0f}%  | "
                  f"FPR H/S/A/V = {fpr['Higham']:.1e}/{fpr['SEA']:.1e}/"
                  f"{fpr['A-ABFT']:.1e}/{fpr['V-ABFT']:.1e}", flush=True)
    return dr_res, fpr_res


# ----------------------------------------------------------------------------
# Markdown rendering
# ----------------------------------------------------------------------------

def _fmt_tight(v: float) -> str:
    if v == float("inf"):
        return "inf"
    if v >= 1000:
        return f"{v:,.0f}x"
    if v >= 100:
        return f"{v:.0f}x"
    return f"{v:.1f}x"


def render_md(meta: Dict, tight: Dict, dr: Dict, fpr: Dict) -> str:
    L: List[str] = []
    L.append("# Unified threshold comparison: Higham / SEA / A-ABFT / V-ABFT (CPU)\n")
    L.append(f"- generated by `baseline_compare.py` on CPU (DEV=cpu), torch {torch.__version__}")
    L.append(f"- V-ABFT recalibrated: {meta['vabft']}")
    L.append(f"- Higham u: fp32 2^-24 / bf16 2^-8 / fp64 2^-53 ; "
             f"SEA eps_M = 2u ; A-ABFT 3-sigma worst-case")
    L.append(f"- tightness trials={meta['trials']} (fp64={meta['fp64_trials']}), "
             f"DR trials/bit={meta['dr_trials']}, FPR trials={meta['fpr_trials']}")
    L.append("- **tightness = threshold / actual (max_i |D1[i]|), lower = tighter**\n")

    L.append("## Tightness (threshold / actual)\n")
    for dk, dd in tight.items():
        L.append(f"### compute = {dk}\n")
        for dn, sizes in dd.items():
            L.append(f"**{dn}**\n")
            L.append("| size | actual | Higham | SEA | A-ABFT | V-ABFT |")
            L.append("|---|---|---|---|---|---|")
            for sz, r in sizes.items():
                L.append(f"| {sz} | {r['actual']:.2e} | {_fmt_tight(r['t_Higham'])} | "
                         f"{_fmt_tight(r['t_SEA'])} | {_fmt_tight(r['t_A-ABFT'])} | "
                         f"{_fmt_tight(r['t_V-ABFT'])} |")
            L.append("")

    L.append("## Detection rate (DR, %) at bits 16-20  (loose bound -> lower DR)\n")
    for cname, dd in dr.items():
        L.append(f"### {cname}\n")
        L.append("| dist | method | b16 | b17 | b18 | b19 | b20 |")
        L.append("|---|---|---|---|---|---|---|")
        for dn, methods in dd.items():
            for s in METHODS:
                cells = " | ".join(f"{methods[s][bit]:.0f}" for bit in DR_TABLE_BITS)
                L.append(f"| {dn} | {s} | {cells} |")
        L.append("")

    L.append("## False positive rate (FPR, clean)\n")
    L.append("| config | dist | Higham | SEA | A-ABFT | V-ABFT |")
    L.append("|---|---|---|---|---|---|")
    for cname, dd in fpr.items():
        for dn, methods in dd.items():
            L.append(f"| {cname} | {dn} | {methods['Higham']:.2e} | {methods['SEA']:.2e} | "
                     f"{methods['A-ABFT']:.2e} | {methods['V-ABFT']:.2e} |")
    L.append("")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtypes", default="fp32", help="comma list: fp32,fp64,bf16")
    ap.add_argument("--trials", type=int, default=100, help="tightness trials (fp32)")
    ap.add_argument("--fp64_trials", type=int, default=15)
    ap.add_argument("--bf16_trials", type=int, default=10, help="CPU bf16 matmul is slow")
    ap.add_argument("--dr_trials", type=int, default=200)
    ap.add_argument("--fpr_trials", type=int, default=2000)
    ap.add_argument("--out_json", default="baseline_compare.json")
    ap.add_argument("--out_md", default="baseline_compare_table.md")
    ap.add_argument("--no_detection", action="store_true", help="skip DR/FPR")
    ap.add_argument("--smoke", action="store_true", help="tiny run to validate wiring")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.smoke:
        args.trials = 3
        args.fp64_trials = 3
        args.bf16_trials = 3
        args.dr_trials = 20
        args.fpr_trials = 30

    torch.manual_seed(args.seed)
    dtypes = [d.strip() for d in args.dtypes.split(",") if d.strip()]

    meta = {"dtypes": dtypes, "trials": args.trials, "fp64_trials": args.fp64_trials,
            "bf16_trials": args.bf16_trials, "dr_trials": args.dr_trials,
            "fpr_trials": args.fpr_trials, "vabft": "e_max(fp32)=1.377e-6, c_sigma=5.39",
            "smoke": args.smoke, "sizes": SIZES, "sizes_fp64": SIZES_FP64,
            "sizes_bf16": SIZES_BF16, "configs": list(CONFIGS)}

    tight = run_tightness(dtypes, args.trials, args.fp64_trials, args.bf16_trials)
    if args.no_detection:
        dr, fpr = {}, {}
    else:
        dr, fpr = run_detection(args.dr_trials, args.fpr_trials)

    result = {"meta": meta, "tightness": tight, "DR": dr, "FPR": fpr}
    with open(args.out_json, "w") as f:
        json.dump(result, f, indent=2)
    md = render_md(meta, tight, dr, fpr)
    with open(args.out_md, "w") as f:
        f.write(md)
    print(f"\nsaved -> {args.out_json} , {args.out_md}", flush=True)


if __name__ == "__main__":
    main()
