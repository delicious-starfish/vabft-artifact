#!/usr/bin/env python3
"""V-ABFT exponent-bit fault-injection experiment on Ascend NPU.

This is the unified successor to ``vabft_acc.py``:

* BF16 bits 7..14 and FP32 bits 23..30 are tested;
* every bit is flipped with XOR in the 0->1 direction only;
* each injection has at most 20 random candidate positions; if none has a
  clear target bit, A/B/C are regenerated and that trial is retried;
* candidate selections are capped at 20 times the condition's maximum total
  injections, so repeated matrix regeneration cannot loop forever;
* A/B/C are regenerated periodically (``--refresh-interval``, default 256);
* C is actually modified before checksums and thresholds are computed;
* detection is evaluated per 128x256 output block;
* row detection, block detection, false positives, NaN/Inf and exact column
  localization are reported.

The default matrix and experiment grid follow ``TEST_SPEC.md``.  A full run
is intentionally large; use --k-values, --dtypes, --distributions and
--trials for smoke tests.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import torch
import torch_npu
from tqdm import tqdm

import utils


BLOCK_M = 128
BLOCK_N = 256
DEFAULT_K_VALUES = (1024, 2048, 3072, 4096, 6144, 8192)


@dataclass(frozen=True)
class Distribution:
    name: str
    generator: Callable[..., torch.Tensor]
    params: dict[str, float]


DISTRIBUTIONS = {
    "N(1,1)": Distribution(
        "N(1,1)", utils.generate_matrice, {"std": 1.0, "mean": 1.0}
    ),
    "N(1e-6,1)": Distribution(
        "N(1e-6,1)", utils.generate_matrice, {"std": 1.0, "mean": 1e-6}
    ),
    # Historical vabft_acc.py used mean=1, std=1.  Current utils clamps to
    # [mean-std, mean+std], hence this is clamp(N(1,1), 0, 2).
    "TruncN": Distribution(
        "TruncN", utils.generate_matrice_clamp, {"std": 1.0, "mean": 1.0}
    ),
    "U(-1,1)": Distribution(
        "U(-1,1)",
        utils.generate_matrice_uniform,
        {"lower": -1.0, "upper": 1.0},
    ),
}

DTYPES = {
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}

EXPONENT_BITS = {
    "bf16": tuple(range(7, 15)),
    "fp32": tuple(range(23, 31)),
}


@dataclass
class Counters:
    total_blocks: int = 0
    all_injected: int = 0
    non_nan_injected: int = 0
    nan_injected: int = 0
    inf_injected: int = 0
    finite_injected: int = 0
    non_nan_injected_rows: int = 0
    non_nan_detected_rows: int = 0
    non_nan_injected_blocks: int = 0
    non_nan_detected_blocks: int = 0
    non_nan_localized: int = 0
    nan_injected_rows: int = 0
    nan_detected_rows: int = 0
    nan_injected_blocks: int = 0
    nan_detected_blocks: int = 0
    nan_localized: int = 0
    # The unprefixed fields are totals across non-NaN and NaN faults.
    injected_rows: int = 0
    detected_rows: int = 0
    false_rows: int = 0
    clean_rows: int = 0
    injected_blocks: int = 0
    detected_blocks: int = 0
    false_blocks: int = 0
    clean_blocks: int = 0
    localized: int = 0


@dataclass(frozen=True)
class InjectedFault:
    block_id: int
    row_start: int
    row_end: int
    col_start: int
    col_end: int
    local_row: int
    local_col: int
    global_row: int
    global_col: int
    direction: str
    raw_before: str
    raw_after: str
    classification: str


def parse_csv_values(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_k_values(value: str) -> list[int]:
    values = [int(item) for item in parse_csv_values(value)]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("K values must be positive integers")
    return values


def parse_bits(value: str) -> list[int]:
    values = [int(item) for item in parse_csv_values(value)]
    if not values or any(item < 0 or item > 31 for item in values):
        raise argparse.ArgumentTypeError("bits must be integers in [0, 31]")
    return values


def stable_condition_seed(
    seed: int, k: int, dtype_name: str, distribution: str, bit: int
) -> int:
    payload = f"{seed}|{k}|{dtype_name}|{distribution}|{bit}".encode()
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little") & 0x7FFF_FFFF_FFFF_FFFF


def set_seeds(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch_npu.npu.is_available():
        torch_npu.npu.manual_seed_all(seed)


def draw_unused_position(
    rng: random.Random,
    used_positions: set[tuple[int, int]],
    row_start: int,
    row_end: int,
    col_start: int,
    col_end: int,
) -> tuple[int, int]:
    """Uniformly draw an unused coordinate without reserving it yet."""
    height = row_end - row_start
    width = col_end - col_start
    capacity = height * width
    if len(used_positions) >= capacity:
        raise RuntimeError(
            f"all {capacity} positions in block "
            f"rows[{row_start}:{row_end}) cols[{col_start}:{col_end}) were used; "
            "reduce --refresh-interval"
        )

    # Rejection sampling is uniform and fast at the intended occupancy
    # (default: at most 256 / 32768 positions per block).
    while True:
        row = rng.randrange(row_start, row_end)
        col = rng.randrange(col_start, col_end)
        if (row, col) not in used_positions:
            return row, col


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def safe_version(module: object) -> str:
    return str(getattr(module, "__version__", "unknown"))


def rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def xor_scalar(
    value: torch.Tensor, bit: int, dtype: torch.dtype
) -> tuple[torch.Tensor, str, str, str]:
    """Return an exact XOR-flipped CPU scalar and its direction."""
    cpu_value = value.detach().to("cpu").contiguous().reshape(())
    if dtype == torch.bfloat16:
        integer = cpu_value.view(torch.int16)
        mask = torch.tensor(1 << bit, dtype=torch.int16)
    elif dtype == torch.float32:
        integer = cpu_value.view(torch.int32)
        mask = torch.tensor(1 << bit, dtype=torch.int32)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    width = 16 if dtype == torch.bfloat16 else 32
    unsigned_mask = (1 << width) - 1
    before_raw = int(integer.item()) & unsigned_mask
    before_set = bool((before_raw & (1 << bit)) != 0)
    flipped = torch.bitwise_xor(integer, mask).view(dtype)
    after_raw = before_raw ^ (1 << bit)
    hex_width = width // 4
    return (
        flipped,
        "1->0" if before_set else "0->1",
        f"0x{before_raw:0{hex_width}x}",
        f"0x{after_raw:0{hex_width}x}",
    )


def detected_mask(d1: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
    """A NaN/Inf difference or an absolute difference over threshold is detected."""
    return torch.isnan(d1) | (d1.abs() >= threshold)


def write_audit(handle, record: dict[str, object]) -> None:
    if handle is not None:
        handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def run_condition(
    *,
    m: int,
    n: int,
    k: int,
    dtype_name: str,
    distribution: Distribution,
    bit: int,
    trials: int,
    refresh_interval: int,
    p_inject: float,
    seed: int,
    device: torch.device,
    block_m: int,
    block_n: int,
    audit_handle,
    show_progress: bool,
    progress_label: str,
) -> dict[str, object]:
    dtype = DTYPES[dtype_name]
    condition_seed = stable_condition_seed(seed, k, dtype_name, distribution.name, bit)
    rng = random.Random(condition_seed)
    counters = Counters()
    a: torch.Tensor | None = None
    b: torch.Tensor | None = None
    c: torch.Tensor | None = None
    matrix_generation = -1
    matrix_seed = condition_seed
    used_positions_by_block: dict[int, set[tuple[int, int]]] = {}
    blocks_per_trial = math.ceil(m / block_m) * math.ceil(n / block_n)
    # A block has at most one injected element in a trial.  p_inject makes the
    # realised count random, so use the deterministic upper bound here.
    max_injection_attempts = 20 * trials * blocks_per_trial
    injection_attempts = 0

    def refresh_matrices() -> None:
        """Create a new consistent A/B/C set and reset per-block reservations."""
        nonlocal a, b, c, matrix_generation, matrix_seed
        matrix_generation += 1
        matrix_seed = stable_condition_seed(
            condition_seed, k, dtype_name, distribution.name, matrix_generation
        )
        set_seeds(matrix_seed)
        used_positions_by_block.clear()
        a = distribution.generator(
            m, k, device=device, dtype=dtype, **distribution.params
        )
        b = distribution.generator(
            k, n, device=device, dtype=dtype, **distribution.params
        )
        c = torch.matmul(a, b)

    trial_iterator = tqdm(
        range(trials),
        desc=progress_label,
        unit="trial",
        leave=False,
        dynamic_ncols=True,
        disable=not show_progress,
    )
    for trial in trial_iterator:
        # Refresh A/B/C at a deterministic interval.  C is always generated
        # as A @ B; merely reseeding C independently would violate GEMM.
        if trial % refresh_interval == 0:
            refresh_matrices()

        assert a is not None and b is not None and c is not None
        # An injection only accepts a clear target bit (0->1).  If a selected
        # block cannot provide one in 20 draws, start the entire trial over
        # with a new consistent A/B/C set; partial faults must not survive the
        # replacement of their source matrix.  The condition-wide candidate
        # count is capped to make this retry path finite.
        while True:
            # The required order is: C=A@B -> copy C -> inject into C_faulty
            # -> calculate threshold/checksums -> compare.
            c_faulty = c.clone()
            faults: dict[int, InjectedFault] = {}
            block_id = 0
            retry_with_new_matrices = False

            for col_start in range(0, n, block_n):
                col_end = min(col_start + block_n, n)
                for row_start in range(0, m, block_m):
                    row_end = min(row_start + block_m, m)
                    if rng.random() < p_inject:
                        used_positions = used_positions_by_block.setdefault(
                            block_id, set()
                        )
                        fault: InjectedFault | None = None
                        for _attempt in range(20):
                            if injection_attempts >= max_injection_attempts:
                                raise RuntimeError(
                                    "maximum injection candidate attempts exceeded "
                                    f"({max_injection_attempts} = 20 * {trials} "
                                    f"trials * {blocks_per_trial} blocks); "
                                    "could not complete 0->1 injection"
                                )
                            injection_attempts += 1
                            global_row, global_col = draw_unused_position(
                                rng,
                                used_positions,
                                row_start,
                                row_end,
                                col_start,
                                col_end,
                            )
                            before = c_faulty[global_row, global_col]
                            after_cpu, direction, raw_before, raw_after = xor_scalar(
                                before, bit, dtype
                            )
                            if direction != "0->1":
                                continue

                            used_positions.add((global_row, global_col))
                            c_faulty[global_row, global_col] = after_cpu.to(
                                device=device
                            )
                            if bool(torch.isnan(after_cpu).item()):
                                classification = "nan"
                            elif bool(torch.isinf(after_cpu).item()):
                                classification = "inf"
                            else:
                                classification = "finite"
                            fault = InjectedFault(
                                block_id=block_id,
                                row_start=row_start,
                                row_end=row_end,
                                col_start=col_start,
                                col_end=col_end,
                                local_row=global_row - row_start,
                                local_col=global_col - col_start,
                                global_row=global_row,
                                global_col=global_col,
                                direction=direction,
                                raw_before=raw_before,
                                raw_after=raw_after,
                                classification=classification,
                            )
                            break

                        if fault is None:
                            retry_with_new_matrices = True
                            break
                        faults[block_id] = fault
                    block_id += 1
                if retry_with_new_matrices:
                    break

            if not retry_with_new_matrices:
                counters.total_blocks += block_id
                counters.all_injected += len(faults)
                break
            refresh_matrices()
            assert a is not None and b is not None and c is not None

        # Everything below is deliberately computed after C_faulty has been
        # modified.  No clean-D1-plus-delta shortcut is used.
        a_fp32 = a.float()
        block_id = 0
        for col_start in range(0, n, block_n):
            col_end = min(col_start + block_n, n)
            width = col_end - col_start
            b_slice = b[:, col_start:col_end]
            b_fp32 = b_slice.float()
            c_slice_fp32 = c_faulty[:, col_start:col_end].float()
            weights = (
                torch.arange(width, dtype=torch.float32, device=device)
                - (width - 1) / 2.0
            )

            # Threshold is recalculated only after fault injection, as is D1/D2.
            threshold_all = utils.my_bound_improve_robust(
                a, b_slice, dtype=dtype
            ).float()
            abe = (a_fp32 @ b_fp32.sum(dim=1, keepdim=True)).squeeze(1)
            abw = (a_fp32 @ (b_fp32 @ weights.unsqueeze(1))).squeeze(1)
            d1_all = abe - c_slice_fp32.sum(dim=1)
            d2_all = abw - c_slice_fp32 @ weights

            for row_start in range(0, m, block_m):
                row_end = min(row_start + block_m, m)
                height = row_end - row_start
                d1 = d1_all[row_start:row_end]
                d2 = d2_all[row_start:row_end]
                threshold = threshold_all[row_start:row_end]
                # The detection rule is uniform: D1 is NaN/Inf, or
                # abs(D1) >= threshold. NaN is separated only in statistics.
                reported = detected_mask(d1, threshold)
                fault = faults.get(block_id)

                if fault is None:
                    false_count = int(reported.sum().item())
                    counters.clean_blocks += 1
                    counters.clean_rows += height
                    counters.false_rows += false_count
                    if false_count:
                        counters.false_blocks += 1
                    block_id += 1
                    continue

                # All non-injected rows remain eligible for row-level FPR.
                other_reported = reported.clone()
                other_reported[fault.local_row] = False
                counters.clean_rows += height - 1
                counters.false_rows += int(other_reported.sum().item())

                if fault.classification == "nan":
                    counters.nan_injected += 1
                    counters.nan_injected_rows += 1
                    counters.nan_injected_blocks += 1
                    counters.injected_rows += 1
                    counters.injected_blocks += 1

                    block_detected = bool(reported.any().item())
                    row_detected = bool(reported[fault.local_row].item())
                    # D1 and D2 are both NaN, so D2 / D1 cannot recover a
                    # column. Do not inspect C to manufacture a localization.
                    localized = False
                    if block_detected:
                        counters.nan_detected_blocks += 1
                        counters.detected_blocks += 1
                    if row_detected:
                        counters.nan_detected_rows += 1
                        counters.detected_rows += 1
                    write_audit(
                        audit_handle,
                        {
                            "trial": trial,
                            "K": k,
                            "dtype": dtype_name,
                            "distribution": distribution.name,
                            "condition_seed": condition_seed,
                            "matrix_generation": matrix_generation,
                            "matrix_seed": matrix_seed,
                            **asdict(fault),
                            "bit": bit,
                            "estimated_local_column": None,
                            "block_detected": block_detected,
                            "row_detected": row_detected,
                            "detected": row_detected,
                            "localized": localized,
                        },
                    )
                    block_id += 1
                    continue

                counters.non_nan_injected += 1
                counters.non_nan_injected_blocks += 1
                counters.non_nan_injected_rows += 1
                counters.injected_blocks += 1
                counters.injected_rows += 1
                if fault.classification == "inf":
                    counters.inf_injected += 1
                else:
                    counters.finite_injected += 1

                row_detected = bool(reported[fault.local_row].item())
                localized_col: int | None = None
                localized = False
                if row_detected:
                    counters.non_nan_detected_rows += 1
                    counters.non_nan_detected_blocks += 1
                    counters.detected_rows += 1
                    counters.detected_blocks += 1
                    d1_value = d1[fault.local_row]
                    d2_value = d2[fault.local_row]
                    if bool(torch.isfinite(d1_value).item()) and bool(
                        torch.isfinite(d2_value).item()
                    ) and float(d1_value.abs().item()) > 0.0:
                        estimate = torch.round(
                            d2_value / d1_value + (width - 1) / 2.0
                        )
                        localized_col = max(
                            0, min(int(estimate.item()), width - 1)
                        )
                        localized = localized_col == fault.local_col
                        if localized:
                            counters.non_nan_localized += 1
                            counters.localized += 1

                write_audit(
                    audit_handle,
                    {
                        "trial": trial,
                        "K": k,
                        "dtype": dtype_name,
                        "distribution": distribution.name,
                        "condition_seed": condition_seed,
                        "matrix_generation": matrix_generation,
                        "matrix_seed": matrix_seed,
                        **asdict(fault),
                        "bit": bit,
                        "estimated_local_column": localized_col,
                        "block_detected": row_detected,
                        "row_detected": row_detected,
                        "detected": row_detected,
                        "localized": localized,
                    },
                )
                block_id += 1

        del c_faulty

    assert counters.all_injected == (
        counters.non_nan_injected + counters.nan_injected
    )
    assert counters.injected_blocks == (
        counters.non_nan_injected_blocks + counters.nan_injected_blocks
    )
    assert counters.detected_blocks == (
        counters.non_nan_detected_blocks + counters.nan_detected_blocks
    )
    assert counters.injected_rows == (
        counters.non_nan_injected_rows + counters.nan_injected_rows
    )
    assert counters.detected_rows == (
        counters.non_nan_detected_rows + counters.nan_detected_rows
    )
    assert counters.localized == (
        counters.non_nan_localized + counters.nan_localized
    )

    result: dict[str, object] = asdict(counters)
    result.update(
        {
            "seed": condition_seed,
            "base_seed": seed,
            "M": m,
            "N": n,
            "K": k,
            "dtype": dtype_name,
            "distribution": distribution.name,
            "bit": bit,
            "C": trials,
            "refresh_interval": refresh_interval,
            "p_inject": p_inject,
            "non_nan_row_DR": rate(
                counters.non_nan_detected_rows, counters.non_nan_injected_rows
            ),
            "non_nan_block_DR": rate(
                counters.non_nan_detected_blocks, counters.non_nan_injected_blocks
            ),
            "non_nan_localization_R": rate(
                counters.non_nan_localized, counters.non_nan_injected_rows
            ),
            "non_nan_conditional_localization_R": rate(
                counters.non_nan_localized, counters.non_nan_detected_rows
            ),
            "nan_row_DR": rate(counters.nan_detected_rows, counters.nan_injected_rows),
            "nan_block_DR": rate(
                counters.nan_detected_blocks, counters.nan_injected_blocks
            ),
            "nan_localization_R": rate(
                counters.nan_localized, counters.nan_injected_rows
            ),
            "nan_conditional_localization_R": rate(
                counters.nan_localized, counters.nan_detected_rows
            ),
            "row_DR": rate(counters.detected_rows, counters.injected_rows),
            "block_DR": rate(counters.detected_blocks, counters.injected_blocks),
            "row_FPR": rate(counters.false_rows, counters.clean_rows),
            "block_FPR": rate(counters.false_blocks, counters.clean_blocks),
            "localization_R": rate(counters.localized, counters.injected_rows),
            "conditional_localization_R": rate(
                counters.localized, counters.detected_rows
            ),
            "nan_rate": rate(counters.nan_injected, counters.all_injected),
            "inf_rate": rate(counters.inf_injected, counters.all_injected),
            "e_max": 8e-3
            if dtype == torch.bfloat16
            else 1.377e-6 * math.sqrt(k / 1024),
            "c_sigma": 5.39,
            "encoding": "linear_centered_per_block",
        }
    )

    del c, b, a
    torch_npu.npu.synchronize()
    return result


def summary_fieldnames() -> list[str]:
    prefix = [
        "run_id",
        "timestamp",
        "commit",
        "hostname",
        "device",
        "torch_version",
        "torch_npu_version",
        "python_version",
    ]
    condition = [
        "seed",
        "base_seed",
        "M",
        "N",
        "K",
        "dtype",
        "distribution",
        "bit",
        "C",
        "refresh_interval",
        "p_inject",
    ]
    counter_names = [item.name for item in fields(Counters)]
    metrics = [
        "non_nan_row_DR",
        "non_nan_block_DR",
        "non_nan_localization_R",
        "non_nan_conditional_localization_R",
        "nan_row_DR",
        "nan_block_DR",
        "nan_localization_R",
        "nan_conditional_localization_R",
        "row_DR",
        "block_DR",
        "row_FPR",
        "block_FPR",
        "localization_R",
        "conditional_localization_R",
        "nan_rate",
        "inf_rate",
        "e_max",
        "c_sigma",
        "encoding",
    ]
    return prefix + condition + counter_names + metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument(
        "--k-values",
        type=parse_k_values,
        default=list(DEFAULT_K_VALUES),
        help="comma-separated K values",
    )
    parser.add_argument(
        "--dtypes",
        type=parse_csv_values,
        default=list(DTYPES),
        help="comma-separated subset of: bf16,fp32",
    )
    parser.add_argument(
        "--bits",
        type=parse_bits,
        help="optional comma-separated bit subset; each bit must match its dtype",
    )
    parser.add_argument(
        "--distributions",
        nargs="+",
        choices=list(DISTRIBUTIONS),
        default=list(DISTRIBUTIONS),
        help="space-separated distribution names (quote names containing parentheses)",
    )
    parser.add_argument("--trials", "--C", dest="trials", type=int, default=10000)
    parser.add_argument(
        "--refresh-interval",
        type=int,
        default=256,
        help="regenerate seeded A/B/C after this many trials (default: 256)",
    )
    parser.add_argument("--p-inject", type=float, default=0.5)
    parser.add_argument(
        "--seed",
        type=int,
        help="base seed; default uses the current time in nanoseconds",
    )
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--block-m", type=int, default=BLOCK_M)
    parser.add_argument("--block-n", type=int, default=BLOCK_N)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("vabft_acc_hiPre_results.csv"),
    )
    parser.add_argument(
        "--audit-jsonl",
        type=Path,
        help="optional per-injection audit log (can be very large)",
    )
    parser.add_argument(
        "--append", action="store_true", help="append to an existing summary CSV"
    )
    parser.add_argument(
        "--no-progress", action="store_true", help="disable tqdm progress bars"
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.m <= 0 or args.n <= 0 or args.trials <= 0 or args.refresh_interval <= 0:
        raise ValueError("M, N, trials and refresh-interval must be positive")
    if args.block_m <= 0 or args.block_n <= 0:
        raise ValueError("block sizes must be positive")
    if not 0.0 <= args.p_inject <= 1.0:
        raise ValueError("p-inject must be in [0, 1]")
    unknown_dtypes = set(args.dtypes) - set(DTYPES)
    unknown_distributions = set(args.distributions) - set(DISTRIBUTIONS)
    if unknown_dtypes:
        raise ValueError(f"unknown dtypes: {sorted(unknown_dtypes)}")
    if unknown_distributions:
        raise ValueError(f"unknown distributions: {sorted(unknown_distributions)}")
    if args.bits is not None:
        invalid = {
            name: sorted(set(args.bits) - set(EXPONENT_BITS[name]))
            for name in args.dtypes
        }
        invalid = {name: bits for name, bits in invalid.items() if bits}
        if invalid:
            raise ValueError(f"bits outside the selected dtype exponent range: {invalid}")
    min_block_height = args.m % args.block_m or min(args.m, args.block_m)
    min_block_width = args.n % args.block_n or min(args.n, args.block_n)
    min_block_capacity = min_block_height * min_block_width
    if args.refresh_interval > min_block_capacity:
        raise ValueError(
            "refresh-interval exceeds the smallest block's number of unique "
            f"positions ({min_block_capacity})"
        )
    if not torch_npu.npu.is_available():
        raise RuntimeError("Ascend NPU is required; CPU fallback is not a formal run")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.seed is None:
        args.seed = time.time_ns() & 0x7FFF_FFFF_FFFF_FFFF
    try:
        validate_args(args)
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))

    device = torch.device(args.device)
    torch_npu.npu.set_device(device)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    metadata = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "commit": git_commit(),
        "hostname": platform.node(),
        "device": args.device,
        "torch_version": safe_version(torch),
        "torch_npu_version": safe_version(torch_npu),
        "python_version": platform.python_version(),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.audit_jsonl is not None:
        args.audit_jsonl.parent.mkdir(parents=True, exist_ok=True)

    file_exists = args.output.exists() and args.output.stat().st_size > 0
    summary_mode = "a" if args.append else "w"
    audit_mode = "a" if args.append else "w"

    total_conditions = sum(
        len(args.bits if args.bits is not None else EXPONENT_BITS[name])
        for _k in args.k_values
        for name in args.dtypes
        for _dist in args.distributions
    )
    completed = 0

    with args.output.open(summary_mode, newline="", encoding="utf-8") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=summary_fieldnames())
        if not (args.append and file_exists):
            writer.writeheader()

        audit_file = (
            args.audit_jsonl.open(audit_mode, encoding="utf-8")
            if args.audit_jsonl is not None
            else None
        )
        try:
            condition_progress = tqdm(
                total=total_conditions,
                desc="All conditions",
                unit="condition",
                dynamic_ncols=True,
                disable=args.no_progress,
            )
            try:
                for k in args.k_values:
                    for dtype_name in args.dtypes:
                        for distribution_name in args.distributions:
                            distribution = DISTRIBUTIONS[distribution_name]
                            bits = (
                                args.bits
                                if args.bits is not None
                                else EXPONENT_BITS[dtype_name]
                            )
                            for bit in bits:
                                completed += 1
                                label = (
                                    f"K={k} {dtype_name} {distribution_name} bit={bit}"
                                )
                                condition_progress.set_postfix_str(label)
                                if args.no_progress:
                                    print(
                                        f"[{completed}/{total_conditions}] {label}",
                                        flush=True,
                                    )
                                result = run_condition(
                                    m=args.m,
                                    n=args.n,
                                    k=k,
                                    dtype_name=dtype_name,
                                    distribution=distribution,
                                    bit=bit,
                                    trials=args.trials,
                                    refresh_interval=args.refresh_interval,
                                    p_inject=args.p_inject,
                                    seed=args.seed,
                                    device=device,
                                    block_m=args.block_m,
                                    block_n=args.block_n,
                                    audit_handle=audit_file,
                                    show_progress=not args.no_progress,
                                    progress_label=label,
                                )
                                writer.writerow({**metadata, **result})
                                summary_file.flush()
                                if audit_file is not None:
                                    audit_file.flush()
                                condition_progress.update(1)
                                condition_progress.set_postfix_str(
                                    f"{label} row_DR={result['row_DR']}"
                                )
                                if args.no_progress:
                                    print(
                                        "  "
                                        f"row_DR={result['row_DR']} "
                                        f"block_DR={result['block_DR']} "
                                        f"loc={result['localization_R']} "
                                        f"row_FPR={result['row_FPR']} "
                                        f"NaN={result['nan_rate']}",
                                        flush=True,
                                    )
            finally:
                condition_progress.close()
        finally:
            if audit_file is not None:
                audit_file.close()

    print(f"Results saved to {args.output}")
    if args.audit_jsonl is not None:
        print(f"Audit log saved to {args.audit_jsonl}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
