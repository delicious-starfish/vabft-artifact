#!/usr/bin/env python3
"""FP32 V-ABFT vs A-ABFT End-to-End Test - 使用原论文 A-ABFT 公式

与 run_e2e_k2048.py 的唯一区别:
  A-ABFT 检测门限改为原论文 (Braun et al. DSN'14) 的单内积公式:
    σ = √((K(K+1)(K+0.5)+2K)/24) × 2^(-t) × y_m
    threshold_m = 3σ
  其中 y_m = max_k |A[m,k] · B_rowsum[k]| (per-row, 原论文 per-element 方式)
"""

import torch
import math
import json
import sys
import numpy as np

device = torch.device('cpu')
dtype = torch.float32
trials_main = 100

M, K, N = 128, 2048, 256

FW_LINEAR = 0.577
FW_SINH = 0.354

# ============================================================================
# Distribution generators
# ============================================================================

def generate_normal(M_: int, K_: int, mean: float = 1e-6, std: float = 1):
    return torch.randn(M_, K_, device=device, dtype=dtype) * std + mean

def generate_uniform(M_: int, K_: int, lower: float = -1, upper: float = 1):
    return torch.rand(M_, K_, device=device, dtype=dtype) * (upper - lower) + lower

def generate_truncnorm(M_: int, K_: int, mean: float = 0, std: float = 1):
    x = torch.randn(M_, K_, device=device, dtype=dtype) * std + mean
    return torch.clamp(x, -3 * std + mean, 3 * std + mean)

all_dists = [
    {'name': 'N(1e-6,1)', 'func': generate_normal, 'params': {'mean': 1e-6, 'std': 1}},
    {'name': 'N(1,1)',     'func': generate_normal, 'params': {'mean': 1, 'std': 1}},
    {'name': 'U(-1,1)',    'func': generate_uniform, 'params': {'lower': -1, 'upper': 1}},
    {'name': 'TruncN',     'func': generate_truncnorm, 'params': {'mean': 0, 'std': 1}},
]

# ============================================================================
# Bit-flip injection
# ============================================================================

def flip_bit(val: torch.Tensor, bit_pos: int):
    if not torch.isfinite(val):
        return val, False
    arr = np.array([val.item()], dtype=np.float32)
    int_arr = arr.view(np.uint32)
    int_arr[0] ^= np.uint32(1 << bit_pos)
    flipped_f = arr.view(np.float32)[0]
    flipped_val = torch.tensor(flipped_f, dtype=torch.float32)
    if not torch.isfinite(flipped_val):
        return val, False
    return flipped_val, True

# ============================================================================
# Encoding functions
# ============================================================================

def sinh_encoding_weights(N_: int, alpha: float = 0.01):
    j = torch.arange(N_, dtype=torch.float32, device=device)
    return torch.sinh(alpha * (j - N_ / 2)) / 2

def linear_encoding_weights(N_: int):
    return torch.arange(1, N_ + 1, dtype=torch.float32, device=device) * 0.003

LINEAR_SCALE = 0.003

# ============================================================================
# Column localization
# ============================================================================

def locate_column_sinh(D1, D2, N_: int, alpha: float = 0.01):
    scale = max(torch.abs(D1).item(), torch.abs(D2).item())
    if scale == 0:
        return None
    ratio = (D2 / scale) / (D1 / scale)
    if not torch.isfinite(ratio):
        return None
    col = torch.asinh(2 * ratio) / alpha + N_ / 2
    if not torch.isfinite(col):
        return None
    return max(0, min(int(torch.round(col).item()), N_ - 1))

def locate_column_linear(D1, D2, N_: int):
    if not torch.isfinite(D2):
        return None
    scale = max(torch.abs(D1).item(), torch.abs(D2).item())
    if scale == 0:
        return None
    ratio = (D2 / scale) / (D1 / scale)
    if not torch.isfinite(ratio):
        return None
    col = ratio / LINEAR_SCALE - 1
    if not torch.isfinite(col):
        return None
    return max(0, min(int(torch.round(col).item()), N_ - 1))

# ============================================================================
# V-ABFT threshold (same as original)
# ============================================================================

def vabft_threshold(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    K_ = A.shape[-1]
    N_ = B.shape[-1]
    e = 2e-6 * math.sqrt(K_ / 1024)

    A_f = A.float()
    B_f = B.float()

    mu_a = A_f.mean(dim=-1)
    mu_b = B_f.mean(dim=-1)
    a_max = A_f.max(dim=-1).values
    a_min = A_f.min(dim=-1).values
    b_max = B_f.max(dim=-1).values
    b_min = B_f.min(dim=-1).values
    sigma_a2 = (a_max - mu_a) * (mu_a - a_min)
    sigma_b2 = (b_max - mu_b) * (mu_b - b_min)

    mu_a_abs = mu_a.abs()
    mu_b_abs = mu_b.abs()
    sqrt_n = math.sqrt(N_)

    sum_mu_b = mu_b_abs.sum()
    sum_mu_b2 = (mu_b_abs ** 2).sum()
    sum_sigma_b2 = sigma_b2.sum()

    bound = e * (
        N_ * mu_a_abs * sum_mu_b
        + 2.5 * N_ * torch.sqrt(mu_a_abs ** 2 * sum_sigma_b2 / N_ + sigma_a2 * sum_mu_b2)
        + 2.5 * sqrt_n * sigma_a2.sqrt() * sum_sigma_b2.sqrt()
    )
    return bound

# ============================================================================
# A-ABFT threshold - 原论文单内积公式 (替换 4-component 版本)
# ============================================================================

def aabft_threshold_original(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """原论文 A-ABFT 门限: per-row, 单内积公式.

    σ_m = √((K(K+1)(K+0.5)+2K)/24) × 2^(-t) × y_m
    y_m = max_k |A[m,k] · B_rowsum[k]|
    threshold_m = 3σ_m
    """
    K_ = A.shape[-1]
    t = 23  # FP32
    u = 2 ** (-t)

    B_rowsum = torch.sum(B, dim=-1)  # (K,)

    # per-row y: y_m = max_k |A[m,k] * B_rowsum[k]|
    products = torch.abs(A * B_rowsum.unsqueeze(0))  # (M, K)
    y_perrow = torch.max(products, dim=-1).values     # (M,)

    var_coeff = (K_ * (K_ + 1) * (K_ + 0.5) + 2 * K_) / 24
    sigma = math.sqrt(var_coeff) * u * y_perrow
    return 3 * sigma  # (M,)

# ============================================================================
# Delta_min
# ============================================================================

def compute_delta_min(D1_vec: torch.Tensor, N_: int, fw_const: float):
    idx = torch.argmax(torch.abs(D1_vec))
    mask = torch.ones(D1_vec.shape[0], dtype=torch.bool, device=D1_vec.device)
    mask[idx] = False
    sigma_D1 = torch.std(D1_vec[mask]).item()
    return 6 * sigma_D1 * fw_const * N_

# ============================================================================
# Main experiment
# ============================================================================

bf16_bits = list(range(31, 15, -1))

sinh_weights = sinh_encoding_weights(N)
linear_weights = linear_encoding_weights(N)

res_detect_v = {}
res_detect_a = {}
res_corr_v = {}
res_corr_a = {}
res_e2e_v = {}
res_e2e_a = {}

print(f"Config: M={M}, K={K}, N={N}, trials={trials_main}")
print(f"A-ABFT: ORIGINAL single inner-product formula (per-row y)")
print(f"Bits: {bf16_bits[0]} -> {bf16_bits[-1]}")
print(f"Distributions: {[d['name'] for d in all_dists]}")
print(f"{'=' * 70}")

for dist_info in all_dists:
    dname = dist_info['name']
    print(f"\n  Distribution: {dname}")

    res_detect_v[dname] = {}
    res_detect_a[dname] = {}
    res_corr_v[dname] = {}
    res_corr_a[dname] = {}
    res_e2e_v[dname] = {}
    res_e2e_a[dname] = {}

    for bit in bf16_bits:
        detected_v = 0
        detected_a = 0
        corrected_v = 0
        corrected_a = 0
        valid_count = 0
        attempts = 0

        while valid_count < trials_main and attempts < trials_main * 5:
            attempts += 1
            try:
                A = dist_info['func'](M, K, **dist_info['params'])
                B = dist_info['func'](K, N, **dist_info['params'])
                C_ref = torch.matmul(A, B)

                ith = torch.randint(0, M, (1,)).item()
                jth = torch.randint(0, N, (1,)).item()

                flipped_val, success = flip_bit(C_ref[ith, jth], bit)
                if not success or not torch.isfinite(flipped_val):
                    continue
                C_ref[ith, jth] = flipped_val
                valid_count += 1

                B_sum = torch.sum(B, dim=1)
                c_check = torch.matmul(A, B_sum)
                c_sum = torch.sum(C_ref, dim=1)
                D1_vec = c_check - c_sum

                D1_abs = torch.abs(D1_vec[ith])

                # V-ABFT Detection
                error_bound_v = vabft_threshold(A, B)
                v_detected = D1_abs > error_bound_v[ith]

                # A-ABFT Detection (原论文公式)
                error_bound_a = aabft_threshold_original(A, B)
                a_detected = D1_abs > error_bound_a[ith]

                D1_err = D1_vec[ith]

                if v_detected:
                    detected_v += 1
                    delta_v = compute_delta_min(D1_vec, N, FW_SINH)
                    if torch.abs(D1_err) > delta_v:
                        B_sinh = torch.matmul(B, sinh_weights)
                        D2_v = (torch.matmul(A, B_sinh) - torch.matmul(C_ref, sinh_weights))[ith]
                        col_v = locate_column_sinh(D1_err, D2_v, N)
                        if col_v == jth:
                            corrected_v += 1

                if a_detected:
                    detected_a += 1
                    delta_a = compute_delta_min(D1_vec, N, FW_LINEAR)
                    if torch.abs(D1_err) > delta_a:
                        B_lin = torch.matmul(B, linear_weights)
                        D2_a = (torch.matmul(A, B_lin) - torch.matmul(C_ref, linear_weights))[ith]
                        col_a = locate_column_linear(D1_err, D2_a, N)
                        if col_a == jth:
                            corrected_a += 1

            except Exception as e:
                continue

        det_v = detected_v / valid_count * 100 if valid_count > 0 else -1
        det_a = detected_a / valid_count * 100 if valid_count > 0 else -1
        cond_v = corrected_v / detected_v * 100 if detected_v > 0 else -1
        cond_a = corrected_a / detected_a * 100 if detected_a > 0 else -1
        e2e_v = corrected_v / valid_count * 100 if valid_count > 0 else -1
        e2e_a = corrected_a / valid_count * 100 if valid_count > 0 else -1

        res_detect_v[dname][str(bit)] = round(det_v, 2) if det_v >= 0 else -1
        res_detect_a[dname][str(bit)] = round(det_a, 2) if det_a >= 0 else -1
        res_corr_v[dname][str(bit)] = round(cond_v, 2) if cond_v >= 0 else -1
        res_corr_a[dname][str(bit)] = round(cond_a, 2) if cond_a >= 0 else -1
        res_e2e_v[dname][str(bit)] = round(e2e_v, 2) if e2e_v >= 0 else -1
        res_e2e_a[dname][str(bit)] = round(e2e_a, 2) if e2e_a >= 0 else -1

        bt = "sign" if bit == 31 else ("exp" if bit >= 23 else "mant")
        print(f"    bit {bit:2d} ({bt:4s}): Vdet={det_v:5.1f}% Adet={det_a:5.1f}%  "
              f"Ve2e={e2e_v:5.1f}%  Ae2e={e2e_a:5.1f}%  "
              f"(n={valid_count}, dv={detected_v}, da={detected_a})")
        sys.stdout.flush()

out = {
    'detect_v': res_detect_v,
    'detect_a': res_detect_a,
    'cond_vabft': res_corr_v,
    'cond_aabft': res_corr_a,
    'e2e_vabft': res_e2e_v,
    'e2e_aabft': res_e2e_a,
}

output_path = 'fp32_e2e_128x2048x256_original_aabft.json'
with open(output_path, 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nResults saved to {output_path}")
