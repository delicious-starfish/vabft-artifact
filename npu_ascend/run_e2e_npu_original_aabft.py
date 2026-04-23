#!/usr/bin/env python3
"""NPU E2E Test - A-ABFT 使用原论文单内积公式

基于 fp32_detection_test.py，仅修改 A-ABFT 门限为原论文公式:
  σ = √((K(K+1)(K+0.5)+2K)/24) × 2^(-t) × y_m
  y_m = max_k |A[m,k] · B_rowsum[k]|
  threshold = 3σ
"""

import torch
import torch_npu
import utils
import json
import sys
import math
import numpy as np

device = utils.device_npu
dtype = torch.float32
trials_main = 100

FW_LINEAR = 0.577
FW_SINH = 0.354


def sinh_encoding_weights(N, alpha=0.01):
    j = torch.arange(N, dtype=torch.float32, device=device)
    return torch.sinh(alpha * (j - N / 2)) / 2


def linear_encoding_weights(N):
    return torch.arange(1, N + 1, dtype=torch.float32, device=device) * 0.003


LINEAR_SCALE = 0.003


def locate_column_sinh(D1, D2, N, alpha=0.01):
    scale = max(torch.abs(D1).item(), torch.abs(D2).item())
    if scale == 0:
        return None
    ratio = (D2 / scale) / (D1 / scale)
    if not torch.isfinite(ratio):
        return None
    col = torch.asinh(2 * ratio) / alpha + N / 2
    if not torch.isfinite(col):
        return None
    return max(0, min(int(torch.round(col).item()), N - 1))


def locate_column_linear(D1, D2, N):
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
    return max(0, min(int(torch.round(col).item()), N - 1))


def compute_delta_min(D1_vec, N, fw_const):
    idx = torch.argmax(torch.abs(D1_vec))
    mask = torch.ones(D1_vec.shape[0], dtype=torch.bool, device=D1_vec.device)
    mask[idx] = False
    sigma_D1 = torch.std(D1_vec[mask]).item()
    return 6 * sigma_D1 * fw_const * N


def aabft_threshold_original(A, B):
    """原论文 A-ABFT: per-row 单内积公式."""
    K_ = A.shape[-1]
    t = 23
    u = 2 ** (-t)
    B_rowsum = torch.sum(B, dim=-1)
    products = torch.abs(A * B_rowsum.unsqueeze(0))
    y_perrow = torch.max(products, dim=-1).values
    var_coeff = (K_ * (K_ + 1) * (K_ + 0.5) + 2 * K_) / 24
    sigma = math.sqrt(var_coeff) * u * y_perrow
    return 3 * sigma


def flip_bit_npu(val, bit_pos):
    """Flip bit on NPU tensor."""
    val_cpu = val.cpu()
    arr = np.array([val_cpu.item()], dtype=np.float32)
    int_arr = arr.view(np.uint32)
    int_arr[0] ^= np.uint32(1 << bit_pos)
    flipped_f = arr.view(np.float32)[0]
    flipped_val = torch.tensor(flipped_f, dtype=torch.float32, device=device)
    if not torch.isfinite(flipped_val):
        return val, False
    return flipped_val, True


configs = {
    "256x4096x256": (256, 4096, 256),
    "256x8192x256": (256, 8192, 256),
}

all_dists = [
    {'name': 'N(1e-6,1)', 'func': utils.generate_matrice, 'params': {'std': 1, 'mean': 1e-6}},
    {'name': 'N(1,1)', 'func': utils.generate_matrice, 'params': {'std': 1, 'mean': 1}},
    {'name': 'U(-1,1)', 'func': utils.generate_matrice_uniform, 'params': {'lower': -1, 'upper': 1}},
    {'name': 'TruncN', 'func': utils.generate_matrice_clamp, 'params': {'std': 1, 'mean': 0}},
]

bf16_bits = list(range(31, 15, -1))

for config_name, (M, K, N) in configs.items():
    res_detect_v = {}
    res_detect_a = {}
    res_corr_v = {}
    res_corr_a = {}
    res_e2e_v = {}
    res_e2e_a = {}

    sinh_w = sinh_encoding_weights(N)
    linear_w = linear_encoding_weights(N)

    print(f"\n{'=' * 70}")
    print(f"Config: {config_name}, Device: {device}")
    print(f"A-ABFT: ORIGINAL single inner-product formula (per-row y)")
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
                    A = dist_info['func'](M, K, **dist_info['params'],
                                          device=device, dtype=dtype)
                    B = dist_info['func'](K, N, **dist_info['params'],
                                          device=device, dtype=dtype)
                    C_ref = torch.matmul(A, B)

                    ith = torch.randint(0, M, (1,)).item()
                    jth = torch.randint(0, N, (1,)).item()

                    flipped_val, success = flip_bit_npu(C_ref[ith, jth], bit)
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
                    error_bound_v = utils.my_bound_improve_robust(A, B, dtype=torch.float32)
                    v_detected = D1_abs > error_bound_v[ith]

                    # A-ABFT Detection (原论文公式)
                    error_bound_a = aabft_threshold_original(A, B)
                    a_detected = D1_abs > error_bound_a[ith]

                    D1_err = D1_vec[ith]

                    if v_detected:
                        detected_v += 1
                        delta_v = compute_delta_min(D1_vec, N, FW_SINH)
                        if torch.abs(D1_err) > delta_v:
                            B_sinh = torch.matmul(B, sinh_w)
                            D2_v = (torch.matmul(A, B_sinh) -
                                    torch.matmul(C_ref, sinh_w))[ith]
                            col_v = locate_column_sinh(D1_err, D2_v, N)
                            if col_v == jth:
                                corrected_v += 1

                    if a_detected:
                        detected_a += 1
                        delta_a = compute_delta_min(D1_vec, N, FW_LINEAR)
                        if torch.abs(D1_err) > delta_a:
                            B_lin = torch.matmul(B, linear_w)
                            D2_a = (torch.matmul(A, B_lin) -
                                    torch.matmul(C_ref, linear_w))[ith]
                            col_a = locate_column_linear(D1_err, D2_a, N)
                            if col_a == jth:
                                corrected_a += 1

                except Exception as e:
                    continue

            det_v = detected_v / valid_count * 100 if valid_count > 0 else -1
            det_a = detected_a / valid_count * 100 if valid_count > 0 else -1
            e2e_v = corrected_v / valid_count * 100 if valid_count > 0 else -1
            e2e_a = corrected_a / valid_count * 100 if valid_count > 0 else -1

            res_detect_v[dname][str(bit)] = round(det_v, 2) if det_v >= 0 else -1
            res_detect_a[dname][str(bit)] = round(det_a, 2) if det_a >= 0 else -1
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
        'e2e_vabft': res_e2e_v,
        'e2e_aabft': res_e2e_a,
    }

    output_path = f'fp32_e2e_{config_name}_original_aabft.json'
    with open(output_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {output_path}")
