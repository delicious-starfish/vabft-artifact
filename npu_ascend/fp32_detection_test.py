#!/usr/bin/env python3
"""FP32 V-ABFT vs A-ABFT End-to-End Test: separate detection thresholds.
V-ABFT uses my_bound_improve_robust (per-row variance-based).
A-ABFT uses aabft_corrected (4-component deterministic bound).
"""

import torch
import torch_npu
import utils
import json
import sys

device = utils.device_npu
dtype = torch.float32
trials_main = 100

FW_LINEAR = 0.577
FW_SINH = 0.354

def sinh_encoding_weights(N, alpha=0.01):
    j = torch.arange(N, dtype=torch.float32, device=device)
    return torch.sinh(alpha * (j - N/2)) / 2

def linear_encoding_weights(N):
    return torch.arange(1, N+1, dtype=torch.float32, device=device) * 0.003

LINEAR_SCALE = 0.003


def locate_column_sinh(D1, D2, N, alpha=0.01):
    scale = max(torch.abs(D1).item(), torch.abs(D2).item())
    if scale == 0:
        return None
    ratio = (D2 / scale) / (D1 / scale)
    if not torch.isfinite(ratio):
        return None
    col = torch.asinh(2 * ratio) / alpha + N/2
    if not torch.isfinite(col):
        return None
    return max(0, min(int(torch.round(col).item()), N-1))

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
    return max(0, min(int(torch.round(col).item()), N-1))

def compute_delta_min(D1_vec, N, fw_const):
    """Compute delta_min: remove largest |D1| (likely error row), then std."""
    idx = torch.argmax(torch.abs(D1_vec))
    mask = torch.ones(D1_vec.shape[0], dtype=torch.bool, device=D1_vec.device)
    mask[idx] = False
    sigma_D1 = torch.std(D1_vec[mask]).item()
    return 6 * sigma_D1 * fw_const * N

configs = {
    "128x1024x256": (128, 1024, 256),
    "128x4096x256": (128, 4096, 256),
}

all_dists = [
    {'name': 'N(1e-6,1)', 'func': utils.generate_matrice, 'params': {'std': 1, 'mean': 1e-6}},
    {'name': 'N(1,1)', 'func': utils.generate_matrice, 'params': {'std': 1, 'mean': 1}},
    {'name': 'U(-1,1)', 'func': utils.generate_matrice_uniform, 'params': {'lower': -1, 'upper': 1}},
    {'name': 'TruncN', 'func': utils.generate_matrice_clamp, 'params': {'std': 1, 'mean': 0}},
]

bf16_bits = list(range(31, 15, -1))

for config_name, (M, K, N) in configs.items():
    # Per-config results: separate detection for V-ABFT and A-ABFT
    res_detect_v = {}
    res_detect_a = {}
    res_corr_v = {}
    res_corr_a = {}
    res_e2e_v = {}
    res_e2e_a = {}

    print(f"\n{'='*60}")
    print(f"Config: {config_name} (M={M}, K={K}, N={N})")
    print(f"{'='*60}")

    sinh_weights = sinh_encoding_weights(N)
    linear_weights = linear_encoding_weights(N)

    for init in all_dists:
        dname = init['name']
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
                    A = init['func'](M, K, device=device, dtype=dtype, **init['params'])
                    B = init['func'](K, N, device=device, dtype=dtype, **init['params'])
                    C_ref = torch.matmul(A, B)

                    ith = torch.randint(0, M, (1,)).item()
                    jth = torch.randint(0, N, (1,)).item()

                    val_cpu = C_ref[ith, jth].cpu()
                    flipped_val, success = utils.flip_infuse(val_cpu, bit)
                    if not success or not torch.isfinite(flipped_val):
                        continue
                    C_ref[ith, jth] = flipped_val.to(device)
                    valid_count += 1

                    # D1 vector (shared computation)
                    B_sum = torch.sum(B, dim=1)
                    c_check = torch.matmul(A, B_sum)
                    c_sum = torch.sum(C_ref, dim=1)
                    D1_vec = c_check - c_sum

                    D1_abs = torch.abs(D1_vec[ith])

                    # === V-ABFT Detection: per-row variance-based bound ===
                    error_bound_v = utils.my_bound_improve_robust(A, B, dtype=dtype)
                    v_detected = D1_abs > error_bound_v[ith]

                    # === A-ABFT Detection: 4-component corrected bound ===
                    _, error_bound_a = utils.aabft_corrected(A, B, C_ref)
                    a_detected = D1_abs > error_bound_a[ith]

                    D1_err = D1_vec[ith]

                    # V-ABFT path: detection → localization → correction
                    if v_detected:
                        detected_v += 1
                        delta_v = compute_delta_min(D1_vec, N, FW_SINH)
                        if torch.abs(D1_err) > delta_v:
                            B_sinh = torch.matmul(B, sinh_weights)
                            D2_v = (torch.matmul(A, B_sinh) - torch.matmul(C_ref, sinh_weights))[ith]
                            col_v = locate_column_sinh(D1_err, D2_v, N)
                            if col_v == jth:
                                corrected_v += 1

                    # A-ABFT path: detection → localization → correction
                    if a_detected:
                        detected_a += 1
                        delta_a = compute_delta_min(D1_vec, N, FW_LINEAR)
                        if torch.abs(D1_err) > delta_a:
                            B_lin = torch.matmul(B, linear_weights)
                            D2_a = (torch.matmul(A, B_lin) - torch.matmul(C_ref, linear_weights))[ith]
                            col_a = locate_column_linear(D1_err, D2_a, N)
                            if col_a == jth:
                                corrected_a += 1

                except Exception:
                    try:
                        torch.npu.synchronize()
                        torch.npu.empty_cache()
                    except:
                        pass
                    continue

            # Compute rates
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
                  f"Vcond={cond_v:5.1f}%|{e2e_v:5.1f}%  Acond={cond_a:5.1f}%|{e2e_a:5.1f}%  "
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
    path = f'/home/gyh/V-ABFT_TEST/fp32_e2e_{config_name}.json'
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved to {path}")
