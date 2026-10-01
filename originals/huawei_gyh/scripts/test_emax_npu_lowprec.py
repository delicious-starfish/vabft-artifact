"""NPU BF16/FP16 e_max measurement (Appendix B Table 2 — NPU section).

Measures e_max = max|checksum - rowsum| / |checksum| for BF16/FP16 matrices
on Ascend 910B, sweeping K. Both checksum and rowsum computed in low precision
(offline detection). Verifies the constant-K behavior claimed by the paper:
  BF16: e_max ~ 8e-3
  FP16: e_max ~ 1e-3

Distribution: torch.randn(...).abs() + 0.5  (matches the GPU original protocol)
"""
import os
import math
import json
import argparse
from datetime import datetime

import torch
import torch_npu

device = torch.device('npu:0' if torch_npu.npu.is_available() else 'cpu')


def measure_emax(M, K, N, dtype, num_trials=2000, batch_size=50):
    emax_values = []
    nan_skip = 0
    inf_skip = 0
    ones = torch.ones(N, 1, device=device, dtype=dtype)

    for start in range(0, num_trials, batch_size):
        bs = min(batch_size, num_trials - start)
        A_f32 = torch.abs(torch.randn(bs, M, K, dtype=torch.float32)) + 0.5
        B_f32 = torch.abs(torch.randn(bs, K, N, dtype=torch.float32)) + 0.5

        if dtype == torch.float16:
            scale = min(1.0, (65504 / (K * N * 4)) ** 0.5)
            A_f32 = A_f32 * scale
            B_f32 = B_f32 * scale

        A_lp = A_f32.to(dtype).to(device)
        B_lp = B_f32.to(dtype).to(device)

        # checksum = A @ (B @ ones)
        checksum = torch.matmul(A_lp, torch.matmul(B_lp, ones)).squeeze(-1)
        # rowsum   = (A @ B) @ ones
        C = torch.matmul(A_lp, B_lp)
        rowsum = torch.matmul(C, ones).squeeze(-1)

        if torch.isnan(checksum).any() or torch.isinf(checksum).any():
            inf_skip += bs
            continue

        diff = torch.abs(checksum.to(torch.float32) - rowsum.to(torch.float32))
        emax = diff / (torch.abs(checksum.to(torch.float32)) + 1e-30)
        emax_values.extend(emax.cpu().flatten().tolist())

    if not emax_values:
        return None
    import numpy as np
    arr = np.array(emax_values)
    arr = arr[~(np.isnan(arr) | np.isinf(arr))]
    return {
        'max': float(arr.max()),
        'p999': float(np.percentile(arr, 99.9)),
        'mean': float(arr.mean()),
        'inf_skip': inf_skip,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dtype', choices=['bfloat16', 'float16'], default='bfloat16')
    parser.add_argument('--trials', type=int, default=2000)
    parser.add_argument('--K', type=str, default='256,512,1024,2048,4096,8192,16384')
    parser.add_argument('--M', type=int, default=1024)
    parser.add_argument('--N', type=int, default=256)
    parser.add_argument('--out', default=os.path.join(os.environ.get('VABFT_ROOT', '.'), 'results_phase2', 'B4_emax_npu_lowprec.json'))
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float16
    u = 2 ** -8 if dtype == torch.bfloat16 else 2 ** -11
    Ks = [int(x) for x in args.K.split(',')]

    print(f'[{datetime.now()}] NPU low-prec e_max measurement')
    print(f'Device: {device}  Dtype: {dtype}  u={u:.4e}')
    print(f'Trials/K: {args.trials}  M={args.M}  N={args.N}')
    print('=' * 90)
    print(f'{"K":<8} {"max(e_max)":<14} {"max/u":<10} {"p99.9":<14} {"p99.9/u":<10} {"mean":<14} {"inf_skip":<10}')
    print('-' * 90)

    results = []
    for K in Ks:
        r = measure_emax(args.M, K, args.N, dtype, args.trials)
        if r is None:
            print(f'{K:<8} [all trials skipped]')
            continue
        print(f'{K:<8} {r["max"]:<14.4e} {r["max"]/u:<10.2f} {r["p999"]:<14.4e} {r["p999"]/u:<10.2f} {r["mean"]:<14.4e} {r["inf_skip"]:<10}')
        results.append({'K': K, **r, 'max_over_u': r['max'] / u})

    print('=' * 90)
    if results:
        max_over_u = [r['max_over_u'] for r in results]
        cv = (sum((x - sum(max_over_u)/len(max_over_u))**2 for x in max_over_u) / len(max_over_u))**0.5 / (sum(max_over_u)/len(max_over_u)) * 100
        print(f'CV(max/u) = {cv:.1f}%   {"≈ constant" if cv < 30 else "varies with K"}')
        print(f'Mean max/u = {sum(max_over_u)/len(max_over_u):.2f}u')
        print(f'Max max/u  = {max(max_over_u):.2f}u')

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump({'dtype': str(dtype), 'u': u, 'results': results}, f, indent=2)
    print(f'\nSaved: {args.out}')


if __name__ == '__main__':
    main()
