import os
import torch
import torch_npu
import math
import numpy as np
from datetime import datetime

device_npu = torch.device('npu:0')
print(f'Using device: {device_npu}', flush=True)

def estimate_emax_relative(M, K, N, dtype, num_trials=100000):
    max_relative_error = 0.0
    for _ in range(num_trials):
        A = torch.abs(torch.randn(M, K, device=device_npu, dtype=dtype) + 1.0)
        B = torch.abs(torch.randn(K, N, device=device_npu, dtype=dtype) + 1.0)
        C = torch.matmul(A, B)
        ones = torch.ones(N, 1, device=device_npu, dtype=dtype)
        path1 = torch.matmul(C, ones).squeeze(-1)
        B_rowsum = torch.matmul(B, ones)
        path2 = torch.matmul(A, B_rowsum).squeeze(-1)
        error = torch.abs(path1 - path2)
        relative_error = (error / torch.abs(path1)).max().item()
        max_relative_error = max(max_relative_error, relative_error)
    return max_relative_error

print('=' * 80, flush=True)
print(f'e_max Scaling Test - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', flush=True)
print('=' * 80, flush=True)

size_choices = [128, 256, 512, 1024, 2048, 4096, 8192]
sizes = [(128, K, N) for K in size_choices for N in size_choices]

dtype = torch.float32
u = 2**-24
num_trials = 100000

print(f'\nTesting {dtype}, unit roundoff u = {u:.2e}', flush=True)
print(f'Number of configurations: {len(sizes)}', flush=True)
print(f'Trials per config: {num_trials}', flush=True)
print(f'{"M":<6} {"K":<6} {"N":<6} {"KN":<12} {"log(KN)":<10} {"sqrt(KN)":<10} {"e_max":<15}', flush=True)
print('-' * 90, flush=True)

results = []
for i, (M, K, N) in enumerate(sizes):
    emax = estimate_emax_relative(M, K, N, dtype, num_trials)
    kn = K * N
    log_kn = math.log(kn)
    sqrt_kn = math.sqrt(kn)
    results.append({'M': M, 'K': K, 'N': N, 'KN': kn, 'log_KN': log_kn, 'sqrt_KN': sqrt_kn, 'emax': emax})
    print(f'{M:<6} {K:<6} {N:<6} {kn:<12} {log_kn:<10.2f} {sqrt_kn:<10.1f} {emax:<15.6e} [{i+1}/{len(sizes)}]', flush=True)

print('\n' + '=' * 80, flush=True)
print('Regression Analysis', flush=True)
print('=' * 80, flush=True)

emax_values = np.array([r['emax'] for r in results])
log_kn_values = np.array([r['log_KN'] for r in results])
sqrt_kn_values = np.array([r['sqrt_KN'] for r in results])

coef_log, intercept_log = np.polyfit(log_kn_values, emax_values, 1)
y_pred_log = coef_log * log_kn_values + intercept_log
r2_log = 1 - np.sum((emax_values - y_pred_log)**2) / np.sum((emax_values - np.mean(emax_values))**2)

coef_sqrt, intercept_sqrt = np.polyfit(sqrt_kn_values, emax_values, 1)
y_pred_sqrt = coef_sqrt * sqrt_kn_values + intercept_sqrt
r2_sqrt = 1 - np.sum((emax_values - y_pred_sqrt)**2) / np.sum((emax_values - np.mean(emax_values))**2)

print(f'\ne_max = {coef_log:.6e} * log(KN) + {intercept_log:.6e}, R² = {r2_log:.6f}', flush=True)
print(f'e_max = {coef_sqrt:.6e} * sqrt(KN) + {intercept_sqrt:.6e}, R² = {r2_sqrt:.6f}', flush=True)

output_file = os.path.join(os.environ.get('VABFT_ROOT', '.'), 'emax_scaling_results.txt')
with open(output_file, 'w') as f:
    f.write(f'e_max Scaling Test Results - {datetime.now()}\n')
    f.write('=' * 80 + '\n\n')
    f.write(f'e_max = {coef_log:.6e} * log(KN) + {intercept_log:.6e}, R² = {r2_log:.6f}\n')
    f.write(f'e_max = {coef_sqrt:.6e} * sqrt(KN) + {intercept_sqrt:.6e}, R² = {r2_sqrt:.6f}\n\n')
    for r in results:
        f.write(f'{r["M"]} {r["K"]} {r["N"]} {r["KN"]} {r["log_KN"]:.2f} {r["sqrt_KN"]:.1f} {r["emax"]:.6e}\n')

print(f'\nResults saved to {output_file}', flush=True)
