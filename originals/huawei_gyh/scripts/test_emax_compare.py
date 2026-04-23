"""
对比 NPU 脚本逻辑 vs GPU 脚本逻辑在同一个 GPU 上跑，确认差异来源
"""
import torch
import numpy as np

device = torch.device('cuda')
dtype = torch.bfloat16
M, K, N = 256, 1024, 256
num_trials = 10000

def method_npu_style():
    """NPU 脚本的逻辑（在 GPU 上跑）"""
    emax_values = []
    for _ in range(num_trials):
        A = torch.abs(torch.randn(M, K, dtype=torch.float32)) + 0.5
        B = torch.abs(torch.randn(K, N, dtype=torch.float32)) + 0.5

        A_dev = A.to(dtype).to(device)
        B_dev = B.to(dtype).to(device)
        C_dev = torch.matmul(A_dev, B_dev)

        b_rowsum = torch.sum(B, dim=-1, keepdim=True)
        checksum = torch.matmul(A, b_rowsum).squeeze(-1)
        c_rowsum = torch.sum(C_dev.cpu().to(torch.float32), dim=-1)
        actual_diff = torch.abs(checksum - c_rowsum)
        emax = actual_diff / (torch.abs(checksum) + 1e-30)
        emax_values.extend(emax.tolist())
    return np.max(emax_values), np.mean(emax_values)

def method_gpu_style():
    """GPU 脚本的逻辑"""
    emax_values = []
    for _ in range(num_trials):
        A_f32 = torch.abs(torch.randn(M, K, dtype=torch.float32, device=device)) + 0.5
        B_f32 = torch.abs(torch.randn(K, N, dtype=torch.float32, device=device)) + 0.5

        A_lp = A_f32.to(dtype)
        B_lp = B_f32.to(dtype)
        C = torch.matmul(A_lp, B_lp).to(torch.float32)

        B_rowsum = torch.sum(B_f32, dim=1)
        checksum = torch.matmul(A_f32, B_rowsum)
        C_rowsum = torch.sum(C, dim=1)

        diff = torch.abs(checksum - C_rowsum)
        emax = diff / (torch.abs(checksum) + 1e-30)
        emax_values.extend(emax.cpu().tolist())
    return np.max(emax_values), np.mean(emax_values)

print(f"Config: M={M}, K={K}, N={N}, dtype={dtype}, trials={num_trials}")
print(f"u_BF16 = {2**(-8):.4e}\n")

print("Running NPU-style logic on GPU...")
npu_max, npu_mean = method_npu_style()
print(f"  NPU-style:  max={npu_max:.4e}, max/u={npu_max/2**(-8):.2f}, mean={npu_mean:.4e}")

print("Running GPU-style logic on GPU...")
gpu_max, gpu_mean = method_gpu_style()
print(f"  GPU-style:  max={gpu_max:.4e}, max/u={gpu_max/2**(-8):.2f}, mean={gpu_mean:.4e}")

print(f"\nRatio (NPU-style / GPU-style): {npu_max/gpu_max:.2f}x")
