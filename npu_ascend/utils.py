import torch
import torch_npu
import math
from tqdm import tqdm

device_npu = torch.device("npu:0" if torch_npu.npu.is_available() else "cpu")

def checksum_bound_high_precision(a, b, c):
    t = 23
    b1 = torch.sum(b, dim=-1, keepdim=True, dtype=torch.float32)
    c1= torch.matmul(a.to(torch.float32), b1)
    c1_trans = c1.squeeze()
    
    n_b= b.shape[-1]
    m_b = b.shape[0]
    n = c.shape[-1]
    
    c_max, _ = torch.max(torch.abs(c), dim=-1)
    c_sum_accum_error = math.sqrt(n*(n+1)*(2*n+1)/48)*c_max*2**(-t)
    c_ele_round_error_accum = c_max*2**(-8)*math.sqrt(n_b)
    
    b_max,_= torch.max(torch.abs(b), dim=-1, keepdim=True)
    delta_1 = math.sqrt(n_b*(n_b+1)*(2*n_b+1)/48)*b_max*2**(-t)
    delta_4 = torch.matmul(torch.abs(a), delta_1).squeeze()
    a_max,_= torch.max(torch.abs(a),dim=-1)
    delta_2_3 = math.sqrt((m_b*(m_b+1)*(m_b+0.5)+2*m_b)/24)*a_max*torch.max(b_max.squeeze())*2**(-t)
    error_total =(c_sum_accum_error + c_ele_round_error_accum + delta_2_3.squeeze()+delta_4).to(torch.float)
    return c1_trans, error_total

def my_bound(a, b):
    e = 1e-2
    k = a.shape[-1]
    n = b.shape[-1]
    mu_a = torch.mean(a, dim=-1)
    mu_b = torch.mean(b)
    a_max = torch.max(a, dim=-1).values
    b_max = torch.max(b)
    sigma_a = (a_max - mu_a)/torch.sqrt(torch.tensor(2*math.log(k))) * 1.1
    sigma_b = (b_max - mu_b)/torch.sqrt(torch.tensor(2*math.log(k*n))) * 1.1
    
    mu_a = torch.abs(mu_a)
    mu_b = torch.abs(mu_b)
    
    sqrt_k = torch.sqrt(torch.tensor(k, dtype=torch.float32))
    sqrt_n = torch.sqrt(torch.tensor(n, dtype=torch.float32))
    
    bound = e * (
        k * n * mu_a * mu_b 
        + 4 * torch.sqrt(k * n * mu_a**2 * sigma_b**2 + n**2 * k * sigma_a**2 * mu_b**2)
        + 4 * sqrt_k * sqrt_n * sigma_a * sigma_b
    )
    
    bound = bound.squeeze()
    return bound

def my_bound_improve(a, b):
    e = 1e-2
    k = a.shape[-1]
    n = b.shape[-1]
    mu_a = torch.mean(a, dim=-1)
    mu_b = torch.mean(b, dim=-1)
    a_max = torch.max(a, dim=-1).values
    b_max = torch.max(b, dim=-1).values
    sigma_a = (a_max - mu_a)/torch.sqrt(torch.tensor(2*math.log(k))) * 1.1
    sigma_b = (b_max - mu_b)/torch.sqrt(torch.tensor(2*math.log(k*n))) * 1.1
    
    mu_a = torch.abs(mu_a)
    mu_b = torch.abs(mu_b)
    
    sqrt_n = torch.sqrt(torch.tensor(n, dtype=torch.float32))
    
    sum_mu_b = torch.sum(mu_b)
    sum_mu_b2 = torch.sum(mu_b**2)
    sum_sigma_b2 = torch.sum(sigma_b**2)
    
    bound = e * (
        n * mu_a * sum_mu_b
        + 4 * torch.sqrt(n * mu_a**2 * sum_sigma_b2 + n**2 * sigma_a**2 * sum_mu_b2)
        + 4 * sqrt_n * sigma_a * sum_sigma_b2.sqrt()
    )
    
    bound = bound.squeeze()
    return bound

def my_bound_improve_robust(a, b, dtype=torch.bfloat16):
    us = {torch.bfloat16: 8e-3,
          torch.float16: 1e-3,
          torch.float32: 2e-06}
    e = us[dtype]
    k = a.shape[-1]
    n = b.shape[-1]
    if dtype == torch.float32:
        e *= math.sqrt(k / 1024)
    
    mu_a = torch.mean(a, dim=-1)
    mu_b = torch.mean(b, dim=-1)
    a_max = torch.max(a, dim=-1).values
    b_max = torch.max(b, dim=-1).values
    a_min = torch.min(a, dim=-1).values
    b_min = torch.min(b, dim=-1).values
    sigma_a2 = (a_max - mu_a)*(mu_a - a_min)
    sigma_b2 = (b_max - mu_b)*(mu_b - b_min)

    mu_a = torch.abs(mu_a)
    mu_b = torch.abs(mu_b)
    
    sqrt_n = torch.sqrt(torch.tensor(n, dtype=torch.float32))
    
    sum_mu_b = torch.sum(mu_b)
    sum_mu_b2 = torch.sum(mu_b**2)
    sum_sigma_b2 = torch.sum(sigma_b2)
    
    bound = e * (
        n * mu_a * sum_mu_b
        +  2.5 * n * torch.sqrt(mu_a**2 * sum_sigma_b2 / n + sigma_a2 * sum_mu_b2)
        +  2.5 * sqrt_n * sigma_a2.sqrt() * sum_sigma_b2.sqrt()
    )
    
    bound = bound.squeeze()
    return bound


def my_bound_improve_robust_sampling(a, b, dtype=torch.bfloat16):
    us = {torch.bfloat16: 8e-3,
          torch.float16: 1e-3,
          torch.float32: 2e-6}
    e = us[dtype]
    k = a.shape[-1]
    n = b.shape[-1]
    if dtype == torch.float32:
        e *= math.sqrt(k / 1024)
    maskn = torch.arange(n) % 32 < 32
    maskk = torch.arange(k) % 32 < 4
    mu_a = torch.mean(a[..., maskk], dim=-1)
    mu_b = torch.mean(b[..., maskn], dim=-1)
    a_max = torch.max(a[..., maskk], dim=-1).values
    b_max = torch.max(b[..., maskn], dim=-1).values
    a_min = torch.min(a[..., maskk], dim=-1).values
    b_min = torch.min(b[..., maskn], dim=-1).values
    sigma_a2 = (a_max - mu_a)*(mu_a - a_min)
    sigma_b2 = (b_max - mu_b)*(mu_b - b_min)

    mu_a = torch.abs(mu_a)
    mu_b = torch.abs(mu_b)
    
    sqrt_n = torch.sqrt(torch.tensor(n, dtype=torch.float32))
    
    sum_mu_b = torch.sum(mu_b)
    sum_mu_b2 = torch.sum(mu_b**2)
    sum_sigma_b2 = torch.sum(sigma_b2)
    
    bound = e * (
        n * mu_a * sum_mu_b
        +  3 * n * torch.sqrt(mu_a**2 * sum_sigma_b2 / n + sigma_a2 * sum_mu_b2)
        +  3 * sqrt_n * sigma_a2.sqrt() * sum_sigma_b2.sqrt()
    )
    
    bound = bound.squeeze()
    return bound

def my_bound_improve_robust_sampling2(a, b, dtype=torch.bfloat16):
    us = {torch.bfloat16: 8e-3,
          torch.float16: 1e-3,
          torch.float32: 2e-6}
    e = us[dtype]
    k = a.shape[-1]
    n = b.shape[-1]
    if dtype == torch.float32:
        e *= math.sqrt(k / 1024)
    maskk = torch.arange(k) % 32 < 16
    mu_a = torch.mean(a[..., :k//2], dim=-1)
    mu_b = torch.mean(b[maskk, :], dim=-1)
    a_max = torch.max(a[..., :k//2], dim=-1).values
    b_max = torch.max(b[maskk, :], dim=-1).values
    a_min = torch.min(a[..., :k//2], dim=-1).values
    b_min = torch.min(b[maskk, :], dim=-1).values
    sigma_a2 = (a_max - mu_a)*(mu_a - a_min)
    sigma_b2 = (b_max - mu_b)*(mu_b - b_min)

    mu_a = torch.abs(mu_a)
    mu_b = torch.abs(mu_b)
    
    sqrt_n = torch.sqrt(torch.tensor(n, dtype=torch.float32))
    
    sum_mu_b = torch.sum(mu_b)*2
    sum_mu_b2 = torch.sum(mu_b**2)*2
    sum_sigma_b2 = torch.sum(sigma_b2)*2
    
    bound = e * (
        n * mu_a * sum_mu_b
        +  3 * n * torch.sqrt(mu_a**2 * sum_sigma_b2 / n + sigma_a2 * sum_mu_b2)
        +  3 * sqrt_n * sigma_a2.sqrt() * sum_sigma_b2.sqrt()
    )
    
    bound = bound.squeeze()
    return bound


def generate_matrice(sizem, sizek, std, mean, device, dtype):
    a = torch.randn(sizem, sizek, device=device, dtype=dtype) * std + mean
    return a

def generate_matrice_uniform(sizem, sizek, lower, upper, device, dtype):
    a = torch.rand(sizem, sizek, device=device, dtype=dtype)
    a = a * (upper - lower) + lower
    return a

def generate_matrice_clamp(sizem, sizek, std, mean, device, dtype):
    a = torch.randn(sizem, sizek, device=device, dtype=dtype) * std + mean
    a = torch.clamp(a, min=mean - std, max=mean + std)
    return a

def generate_matrice_almost_Bernoulli(sizem, sizek, std, mean, device, dtype):
    small_std = std / 10
    bias = torch.randn(sizem, sizek, device=device, dtype=dtype) * small_std
    a = (torch.randint(0, 2, (sizem, sizek), device=device, dtype=dtype) * 2 - 1) * std + mean + bias
    return a

def flip_infuse(a, i):
    atype = a.dtype
    if atype == torch.bfloat16 or atype == torch.float16:
        if not 0 <= i <= 15:
            raise ValueError("指数位位置 'i' 必须在 0 到 15 之间。")
    elif atype == torch.float32:
        if not 0 <= i <= 31:
            raise ValueError("指数位位置 'i' 必须在 0 到 31 之间。")
    success = False

    if atype == torch.bfloat16 or atype == torch.float16:
        int_val = a.view(torch.int16)
        #    bfloat16 格式: [符号(1) | 指数(8) | 尾数(7)]
        bit_to_flip = i
        mask = 1 << bit_to_flip
        flipped_int_val = int_val ^ mask
        flipped_float_tensor = flipped_int_val.view(atype)
        if int_val & mask == 0:
            success = True
        return flipped_float_tensor, success
    elif atype == torch.float32:
        int_val = a.view(torch.int32)
        #    float32 格式: [符号(1) | 指数(8) | 尾数(23)]
        bit_to_flip = i
        mask = 1 << bit_to_flip
        flipped_int_val = int_val ^ mask
        flipped_float_tensor = flipped_int_val.view(atype)
        if int_val & mask == 0:
            success = True
        return flipped_float_tensor, success

def FT_matmul(a, b, FT_algorithm=my_bound_improve_robust_sampling2):
    # 将A，B切割为 128*1024，1024*256 的小矩阵进行计算
    # 先将 A，B 补全为块大小的整数倍
    success = True
    a = a.to(torch.bfloat16)
    b = b.to(torch.bfloat16)
    sizem = a.shape[0]
    sizek = a.shape[1]
    sizen = b.shape[1]
    block_m = 128
    block_n = 256
    block_k = 1024
    pad_m = (block_m - sizem % block_m) % block_m
    pad_n = (block_n - sizen % block_n) % block_n
    pad_k = (block_k - sizek % block_k) % block_k
    if pad_m > 0:
        a = torch.cat([a, torch.zeros(pad_m, sizek, device=a.device, dtype=a.dtype)], dim=0)
    if pad_k > 0:
        a = torch.cat([a, torch.zeros(a.shape[0], pad_k, device=a.device, dtype=a.dtype)], dim=1)
        b = torch.cat([b, torch.zeros(pad_k, b.shape[1], device=b.device, dtype=b.dtype)], dim=0)
    if pad_n > 0:
        b = torch.cat([b, torch.zeros(b.shape[0], pad_n, device=b.device, dtype=b.dtype)], dim=1)

    c = torch.zeros(sizem + pad_m, sizen + pad_n, device=a.device, dtype=a.dtype)
    # for i in tqdm(range(0, (sizem + pad_m) // block_m), desc="外层循环"):
    for i in range(0, (sizem + pad_m) // block_m):
        for j in range(0, (sizen + pad_n) // block_n):
            for k in range(0, (sizek + pad_k) // block_k):
                a_block = a[i*block_m:(i+1)*block_m, k:k + block_k]
                b_block = b[k:k + block_k, j*block_n:(j+1)*block_n]
                c_block = torch.matmul(a_block, b_block)
                bound = FT_algorithm(a_block, b_block)
                c_check = torch.matmul(a_block, torch.sum(b_block, dim=-1, keepdim=True)).squeeze()
                c_sum = torch.sum(c_block, dim=-1, keepdim=True)
                diff = torch.abs(c_check - c_sum.squeeze())
                if (diff <= bound).all():
                    c[i*block_m:(i+1)*block_m, j*block_n:(j+1)*block_n] += c_block
                else:
                    # print(i, j, k)
                    # save a_block and b_block for debugging
                    # torch.save(a_block, "/home/gyh/data/a_block_error.pth")
                    # torch.save(b_block, "/home/gyh/data/b_block_error.pth")
                    # raise ValueError("FT_matmul: Error bound exceeded during block multiplication.")
                    error_msg = f"Block fail at i={i}, j={j}, k={k}. Max diff: {diff.max().item()}"
                    success = False
                    return success, error_msg
    return success, "Success"


# =============================================================================
# Localization Encoding (linear vs sinh) — 零中心版本
# =============================================================================

import math


def make_linear_centered(N: int, dtype=torch.float32):
    """零中心 linear 编码: w(j) = j - (N-1)/2, 范围 [-(N-1)/2, (N-1)/2]."""
    return (torch.arange(N, dtype=dtype, device=device_npu) - (N - 1) / 2.0)


def make_sinh_centered(N: int, dtype=torch.float32):
    """零中心 sinh 编码, 与 linear 相同方差.

    w*(j) = scale * sinh(lambda * (j - (N-1)/2))
    其中 lambda = 2.821 / N, scale 使 Var(w_sinh) = Var(w_linear).

    Returns:
        (w_sinh, lambda, scale) 用于解析逆函数定位.
    """
    lam = 2.821 / N
    j_idx = torch.arange(N, dtype=torch.float64)
    w_sinh_raw = torch.sinh(lam * (j_idx - (N - 1) / 2.0))
    w_lin_c = j_idx - (N - 1) / 2.0
    scale = w_lin_c.std().item() / w_sinh_raw.std().item()
    w_sinh = (w_sinh_raw * scale).to(dtype=dtype, device=device_npu)
    return w_sinh, lam, scale


def compute_D1_D2(A, B, C, w):
    """计算 ABFT 检测差异 D1 和定位差异 D2.

    D1 = sum(C, dim=-1) - A @ sum(B, dim=-1)   (不依赖 encoding)
    D2 = C @ w - A @ (B @ w)                     (依赖 encoding)

    Returns:
        (D1, D2): shape 均为 (M,)
    """
    A_f = A.float()
    B_f = B.float()
    C_f = C.float()
    w_f = w.float()

    B_rowsum = B_f.sum(dim=-1, keepdim=True)
    checksum_r1 = (A_f @ B_rowsum).squeeze(-1)
    Bw = B_f @ w_f.unsqueeze(-1)
    checksum_r2 = (A_f @ Bw).squeeze(-1)

    verify_r1 = C_f.sum(dim=-1)
    verify_r2 = C_f @ w_f

    D1 = verify_r1 - checksum_r1
    D2 = verify_r2 - checksum_r2
    return D1, D2


def localize_linear_centered(D1_val: float, D2_val: float, N: int) -> int:
    """零中心 linear 定位: j = round(D2/D1 + (N-1)/2)."""
    if abs(D1_val) < 1e-30:
        return N // 2
    ratio = D2_val / D1_val
    j_est = round(ratio + (N - 1) / 2.0)
    return max(0, min(N - 1, j_est))


def localize_sinh_analytic(D1_val: float, D2_val: float,
                           N: int, lam: float, scale: float) -> int:
    """Sinh 解析逆函数定位: j = round((N-1)/2 + (1/lam) * arcsinh(ratio/scale)).

    使用解析逆而非 argmin, 确保每个位置的 Voronoi cell 宽度恒为 1.
    """
    if abs(D1_val) < 1e-30:
        return N // 2
    ratio = D2_val / D1_val
    j_continuous = (N - 1) / 2.0 + (1.0 / lam) * math.asinh(ratio / scale)
    j_est = round(j_continuous)
    return max(0, min(N - 1, j_est))


def aabft_corrected(a, b, c, use_high_precision_check=True):
    """
    修正版 A-ABFT 4-component 门限计算 (from H100 validated implementation).

    4个误差分量:
    E1: C 行求和累加误差
    E2: C 元素量化误差累积 (修正版: sqrt(N(N+1)(2N+1)/6))
    E3: B 行求和误差传播
    E4: A @ B_rowsum 内积累加误差

    Returns:
        (checksum, threshold): checksum shape (M,), threshold shape (M,)
    """
    dtype = a.dtype

    if dtype == torch.float64:
        t = 53; u_low = 2**(-53)
    elif dtype == torch.float32:
        t = 23; u_low = 2**(-23)
    elif dtype == torch.bfloat16:
        t = 7; u_low = 2**(-8)
    elif dtype == torch.float16:
        t = 10; u_low = 2**(-11)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    u_high = 2**(-t) if use_high_precision_check else u_low

    M, K = a.shape
    K2, N = b.shape
    assert K == K2

    # Compute checksum (in higher precision if available)
    if use_high_precision_check:
        b_rowsum = torch.sum(b.to(torch.float32), dim=-1, keepdim=True)
        checksum = torch.matmul(a.to(torch.float32), b_rowsum).squeeze(-1)
    else:
        b_rowsum = torch.sum(b, dim=-1, keepdim=True)
        checksum = torch.matmul(a, b_rowsum).squeeze(-1)

    c_f = c.to(torch.float32)
    a_f = a.to(torch.float32)
    b_f = b.to(torch.float32)

    c_max, _ = torch.max(torch.abs(c_f), dim=-1)
    b_max, _ = torch.max(torch.abs(b_f), dim=-1, keepdim=True)
    a_max, _ = torch.max(torch.abs(a_f), dim=-1)

    # E1: C row sum accumulation error
    E1 = math.sqrt(N*(N+1)*(2*N+1)/48) * c_max * u_high

    # E2: C element quantization error (corrected: ~N^1.5 scaling)
    E2 = c_max * u_low * math.sqrt(N*(N+1)*(2*N+1)/6)

    # E3: B row sum error propagation
    delta_b = math.sqrt(N*(N+1)*(2*N+1)/48) * b_max * u_high
    E3 = torch.matmul(torch.abs(a_f), delta_b).squeeze(-1)

    # E4: A @ B_rowsum inner product error (y uses B_rowsum, not B element max)
    b_rowsum_f = torch.sum(b_f, dim=-1)  # (K,)
    y_e4 = torch.max(torch.abs(a_f) * torch.abs(b_rowsum_f).unsqueeze(0), dim=-1).values  # per-row
    E4 = math.sqrt((K*(K+1)*(K+0.5)+2*K)/24) * y_e4 * u_high

    threshold = (E1 + E2 + E3 + E4).to(torch.float32)
    return checksum.to(torch.float32), threshold
