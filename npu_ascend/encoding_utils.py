"""
encoding 实验相关的工具函数。

提供 linear 和 sinh 两种定位编码的生成、校验和计算、错误定位功能。
"""

import torch
import torch_npu
import math
from typing import Tuple, Optional

device_npu = torch.device("npu:2" if torch_npu.npu.is_available() else "cpu")


# =============================================================================
# Encoding 向量生成
# =============================================================================

_SINH_LAMBDA = None  # populated by make_sinh_encoding for analytic inverse
_SINH_SCALE = None


def make_linear_encoding(N: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Zero-centered linear encoding: w(j) = j - (N-1)/2.

    论文 Sec.4 line 480: "Both encodings are zero-centered with equal variance."
    """
    j_idx = torch.arange(N, dtype=torch.float64)
    w = j_idx - (N - 1) / 2.0
    return w.to(dtype=dtype, device=device_npu)


def make_sinh_encoding(N: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Zero-centered sinh encoding: w*(j) = scale * sinh(lambda * (j - (N-1)/2)).

    lambda * N ≈ 2.821 (sinh(x) = 3x 的正根, Theorem 2)。
    Scale 使 std(w_sinh) == std(w_linear_centered).
    Module-level _SINH_LAMBDA / _SINH_SCALE 给 localize_error_sinh 反演用。
    """
    global _SINH_LAMBDA, _SINH_SCALE
    lam = 2.821 / N
    j_idx = torch.arange(N, dtype=torch.float64)
    w_raw = torch.sinh(lam * (j_idx - (N - 1) / 2.0))

    w_linear_centered = j_idx - (N - 1) / 2.0
    target_std = w_linear_centered.std()
    scale = (target_std / w_raw.std()).item() if w_raw.std() > 1e-15 else 1.0
    w_sinh = w_raw * scale

    _SINH_LAMBDA = lam
    _SINH_SCALE = scale
    return w_sinh.to(dtype=dtype, device=device_npu)


# =============================================================================
# 校验和计算
# =============================================================================

def compute_checksums(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    w: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """计算 ABFT 校验和。

    路径 1 (pre-computed): checksum_r1 = A @ B_rowsum, checksum_r2 = A @ (B @ w)
    路径 2 (post-computed): verify_r1 = C @ ones,      verify_r2 = C @ w

    Args:
        A: (M, K) 输入矩阵
        B: (K, N) 输入矩阵
        C: (M, N) 计算结果（可能含错误）
        w: (N,)   定位编码向量

    Returns:
        (D1, D2): 每行的检测差异和定位差异, shape 均为 (M,)
        D1 = verify_r1 - checksum_r1
        D2 = verify_r2 - checksum_r2
    """
    compute_dtype = torch.float32
    A_f = A.to(compute_dtype)
    B_f = B.to(compute_dtype)
    C_f = C.to(compute_dtype)
    w_f = w.to(compute_dtype)

    # 路径 1: pre-computed checksums
    B_rowsum = B_f.sum(dim=-1, keepdim=True)            # (K, 1)
    checksum_r1 = (A_f @ B_rowsum).squeeze(-1)           # (M,)

    Bw = (B_f @ w_f.unsqueeze(-1))                       # (K, 1)
    checksum_r2 = (A_f @ Bw).squeeze(-1)                 # (M,)

    # 路径 2: post-computed checksums
    verify_r1 = C_f.sum(dim=-1)                           # (M,)
    verify_r2 = (C_f @ w_f)                               # (M,)

    D1 = verify_r1 - checksum_r1
    D2 = verify_r2 - checksum_r2

    return D1, D2


# =============================================================================
# 错误定位
# =============================================================================

def localize_error_linear(
    D1: torch.Tensor,
    D2: torch.Tensor,
    N: int,
) -> torch.Tensor:
    """Zero-centered linear inverse: j = D2/D1 + (N-1)/2."""
    ratio = D2 / (D1 + 1e-30)
    j_est = ratio + (N - 1) / 2.0
    j_est = torch.round(j_est).long()
    j_est = torch.clamp(j_est, 0, N - 1)
    return j_est


def localize_error_sinh(
    D1: torch.Tensor,
    D2: torch.Tensor,
    w_sinh: torch.Tensor,
    N: int,
) -> torch.Tensor:
    """Analytic inverse (论文 Sec.4 line 480):
    j = round((N-1)/2 + arcsinh((D2/D1)/scale) / lambda).
    """
    lam = _SINH_LAMBDA if _SINH_LAMBDA is not None else 2.821 / N
    scale = _SINH_SCALE if _SINH_SCALE is not None else 1.0
    ratio = D2 / (D1 + 1e-30)
    arg = ratio / scale
    arg = torch.clamp(arg.to(torch.float64), -1e30, 1e30)
    j_est = (N - 1) / 2.0 + torch.asinh(arg) / lam
    j_est = torch.round(j_est).long()
    j_est = torch.clamp(j_est, 0, N - 1)
    return j_est


# =============================================================================
# V-ABFT 阈值计算 (复用 my_bound_improve_robust)
# =============================================================================

def vabft_threshold(
    A: torch.Tensor,
    B: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """计算 V-ABFT 检测阈值（与 my_bound_improve_robust 一致）。

    Args:
        A: (M, K)
        B: (K, N)
        dtype: 目标精度

    Returns:
        (M,) 每行的检测阈值
    """
    us = {
        torch.bfloat16: 8e-3,
        torch.float16: 1e-3,
        torch.float32: 2e-6,
    }
    e = us[dtype]
    K = A.shape[-1]
    N = B.shape[-1]
    if dtype == torch.float32:
        e *= torch.sqrt(torch.tensor(K / 1024, dtype=torch.float32))

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
    sqrt_n = math.sqrt(N)

    sum_mu_b = mu_b_abs.sum()
    sum_mu_b2 = (mu_b_abs ** 2).sum()
    sum_sigma_b2 = sigma_b2.sum()

    bound = e * (
        N * mu_a_abs * sum_mu_b
        + 2.5 * N * torch.sqrt(mu_a_abs ** 2 * sum_sigma_b2 / N + sigma_a2 * sum_mu_b2)
        + 2.5 * sqrt_n * sigma_a2.sqrt() * sum_sigma_b2.sqrt()
    )
    return bound


# =============================================================================
# 完整的 ABFT 检测 + 定位 流程
# =============================================================================

def abft_detect_and_localize(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    encoding: str = "linear",
    dtype: torch.dtype = torch.bfloat16,
    w_cache: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """完整的 ABFT 检测 + 定位。

    Args:
        A: (M, K)
        B: (K, N)
        C: (M, N) 可能含错误的计算结果
        encoding: "linear" 或 "sinh"
        dtype: 精度（用于阈值计算）
        w_cache: 预计算的编码向量（可选，避免重复生成）

    Returns:
        detected: (M,) bool, 是否检测到错误
        localized_col: (M,) long, 估计的错误列索引（未检测到的行返回 -1）
        D1: (M,) 检测差异
    """
    N = B.shape[-1]

    # 生成或使用缓存的编码向量
    if w_cache is not None:
        w = w_cache
    elif encoding == "linear":
        w = make_linear_encoding(N)
    elif encoding == "sinh":
        w = make_sinh_encoding(N)
    else:
        raise ValueError(f"Unknown encoding: {encoding}")

    # 计算阈值
    threshold = vabft_threshold(A, B, dtype=dtype)

    # 计算校验和差异
    D1, D2 = compute_checksums(A, B, C, w)

    # 检测
    detected = D1.abs() > threshold

    # 定位 (只对检测到错误的行做)
    localized_col = torch.full((A.shape[0],), -1, dtype=torch.long, device=D1.device)
    if detected.any():
        if encoding == "linear":
            j_est = localize_error_linear(D1, D2, N)
        else:
            j_est = localize_error_sinh(D1, D2, w, N)
        localized_col[detected] = j_est[detected]

    return detected, localized_col, D1


# =============================================================================
# 辅助函数
# =============================================================================

def inject_error(
    C: torch.Tensor,
    row: int,
    col: int,
    magnitude: float,
) -> torch.Tensor:
    """在矩阵 C 的指定位置注入加性错误。

    Args:
        C: (M, N) 矩阵
        row, col: 注入位置
        magnitude: 错误大小（直接加到元素上）

    Returns:
        修改后的矩阵（原地修改）
    """
    C[row, col] = C[row, col] + magnitude
    return C
