import torch
import torch_npu
import numpy as np
import random

global npu_device
npu_device = torch.device("npu:0" if torch_npu.npu.is_available() else "cpu")
cpu_device = torch.device("cpu")

def generate_random_matrix(dtype, size=(16, 16), std = 1.0):
    """生成指定数据类型的随机矩阵"""
    if dtype == torch.float16:
        return torch.randn(size, dtype=torch.float16) * std
    elif dtype == torch.bfloat16:
        return torch.randn(size, dtype=torch.bfloat16) * std + 0.1
    elif dtype == torch.float32:
        return torch.randn(size, dtype=torch.float32) * std
    else:
        raise ValueError("不支持的dtype")
    
def flip_infuse(a, i):
    if a.dtype == torch.bfloat16 or a.dtype == torch.float16:
        if not 0 <= i <= 15:
            raise ValueError("指数位位置 'i' 必须在 0 到 15 之间。")
    elif a.dtype == torch.float32:
        if not 0 <= i <= 31:
            raise ValueError("指数位位置 'i' 必须在 0 到 31 之间。")

    if a.dtype == torch.bfloat16 or a.dtype == torch.float16:
        int_val = a.view(torch.int16)
        #    bfloat16 格式: [符号(1) | 指数(8) | 尾数(7)]
        bit_to_flip = i
        mask = 1 << bit_to_flip
        flipped_int_val = int_val ^ mask
        flipped_float_tensor = flipped_int_val.view(torch.bfloat16)

        return flipped_float_tensor
    elif a.dtype == torch.float32:
        int_val = a.view(torch.int32)
        #    float32 格式: [符号(1) | 指数(8) | 尾数(23)]
        bit_to_flip = i
        mask = 1 << bit_to_flip
        flipped_int_val = int_val ^ mask
        flipped_float_tensor = flipped_int_val.view(torch.float32)

        return flipped_float_tensor

def flip_infuse_random(a):
    atype = a.dtype
    flipped_float_tensor = a.clone()
    eps = 1e-6 if atype == torch.float32 else 1e-3
    while(torch.abs(flipped_float_tensor-a) <= eps*a or torch.abs(flipped_float_tensor-a) <= eps):
        if a.dtype == torch.bfloat16 or a.dtype == torch.float16:
            int_val = a.view(torch.int16)
            bit_to_flip = random.randint(0, 15)
            
        elif a.dtype == torch.float32:
            int_val = a.view(torch.int32)
            bit_to_flip = random.randint(0, 31)

        mask = 1 << bit_to_flip
        flipped_int_val = int_val ^ mask
        flipped_float_tensor = flipped_int_val.view(atype)

    return flipped_float_tensor

def infuse_bit_flip(C, i=None):
    idx = random.randint(0, 255)
    if i is not None:
        i = random.randint(0, 7) 
    flip_infuse(C[idx // 16, idx % 16], i)
    
    return idx
    
def my_matmul(A0, B0, flag=True):
    """
    执行矩阵乘法 A @ B 的自定义实现
    
    Args:
        A0: torch.Tensor (m x k)
        B0: torch.Tensor (k x n)
    
    Returns:
        torch.Tensor (m x n)
    """
    # 克隆输入以避免修改原始数据
    A = A0.clone()
    B = B0.clone()
    
    # 输入验证
    assert A.dim() == 2 and B.dim() == 2, "输入必须是2D矩阵"
    assert A.size(1) == B.size(0), "A的列数必须等于B的行数"
    
    m, k = A.size()
    n = B.size(1)
    
    # 将A，B padding 为 16的倍数
    pad_m = (16 - m % 16) % 16
    pad_n = (16 - n % 16) % 16
    pad_k = (16 - k % 16) % 16
    if pad_m > 0 or pad_n > 0 or pad_k > 0:
        A = torch.nn.functional.pad(A, (0, pad_n, 0, pad_m), mode='constant', value=0)
        B = torch.nn.functional.pad(B, (0, pad_n, 0, pad_k), mode='constant', value=0)
    m, k = A.size()
    n = B.size(1)
    
    blockx = random.randint(0, m // 16 - 1)
    blocky = random.randint(0, n // 16 - 1)
    C = torch.zeros(m, n, dtype=A.dtype, device=A.device)

    if not flag:
        for i in range(m//16):
            for k in range(k//16):
                for j in range(n//16):
                    A_block = A[i*16:(i+1)*16, k*16:(k+1)*16]
                    B_block = B[k*16:(k+1)*16, j*16:(j+1)*16]
                    C[i*16:(i+1)*16, j*16:(j+1)*16] += torch.matmul(A_block, B_block)
        C = C[:m, :n]  # 去除padding部分
        return C
    
    idx = 0
    idy = 0
    
    for i in range(m//16):
        for k in range(k//16):
            for j in range(n//16):
                A_block = A[i*16:(i+1)*16, k*16:(k+1)*16].clone()
                B_block = B[k*16:(k+1)*16, j*16:(j+1)*16].clone()
                if i == blockx and j == blocky:
                    C_block = torch.matmul(A_block, B_block)
                    info = infuse_bit_flip(C_block)
                    C[i*16:(i+1)*16, j*16:(j+1)*16] += C_block
                    idx = info//16 + i*16
                    idy = info%16 + j*16
                else:  
                    C[i*16:(i+1)*16, j*16:(j+1)*16] += torch.matmul(A_block, B_block)
    C = C[:m, :n]  # 去除padding部分
    return C, idx, idy



def myisclose(x1, x2, atol, rtol):
    return torch.isclose(x1, x2, atol=atol, rtol=rtol).sum() == x1.numel()


def simulate_FA(m, k, n, flip_idx, flag=True, atol=1e-2, rtol=1e-3, dtype=torch.bfloat16):
    """
    模拟BF16矩阵乘法中的错误注入实验
    参数:
        n: 矩阵维度
        flip_idx: 要翻转的指数位（0-7）
        delta: 允许的误差阈值
    返回:
        bool: 是否所有误差都在允许范围内
    """
    # 初始化随机矩阵（BF16类型）
    A = torch.randn((m, k), device=npu_device, dtype=dtype) + 1e-6
    B = torch.randn((k, n), device=npu_device, dtype=dtype) + 1e-6
    e = torch.ones((n, 1), device=npu_device, dtype=dtype)
    et = torch.ones((1, m), device=npu_device, dtype=dtype)

    # 计算矩阵乘积
    C = torch.matmul(A, B)

    # 随机选择一个位置并注入错误
    i, j = np.random.randint(0, n, size=2)
    original_val = C[i, j].clone()
    corrupted_val = flip_infuse_random(original_val)
    if flag:
        C[i, j] = corrupted_val

    # 计算各种矩阵乘积
    C_e = torch.matmul(C, e)
    et_C = torch.matmul(et, C)
    ABe = torch.matmul(A, torch.matmul(B, e))
    etAB = torch.matmul(torch.matmul(et, A), B)

    x1 = torch.cat([C_e.flatten(), et_C.flatten()])
    x2 = torch.cat([ABe.flatten(), etAB.flatten()])
    return [x1, x2]
    # return myisclose(x1, x2, atol, rtol)

def test(m=16, k=16, n=16, atol=1e-2, rtol=1e-3, flip_idx=None, dtype=torch.bfloat16, checksums_infuse=None, checksums_normal=None):
    beta = 16
    while True:
        num = 10000
        cnt1 = 0
        cnt2 = 0
        if checksums_infuse is None:
            checksums_infuse = []
            for i in range(num):
                checksums_infuse.append(simulate_FA(m, k, n, flip_idx, atol=atol, rtol=rtol, dtype=dtype))
        if checksums_normal is None:
            checksums_normal = []
            for i in range(num):
                checksums_normal.append(simulate_FA(m, k, n, flip_idx, flag=False, atol=atol, rtol=rtol, dtype=dtype))
        for i in range(num):
            x1, x2 = checksums_infuse[i]
            y1, y2 = checksums_normal[i]
            if myisclose(x1, x2, atol=atol, rtol=rtol) == False:
                cnt1 += 1
            if myisclose(y1, y2, atol=atol, rtol=rtol) == False:
                cnt2 += 1
        
        true_positive_rate = 100 * cnt1 / num
        false_positive_rate = 100 * cnt2 / num
        # print(f"atal {atol}, rtol {rtol}, flip_idx {flip_idx}")
        # print(f"检查率: {true_positive_rate:.2f}%")
        # print(f"误检率: {false_positive_rate:.2f}%")
        if false_positive_rate > 0.5:
            rtol *= beta
        else:
            if np.sqrt(beta) < 1.1:
                break
            rtol /= beta
            beta = np.sqrt(beta)
            rtol *= beta
        if rtol > 1000:
            break
    return atol, rtol, true_positive_rate, false_positive_rate

atols = [0.1,0.2,0.4,0.8,2,5,10]
infos = []
# types = [torch.float16]
types = [torch.bfloat16]
for atype in types:
    infos = []
    if atype == torch.float32:
        atols = [atol/10000 for atol in atols]
    print("测试类型：", atype)
    for atol in atols:
        infos.append(test(64, 64, 64, atol=atol, rtol=1e-5, flip_idx=5, dtype=atype))

    print("测试结果:")
    for i, (atol,rtol, true_positive_rate, false_positive_rate) in enumerate(infos):
        print(f"atol: {atol:.2e}, rtol: {rtol:.2e}, 检查率: {true_positive_rate:.2f}%, 误检率: {false_positive_rate:.2f}%")
print("测试完成")