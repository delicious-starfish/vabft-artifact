import torch
import torch_npu
import math
import utils
# import matplotlib.pyplot as plt

size = [128, 256, 512, 1024, 2048, 4096, 8192]
rates = []
# u = 2**-8  # float32 的单位舍入误差
print(2**-8, 2**-24, 2**-11)
dtypes = [torch.bfloat16, torch.float32, torch.float16]
dtypes = [torch.float32]
for dtype in dtypes:
    with open("output_error_bound.txt", "a") as f:
        f.write(f"\nData type: {dtype}\n")
    for n in size:
        rate = 0
        max_rate = 0
        bound = 0
        for t in range(100000):
            A = utils.generate_matrice(128, n, std=1, mean=1, device=utils.device_npu, dtype=dtype)
            B = utils.generate_matrice(n, 256, std=1, mean=1, device=utils.device_npu, dtype=dtype)
            A = torch.abs(A)
            B = torch.abs(B)
            check1 = torch.matmul(A, torch.matmul(B, torch.ones(256, 1, device=utils.device_npu, dtype=dtype)))[0,0].to("cpu").to(torch.float64)
            # A1_fp32 = A[0].to(torch.float32)
            # B1_fp32 = B[:,0].to(torch.float32)
            sum1 = torch.matmul(torch.matmul(A, B), torch.ones(256, 1, device=utils.device_npu, dtype=dtype))[0,0].to("cpu").to(torch.float64)
            
            # sum2 = torch.dot(A1_fp32, B1_fp32)
            
            # A_cpu_fp64 = A.to('cpu').to(torch.float64)
            # B_cpu_fp64 = B.to('cpu').to(torch.float64)
            # sum2 = torch.matmul(A_cpu_fp64, B_cpu_fp64)[0,0]
            # sum1 = sum1.to("cpu").to(torch.float64)

            error_rate = torch.abs(check1 - sum1) / torch.abs(sum1)
            # rate += error_rate.item()
            max_rate = max(max_rate, error_rate.item())
            # bound += torch.abs(sum2-sum1)
        # rates.append(rate / 100000)
        
        # print(f"Size: {n}, Average Relative Error: {rate/1000:.6e}")
        print(f"Size: {n}, Max Relative Error: {max_rate:.6e}")
        with open("output_error_bound.txt", "a") as f:
            f.write(f"Array size: {n}, Max Relative Error: {max_rate:.6e}\n")


