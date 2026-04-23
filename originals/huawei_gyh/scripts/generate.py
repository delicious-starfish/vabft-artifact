import torch
import torch_npu
import utils

size = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
max_errors = []
dtype = torch.float32  # 测试的精度类型

output = "output_error_bound({}).txt".format(dtype)

with open(output, "a") as f:
    f.write("dtype: {}, repeat times: 10000\n".format(dtype))

for n in size:
    max_error = 0
    for t in range(100000):  # 每个大小测试1000次
        # 生成随机数组（使用bfloat16精度）
        arr = utils.generate_matrice(1, n, std=1.0, mean=2, 
                                   device=utils.device_npu, dtype=dtype).squeeze()
        
        sign = torch.randint(0, 1, (n,), device=utils.device_npu, dtype=torch.int32) * 2 - 1
        
        arr = arr * sign.to(dtype)
        
        # 在NPU上使用测试精度求和
        sum_fast = torch.sum(arr).to('cpu').to(torch.float64)
        
        # 在CPU上使用FP64计算精确和
        arr_exact = arr.to('cpu').to(torch.float64)
        sum_exact = torch.sum(arr_exact)
        
        # 计算相对误差
        if sum_exact != 0:  # 避免除以零
            rel_error = torch.abs(sum_fast - sum_exact) / torch.abs(sum_exact)
            max_error = max(max_error, rel_error.item())
            
        if max_error > 1e-2: # 如果误差过大，打印出来看看
            torch.set_printoptions(precision=20)
            print(f"arr: {arr}")
            print(f"sum_fast: {sum_fast}, sum_exact: {sum_exact}, rel_error: {rel_error}")
            with open("debug_large_error.txt", "a") as f:
                f.write(f"arr: {arr}\n")
                f.write(f"sum_fast: {sum_fast}, sum_exact: {sum_exact}, rel_error: {rel_error}\n")
            break
    
    max_errors.append(max_error)
    print(f"Array size: {n}, Max Relative Error: {max_error:.6e}")

    with open(output, "a") as f:
        f.write(f"Array size: {n}, Max Relative Error: {max_error:.6e}\n")
