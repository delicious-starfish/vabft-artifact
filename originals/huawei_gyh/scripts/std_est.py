import torch
import torch_npu
import math
import utils
# import matplotlib.pyplot as plt

size = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
rates = []
u = 2**-8  # bfloat16 的单位舍入误差
print("模拟不同规模下的最大值")
for n in size:
    rate = 0
    max_rate = 0
    bound = 0
    for t in range(1):
        
        vec_normal = torch.randn(n, device=utils.device_npu, dtype=torch.bfloat16)
        print("size={}, max_val={}".format(n, torch.max(torch.abs(vec_normal))))
print("理论值")
for n in size:
    print("size={}".format(n),torch.sqrt(2*torch.log(torch.tensor(n, dtype=torch.float32))))