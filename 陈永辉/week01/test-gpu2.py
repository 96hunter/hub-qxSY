import torch
import torch_directml
import time

device = torch_directml.device()

a = torch.randn(8192, 8192).to(device)
b = torch.randn(8192, 8192).to(device)

start = time.perf_counter()

for i in range(30):
    c = torch.matmul(a, b)

# 将结果转回 CPU，确保前面的 GPU 计算已经完成
c_cpu = c.cpu()

end = time.perf_counter()

print("设备:", c.device)
print("计算耗时:", round(end - start, 3), "秒")