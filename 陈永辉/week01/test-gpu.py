import torch
import torch_directml

device = torch_directml.device()

print("PyTorch:", torch.__version__)
print("GPU device:", device)

a = torch.randn(1024, 1024).to(device)
b = torch.randn(1024, 1024).to(device)

c = torch.matmul(a, b)

print("计算成功")
print("结果设备:", c.device)
print("结果形状:", c.shape)