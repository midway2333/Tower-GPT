import torch

if torch.cpu.is_available():
    print("CPU 可用")
else:
    print("CPU 不可用")

if torch.cuda.is_available():
    print("CUDA 可用")

if torch.xpu.is_available():
    print("XPU 可用")

if torch.backends.mps.is_available():
    print("MPS 可用")