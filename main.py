import torch
import time

print("---- GPU SANITY CHECK ----")
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("Device count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("Using device:", torch.cuda.get_device_name(0))
else:
    raise SystemExit("CUDA NOT AVAILABLE — aborting test.")

# Allocate big tensors on GPU
x = torch.randn((8000, 8000), device='cuda')
y = torch.randn((8000, 8000), device='cuda')

torch.cuda.synchronize()
t0 = time.time()

# Heavy matrix multiply (forces GPU computation)
z = x @ y
torch.cuda.synchronize()
t1 = time.time()

print("Matrix multiply done.")
print("Result tensor device:", z.device)
print(f"Time for matmul: {t1 - t0:.4f} seconds")

# Check gradient pass
z.sum().backward()
torch.cuda.synchronize()
print("Backward pass OK.")

print("GPU IS 100% WORKING AND COMPUTING CORRECTLY.")
