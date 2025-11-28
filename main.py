import torch
import time

print("---- GPU SANITY CHECK (FORWARD + BACKWARD) ----")
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
# print("CUDA version:", torch.version.cuda)
# print("Device:", torch.cuda.get_device_name(0))

# Enable autograd
x = torch.randn((4000, 4000),requires_grad=True)
y = torch.randn((4000, 4000),requires_grad=True)

# torch.cuda.synchronize()
t0 = time.time()

z = x @ y
loss = z.sum()

# Backward pass
loss.backward()

# torch.cuda.synchronize()
t1 = time.time()

print("Forward + Backward on GPU Successful!")
print(f"Total time: {t1 - t0:.4f} seconds")
print("Grad of x:", x.grad.norm().item())
print("GPU + Autograd fully functional.")