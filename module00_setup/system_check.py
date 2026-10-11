import platform
import sys

import torch


print("Python:", sys.version)
print("Platform:", platform.platform())
print("PyTorch:", torch.__version__)

print("MPS built:", torch.backends.mps.is_built())
print("MPS available:", torch.backends.mps.is_available())

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Selected device:", device)

x = torch.randn(1000, 1000, device=device)
y = torch.randn(1000, 1000, device=device)

z = x @ y

print("Matrix shape:", z.shape)
print("Tensor device:", z.device)