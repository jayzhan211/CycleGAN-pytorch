import torch
import torch.nn as nn
print(torch.__version__)

x = torch.randn(1, 2, 1, 1)
print(x)
z = (1 - x)
print(z)
