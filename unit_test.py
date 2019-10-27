import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
print(torch.__version__)

x = nn.Parameter(torch.rand(1, 4, 1, 1))
print(x)
y = torch.ones_like(x) - x
print(y)
z = 1 - x
print(z)
