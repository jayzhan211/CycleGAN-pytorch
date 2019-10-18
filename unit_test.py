import torch
import torch.nn as nn
import torch.nn.functional as F
print(torch.__version__)

x = torch.randn(2, 2, 2)
z = F.adaptive_avg_pool2d(x, 1)
print(z.size())
# num_features = 4
# x = nn.Parameter(torch.full((1, num_features, 1, 1), 0.9))
# sz = x.size()
# print()
# b, c, h, w = x.size()
# print(x.expand(4, c, h, w))
# z = torch.ones(4, c, h, w) - x.expand(4, c, h, w)
# print(z)
