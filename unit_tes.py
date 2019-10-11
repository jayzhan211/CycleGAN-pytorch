from collections import OrderedDict
import numpy as np
import torch
from PIL import Image

# dict_A = OrderedDict(
#     dict(
#         a=123,
#         b=456,
#     )
#
# )
# b = dict_A.values()
# for i in b:
#     print(i)
img = torch.rand(4)
print(img.data)
print(img)
img = img.data
print(img.max() <= 1.0)
print(img.min() >= 0.0)
img[0] = 0.9
assert img.max() <= 1.0 and img.min() >= -1.0, 'torch.tensor is out of range [-1.0, 1.0]'

# img = np.random.randint(1000, size=1000) / 999.0 * 2.0 - 1.0

