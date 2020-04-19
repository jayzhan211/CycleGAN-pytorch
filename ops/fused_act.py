from torch.utils.cpp_extension import load
import torch.nn as nn
import torch
from torch.autograd import Function
import os

module_path = os.path.dirname(__file__)
fused = load(
    'fused',
    sources=[
        os.path.join(module_path, 'fused_bias_act.cpp'),
        os.path.join(module_path, 'fused_bias_act_kernel.cu'),
    ],
)


class FusedLeakyReLUFunction(Function):
    @staticmethod
    def forward(ctx, input, bias, negative_slope, scale):
        empty = input.new_empty(0)


class FusedLeakyReLU(nn.Module):
    def __init__(self, channel, negative_slope=0.2, scale=2 ** 0.5):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(channel))
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input):
        return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)

def fused_leaky_relu(input, bias, negative_slope=0.2, scale=2 ** 0.5):
    return FusedLeakyReLUFunction.apply(input, bias, negative_slope, scale)



