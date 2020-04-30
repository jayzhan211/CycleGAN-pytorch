import torch.nn as nn
import numpy as np
import torch
from torch.nn import functional as F


class Linear(nn.Module):
    def __init__(self, input_nc, output_nc, bias=True, gain=np.sqrt(2.0), lrmul=1.0, implicit_lreq=True):
        super(Linear, self).__init__()
        self.weight = nn.Parameter(torch.randn(output_nc, input_nc), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_nc), requires_grad=True)
        else:
            self.bias = None

        std = gain / np.sqrt(input_nc) * lrmul

        if implicit_lreq:
            nn.init.normal_(self.weight, mean=0, std=std / lrmul)
            self.weight.lr_equalization_coef = std
            if bias:
                self.bias.lr_equalization_coef = lrmul
        else:
            nn.init.normal_(self.weight, mean=0, std=1.0 / lrmul)

        self.implicit_lreq = implicit_lreq
        self.std = std
        self.lrmul = lrmul

    def forward(self, x):
        if self.implicit_lreq:
            return F.linear(x, self.weight, self.bias)
        else:
            if self.bias is None:
                return F.linear(x, self.weight * self.std)
            else:
                return F.linear(x, self.weight * self.std, self.bias * self.lrmul)

