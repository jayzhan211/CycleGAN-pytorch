import torch.nn as nn
import numpy as np
import torch
from torch.nn import functional as F


class LREQLinear(nn.Module):
    def __init__(self, input_nc, output_nc, bias=True, gain=np.sqrt(2.0), lrmul=1.0, implicit_lreq=True):
        super(LREQLinear, self).__init__()
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


class LREQConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, dilation=1,
                 groups=1, use_bias=True, gain=np.sqrt(2.0), transpose=False, transform_kernel=False, lrmul=1.0,
                 implicit_lreq=True):
        super(LREQConv2d, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')

        fan_in = np.prod((kernel_size, kernel_size)) * in_channels // groups

        if transpose:
            self.weight = nn.Parameter(torch.zeros(in_channels, out_channels // groups, kernel_size, kernel_size),
                                       requires_grad=True)
        else:
            self.weight = nn.Parameter(torch.zeros(out_channels, in_channels // groups, kernel_size, kernel_size),
                                       requires_grad=True)
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(out_channels), requires_grad=True)
        else:
            self.bias = None

        std = gain / np.sqrt(fan_in)

        if not implicit_lreq:
            nn.init.normal_(self.weight, mean=0, std=1.0 / lrmul)
        else:
            nn.init.normal_(self.weight, mean=0, std=std / lrmul)
            setattr(self.weight, 'lr_equalization_coef', std)
            if use_bias:
                setattr(self.bias, 'lr_equalization_coef', lrmul)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)
        self.padding = (padding, padding)
        self.output_padding = (output_padding, output_padding)
        self.dilation = (dilation, dilation)
        self.groups = groups
        self.gain = gain
        self.lrmul = lrmul
        self.transpose = transpose
        self.implicit_lreq = implicit_lreq
        self.std = std
        self.transform_kernel = transform_kernel

    def forward(self, x):
        if self.transpose:
            w = self.weight
            if self.transform_kernel:
                w = F.pad(w, [1, 1, 1, 1], mode='constant')
                w = w[:, :, 1:, 1:] + w[:, :, :-1, 1:] + w[:, :, 1:, :-1] + w[:, :, :-1, :-1]
            if not self.implicit_lreq:
                bias = self.bias
                if bias is not None:
                    bias = bias * self.lrmul
                return F.conv_transpose2d(x, w * self.std, bias, stride=self.stride,
                                          padding=self.padding, output_padding=self.output_padding,
                                          dilation=self.dilation, groups=self.groups)
            else:
                return F.conv_transpose2d(x, w, self.bias, stride=self.stride, padding=self.padding,
                                          output_padding=self.output_padding, dilation=self.dilation,
                                          groups=self.groups)
        else:
            w = self.weight
            if self.transform_kernel:
                w = F.pad(w, [1, 1, 1, 1], mode='constant')
                w = (w[:, :, 1:, 1:] + w[:, :, :-1, 1:] + w[:, :, 1:, :-1] + w[:, :, :-1, :-1]) * 0.25
            if not self.implicit_lreq:
                bias = self.bias
                if bias is not None:
                    bias = bias * self.lrmul
                return F.conv2d(x, w * self.std, bias, stride=self.stride, padding=self.padding,
                                dilation=self.dilation, groups=self.groups)
            else:
                return F.conv2d(x, w, self.bias, stride=self.stride, padding=self.padding,
                                dilation=self.dilation, groups=self.groups)


class LREQConvTranspose2d(LREQConv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, dilation=1,
                 groups=1, use_bias=True, gain=np.sqrt(2.0), transform_kernel=False, lrmul=1.0,
                 implicit_lreq=True):
        super(LREQConvTranspose2d, self).__init__(in_channels=in_channels,
                                                  out_channels=out_channels,
                                                  kernel_size=kernel_size,
                                                  stride=stride,
                                                  padding=padding,
                                                  output_padding=output_padding,
                                                  dilation=dilation,
                                                  groups=groups,
                                                  use_bias=use_bias,
                                                  gain=gain,
                                                  transpose=True,
                                                  transform_kernel=transform_kernel,
                                                  lrmul=lrmul,
                                                  implicit_lreq=implicit_lreq)
