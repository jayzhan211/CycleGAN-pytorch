import torch.nn as nn
from torch.nn import functional as F
from .lreq import LREQLinear, LREQConv2d, LREQConvTranspose2d
import torch
import numpy as np


def pixel_norm(x, epsilon=1e-8):
    return x * torch.rsqrt(torch.mean(x.pow(2.0), dim=1, keepdim=True) + epsilon)


def style_mod(x, style):
    style = style.view(style.shape[0], 2, x.shape[1], 1, 1)
    return torch.addcmul(style[:, 1], value=1.0, tensor1=x, tensor2=style[:, 0] + 1)


def upscale2d(x, factor=2):
    s = x.shape
    x = torch.reshape(x, [-1, s[1], s[2], 1, s[3], 1])
    x = x.repeat(1, 1, 1, factor, 1, factor)
    x = torch.reshape(x, [-1, s[1], s[2] * factor, s[3] * factor])
    return x


def downscale2d(x, factor=2):
    return F.avg_pool2d(x, factor)


class Blur(nn.Module):
    def __init__(self, channels):
        super(Blur, self).__init__()
        f = np.array([1, 2, 1], dtype=np.float32)
        f = f[:, np.newaxis] * f[np.newaxis, :]
        f /= np.sum(f)
        kernel = torch.tensor(f).view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
        self.register_buffer('weight', kernel)
        self.groups = channels

    def forward(self, x):
        return F.conv2d(x, weight=self.weight, groups=self.groups, padding=1)


class MappingBlock(nn.Module):
    def __init__(self, input_nc, output_nc, lrmul):
        super(MappingBlock, self).__init__()
        self.fc = LREQLinear(input_nc, output_nc, lrmul=lrmul)

    def forward(self, x):
        x = F.leaky_relu(self.fc(x), 0.2)
        return x


class DecodeBlock(nn.Module):
    def __init__(self, input_nc, output_nc, latent_size, has_first_conv=True, fused_scale=True, layer=0):
        super(DecodeBlock, self).__init__()
        if has_first_conv:
            if fused_scale:
                self.conv_1 = LREQConvTranspose2d(input_nc, output_nc, kernel_size=3, stride=2, padding=1,
                                                  use_bias=False,
                                                  transform_kernel=True)
            else:
                self.conv_1 = LREQConv2d(input_nc, output_nc, kernel_size=3, stride=1, padding=1, use_bias=False)

        self.blur = Blur(output_nc)
        self.noise_weight_1 = nn.Parameter(torch.zeros(1, output_nc, 1, 1), requires_grad=True)
        self.bias_1 = nn.Parameter(torch.zeros(1, output_nc, 1, 1), requires_grad=True)
        self.instance_norm_1 = nn.InstanceNorm2d(output_nc, affine=False, eps=1e-8)
        self.style_1 = LREQLinear(latent_size, 2 * output_nc, gain=1)

        self.conv_2 = LREQConv2d(output_nc, output_nc, kernel_size=3, stride=1, padding=1, use_bias=False)
        self.noise_weight_2 = nn.Parameter(torch.zeros(1, output_nc, 1, 1), requires_grad=True)
        self.bias_2 = nn.Parameter(torch.zeros(1, output_nc, 1, 1), requires_grad=True)
        self.instance_norm_2 = nn.InstanceNorm2d(output_nc, affine=False, eps=1e-8)
        self.style_2 = LREQLinear(latent_size, 2 * output_nc, gain=1)

        self.has_first_conv = has_first_conv
        self.fused_scale = fused_scale
        self.layer = layer

    def forward(self, x, s1, s2, noise):
        if self.has_first_conv:
            if not self.fused_scale:
                x = upscale2d(x)
            x = self.conv_1(x)
            x = self.blur(x)

        if noise:
            if noise == 'batch_constant':
                x_size = torch.Size((1, 1, x.size(2), x.size(3)))
                z = torch.cuda.FloatTensor(x_size)

                x = torch.addcmul(x, value=1.0, tensor1=self.noise_weight_1,
                                  tensor2=torch.randn(x_size, out=z))
            else:
                x_size = torch.Size((x.size(0), 1, x.size(2), x.size(3)))
                z = torch.cuda.FloatTensor(x_size)

                x = torch.addcmul(x, value=1.0, tensor1=self.noise_weight_1,
                                  tensor2=torch.randn(x_size, out=z))

        else:
            s = np.power(self.layer + 1, 0.5)
            x = x + s * torch.exp(-x * x / (2.0 * s * s)) / np.sqrt(2 * np.pi) * 0.8

        x = x + self.bias_1

        x = F.leaky_relu(x, 0.2)

        x = self.instance_norm_1(x)

        x = style_mod(x, self.style_1(s1))

        x = self.conv_2(x)

        if noise:
            if noise == 'batch_constant':
                x_size = torch.Size((1, 1, x.size(2), x.size(3)))
                z = torch.cuda.FloatTensor(x_size)

                x = torch.addcmul(x, value=1.0, tensor1=self.noise_weight_2,
                                  tensor2=torch.randn(x_size, out=z))
            else:
                x_size = torch.Size((x.size(0), 1, x.size(2), x.size(3)))
                z = torch.cuda.FloatTensor(x_size)

                x = torch.addcmul(x, value=1.0, tensor1=self.noise_weight_2,
                                  tensor2=torch.randn(x_size, out=z))

        else:
            s = np.power(self.layer + 1, 0.5)
            x = x + s * torch.exp(-x * x / (2.0 * s * s)) / np.sqrt(2 * np.pi) * 0.8

        x = x + self.bias_2

        x = F.leaky_relu(x, 0.2)
        x = self.instance_norm_2(x)

        x = style_mod(x, self.style_2(s2))

        return x


class FromRGB(nn.Module):
    def __init__(self, input_nc, output_nc):
        super(FromRGB, self).__init__()
        self.from_rgb = LREQConv2d(input_nc, output_nc, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.from_rgb(x)
        x = F.leaky_relu(x, 0.2)

        return x


class ToRGB(nn.Module):
    def __init__(self, input_nc, output_nc):
        super(ToRGB, self).__init__()
        self.to_rgb = LREQConv2d(input_nc, output_nc, kernel_size=1, stride=1, padding=0, gain=0.03)

    def forward(self, x):
        x = self.to_rgb(x)
        return x


class VAEMappingFromLatent(nn.Module):
    """
    input: (batch_size, latent_size)
    output: (batch_size, latent_size)
    """

    def __init__(self, num_layers, mapping_layers=5, latent_size=256, mapping_fmaps=256):
        super(VAEMappingFromLatent, self).__init__()
        input_nc = latent_size
        blocks = nn.ModuleList()
        for i in range(mapping_layers):
            output_nc = latent_size if i == mapping_layers - 1 else mapping_fmaps
            block = MappingBlock(input_nc, output_nc, lrmul=0.1)
            input_nc = output_nc
            blocks.append(block)

        self.blocks = blocks
        self.mapping_layers = mapping_layers
        self.num_layers = num_layers

    def forward(self, x):
        x = pixel_norm(x)
        for i in range(self.mapping_layers):
            x = self.blocks[i](x)

        return x


class Generator(nn.Module):
    def __init__(self, init_f=32, max_f=256, num_layers=6, latent_size=128, num_channels=3):
        super(Generator, self).__init__()

        mul = 2 ** (num_layers - 1)
        input_nc = min(init_f * mul, max_f)
        self.init_image = nn.Parameter(torch.ones(1, input_nc, 4, 4), requires_grad=True)
        resolution = 2

        decode_blocks = nn.ModuleList()
        style_sizes = []

        for i in range(num_layers):
            output_nc = min(init_f * mul, max_f)
            has_first_conv = i != 0
            fused_scale = resolution >= 64
            block = DecodeBlock(input_nc, output_nc, latent_size, has_first_conv, fused_scale, layer=i)
            resolution *= 2

            style_sizes += [2 * (input_nc if has_first_conv else output_nc), 2 * output_nc]

            decode_blocks.append(block)

            input_nc = output_nc
            mul //= 2


        self.to_rgb = ToRGB(input_nc, num_channels)
        self.decode_blocks = decode_blocks
        self.style_sizes = style_sizes

        self.num_layers = num_layers

    def decode(self, styles, noise):
        x = self.init_image

        for i in range(self.num_layers):
            x = self.decode_blocks[i](x, styles[:, 2 * i + 0], styles[:, 2 * i + 1], noise)
        x = self.to_rgb(x)
        return x

    def forward(self, styles, noise):
        return self.decode(styles, noise)


class EncodeBlock(nn.Module):
    def __init__(self, input_nc, output_nc, latent_size, last=False, fused_scale=True):
        super(EncodeBlock, self).__init__()
        self.conv_1 = LREQConv2d(input_nc, input_nc, kernel_size=3, stride=1, padding=1, use_bias=False)
        self.bias_1 = nn.Parameter(torch.zeros(1, input_nc, 1, 1), requires_grad=True)
        self.instance_norm_1 = nn.InstanceNorm2d(input_nc, affine=False)
        self.blur = Blur(input_nc)
        if last:
            self.dense = LREQLinear(input_nc * 4 * 4, output_nc)
        else:
            if fused_scale:
                self.conv_2 = LREQConv2d(input_nc, output_nc, kernel_size=3, stride=2, padding=1, use_bias=False,
                                         transform_kernel=True)
            else:
                self.conv_2 = LREQConv2d(input_nc, output_nc, kernel_size=3, stride=1, padding=1, use_bias=False)

        self.bias_2 = nn.Parameter(torch.zeros(1, output_nc, 1, 1), requires_grad=True)
        self.instance_norm_2 = nn.InstanceNorm2d(output_nc, affine=False)

        self.style_1 = LREQLinear(2 * input_nc, latent_size)
        if last:
            self.style_2 = LREQLinear(output_nc, latent_size)
        else:
            self.style_2 = LREQLinear(2 * output_nc, latent_size)

        self.fused_scale = fused_scale
        self.last = last


    def forward(self, x):
        x = self.conv_1(x) + self.bias_1
        x = F.leaky_relu(x, 0.2)

        m = torch.mean(x, dim=[2, 3], keepdim=True)
        std = torch.sqrt(torch.mean((x - m) ** 2, dim=[2, 3], keepdim=True))
        style_1 = torch.cat((m, std), dim=1)
        x = self.instance_norm_1(x)

        if self.last:
            x = self.dense(x.view(x.shape[0], -1))
            x = F.leaky_relu(x, 0.2)
            w1 = self.style_1(style_1.view(style_1.size(0), style_1.size(1)))
            w2 = self.style_2(x.view(x.size(0),x.size(1)))
        else:
            x = self.conv_2(self.blur(x))
            if not self.fused_scale:
                x = downscale2d(x)
            x = x + self.bias_2
            x = F.leaky_relu(x, 0.2)
            m = torch.mean(x, dim=[2, 3], keepdim=True)
            std = torch.sqrt(torch.mean((x - m) ** 2, dim=[2, 3], keepdim=True))
            style_2 = torch.cat((m, std), dim=1)

            x = self.instance_norm_2(x)
            w1 = self.style_1(style_1.view(style_1.size(0), style_1.size(1)))
            w2 = self.style_2(style_2.view(style_2.size(0), style_2.size(1)))

        return x, w1, w2


class Encoder(nn.Module):
    """
    input: b, c=3, h=256, w=256
    output: b, latent_size
    """

    def __init__(self, init_f, max_f, latent_size, num_layers=6, num_channels=3):
        super(Encoder, self).__init__()

        input_nc = init_f

        from_rgb = FromRGB(num_channels, input_nc)
        encode_blocks = nn.ModuleList()

        resolution = 2 ** (num_layers + 1)

        mul = 2
        for i in range(num_layers):
            output_nc = min(max_f, init_f * mul)
            fused_scale = resolution >= 128
            block = EncodeBlock(input_nc, output_nc, latent_size, last=False, fused_scale=fused_scale)
            resolution //= 2
            encode_blocks.append(block)
            input_nc = output_nc
            mul *= 2

        self.num_layers = num_layers
        self.from_rgb = from_rgb
        self.latent_size = latent_size
        self.encode_blocks = encode_blocks

    def encode(self, x):
        # styles = torch.zeros(x.shape[0], self.latent_size)
        styles = torch.cuda.FloatTensor(x.shape[0], self.latent_size).fill_(0)

        x = self.from_rgb(x)
        x = F.leaky_relu(x, 0.2)

        for i in range(self.num_layers):
            x, s1, s2 = self.encode_blocks[i](x)
            styles += s1 + s2

        return styles

    def forward(self, x):
        return self.encode(x)


class Discriminator(nn.Module):
    """
    input: batch_size, latent_size
    output: batch_size, 1
    """

    def __init__(self, mapping_layers=5, latent_size=256, mapping_fmaps=256):
        super(Discriminator, self).__init__()
        input_nc = latent_size
        map_blocks = nn.ModuleList()
        for i in range(mapping_layers):
            output_nc = 1 if i + 1 == mapping_layers else mapping_fmaps
            block = LREQLinear(input_nc, output_nc, lrmul=0.1)
            input_nc = output_nc
            map_blocks.append(block)

        self.mapping_layers = mapping_layers
        self.map_blocks = map_blocks

    def forward(self, x):
        for i in range(self.mapping_layers):
            x = self.map_blocks[i](x)

        return x


