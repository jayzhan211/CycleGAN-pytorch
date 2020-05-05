import torch.nn as nn
from .lreq import LREQConv2d, LREQLinear, LREQConvTranspose2d
import torch
from torch.nn import functional as F


def style_mod(x, style):
    style = style.view(style.shape[0], 2, x.shape[1], 1, 1)
    return torch.addcmul(style[:, 1], value=1.0, tensor1=x, tensor2=style[:, 0])





class EncodeBlock(nn.Module):
    def __init__(self, input_nc, output_nc, fused_scale, latent_size=256):
        super(EncodeBlock, self).__init__()

        self.conv_1 = LREQConv2d(input_nc, input_nc, kernel_size=3, stride=1, padding=1, use_bias=False)
        self.bias_1 = nn.Parameter(torch.zeros(1, input_nc, 1, 1), requires_grad=True)
        self.instance_norm_1 = nn.InstanceNorm2d(input_nc, affine=False)

        if fused_scale:
            self.conv_2 = LREQConv2d(input_nc, output_nc, kernel_size=3, stride=2, padding=1, use_bias=False,
                                     transform_kernel=True)
        else:
            self.conv_2 = LREQConv2d(input_nc, output_nc, kernel_size=3, stride=1, padding=1, use_bias=False)

        self.bias_2 = nn.Parameter(torch.zeros(1, output_nc, 1, 1), requires_grad=True)
        self.instance_norm_2 = nn.InstanceNorm2d(output_nc, affine=False)

        self.style_1 = LREQLinear(2 * input_nc, latent_size)
        self.style_2 = LREQLinear(2 * output_nc, latent_size)

        self.fused_scale = fused_scale

    def forward(self, x):
        x = self.conv_1(x) + self.bias_1
        x = F.leaky_relu(x, 0.2)
        m = torch.mean(x, dim=[2, 3], keepdim=True)
        std = torch.sqrt(torch.mean((x - m) ** 2, dim=[2, 3], keepdim=True))
        style_1 = torch.cat((m, std), dim=1)
        x = self.instance_norm_1(x)

        x = self.conv_2(x)
        if not self.fused_scale:
            x = F.avg_pool2d(x, 2)
        x = self.bias_2 + x
        x = F.leaky_relu(x, 0.2)
        m = torch.mean(x, dim=[2, 3], keepdim=True)
        std = torch.sqrt(torch.mean((x - m) ** 2, dim=[2, 3], keepdim=True))
        style_2 = torch.cat((m, std), dim=1)

        x = self.instance_norm_2(x)

        w1 = self.style_1(style_1.view(style_1.size(0), style_1.size(1)))
        w2 = self.style_2(style_2.view(style_2.size(0), style_2.size(1)))

        w = torch.cat([w1, w2], dim=1)
        return x, w


class Encoder(nn.Module):
    def __init__(self, num_channels=3, ngf=64, maxf=256, latent_size=256, num_layers=6):
        super(Encoder, self).__init__()
        """
        input = b, 3, h, w
        output = b, 256, 4, 4, w_styles: b, latent_size * num_layers * 2, 1, 1
        """

        from_rgb = [
            # (h,w,3) => (h,w,64)
            LREQConv2d(num_channels, ngf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2),
        ]

        encode_blocks = nn.ModuleList()

        """
        Default: h,w = 256,256
        (h,w,64) => (h/2,w/2,128)
        (h/2,w/2,128) => (h/4,w/4,256)
        (h/4,w/4,256) = > (h/8,w/8,256)
        (h/8,w/8,256) => (h/16,w/16,256)
        (h/16,w/16,256) => (h/32,w/32,256)
        (h/32,w/32,256) => (h/64,w/64,256)
        """
        input_nc = ngf
        mul = 2
        for i in range(num_layers):
            output_nc = min(ngf * mul, maxf)
            fused_scale = input_nc >= maxf
            block = EncodeBlock(input_nc, output_nc, fused_scale, latent_size)
            encode_blocks.append(block)
            input_nc = output_nc
            mul *= 2

        self.num_layers = num_layers
        self.from_rgb = nn.Sequential(*from_rgb)
        self.latent_size = latent_size
        self.encode_blocks = encode_blocks

    def forward(self, x):
        # x.size = batch, channel=3, h, w
        styles = torch.zeros(x.size(0), 0)

        x = self.from_rgb(x)

        for i in range(self.num_layers):
            x, w = self.encode_blocks[i](x)
            styles = torch.cat([w, styles], dim=1)

        return x, styles.view(styles.size(0), styles.size(1), 1, 1)


class DecodeBlock(nn.Module):
    def __init__(self, input_nc, output_nc, fused_scale, latent_size=256):
        super(DecodeBlock, self).__init__()
        if fused_scale:
            self.conv_1 = LREQConvTranspose2d(input_nc, output_nc, kernel_size=3, stride=2, padding=1,
                                              use_bias=False,
                                              transform_kernel=True)
        else:
            self.conv_1 = LREQConv2d(input_nc, output_nc, kernel_size=3, stride=1, padding=1, use_bias=False)

        self.bias_1 = nn.Parameter(torch.zeros(1, output_nc, 1, 1), requires_grad=True)
        self.instance_norm_1 = nn.InstanceNorm2d(output_nc, affine=False, eps=1e-8)

        self.conv_2 = LREQConv2d(output_nc, output_nc, kernel_size=3, stride=1, padding=1, use_bias=False)
        self.bias_2 = nn.Parameter(torch.zeros(1, output_nc, 1, 1), requires_grad=True)
        self.instance_norm_2 = nn.InstanceNorm2d(output_nc, affine=False, eps=1e-8)

        self.style_1 = LREQLinear(latent_size, 2 * output_nc, gain=1)
        self.style_2 = LREQLinear(latent_size, 2 * output_nc, gain=1)

        self.rho = nn.Parameter(torch.ones(1), requires_grad=True)
        self.fused_scale = fused_scale

    def forward(self, x, s1, s2):
        if not self.fused_scale:
            x = F.interpolate(x, scale_factor=2)
        x = self.conv_1(x)
        x = self.bias_1 + x
        x = F.leaky_relu(x, 0.2)
        x = self.instance_norm_1(x)
        x = x + style_mod(x, self.style_1(s1)) * self.rho

        x = self.conv_2(x)
        x = self.bias_2 + x
        x = F.leaky_relu(x, 0.2)
        x = self.instance_norm_2(x)
        x = x + style_mod(x, self.style_2(s2)) * self.rho

        return x



class Generator(nn.Module):
    def __init__(self, num_channels=3, num_layers=6, ngf=64, maxf=256, latent_size=256):
        super(Generator, self).__init__()

        mul = 2 ** num_layers
        input_nc = min(ngf * mul, maxf)

        decode_blocks = nn.ModuleList()

        for i in range(num_layers):
            output_nc = min(ngf * mul // 2, maxf)
            fused_scale = input_nc != output_nc
            block = DecodeBlock(input_nc, output_nc, fused_scale)
            decode_blocks.append(block)
            input_nc = output_nc

        to_rgb = [
            LREQConv2d(input_nc, num_channels, kernel_size=1, stride=1, padding=0, gain=0.03)
        ]

        self.to_rgb = nn.Sequential(*to_rgb)
        self.decode_blocks = decode_blocks
        self.num_layers = num_layers
        self.latent_size = latent_size

    def forward(self, x, styles):
        for i in range(self.num_layers):
            x = self.decode_blocks[i](x, styles[:, 2 * i * self.latent_size:(2 * i + 1) * self.latent_size], styles[:, (2 * i + 1) * self.latent_size:(2 * i + 2) * self.latent_size])

        x = self.to_rgb(x)
        return x
