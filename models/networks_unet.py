import torch.nn as nn
from .lreq import LREQConv2d, LREQLinear, LREQConvTranspose2d
import torch
from torch.nn import functional as F


def style_mod(x, style):
    style = style.view(style.shape[0], 2, x.shape[1], 1, 1)
    return torch.addcmul(style[:, 1], value=1.0, tensor1=x, tensor2=style[:, 0])


class EncodeBlock(nn.Module):
    def __init__(self, input_nc, output_nc, latent_size=256):
        super(EncodeBlock, self).__init__()

        self.conv_1 = nn.Conv2d(input_nc, input_nc, kernel_size=3, stride=1, padding=1, bias=False)
        self.bias_1 = nn.Parameter(torch.zeros(1, input_nc, 1, 1), requires_grad=True)
        self.instance_norm_1 = nn.InstanceNorm2d(input_nc, affine=False)
        self.conv_2 = nn.Conv2d(input_nc, output_nc, kernel_size=3, stride=1, padding=1, bias=False)
        self.bias_2 = nn.Parameter(torch.zeros(1, output_nc, 1, 1), requires_grad=True)
        self.instance_norm_2 = nn.InstanceNorm2d(output_nc, affine=False)

        # self.style_1 = LREQLinear(2 * input_nc, latent_size)
        # self.style_2 = LREQLinear(2 * output_nc, latent_size)
        self.style_mapping = nn.Linear(2 * output_nc, latent_size)

    def forward(self, x):
        x = self.conv_1(x) + self.bias_1
        x = F.leaky_relu(x, 0.2)
        x = self.instance_norm_1(x)
        x = self.conv_2(x) + self.bias_2
        x = F.leaky_relu(x, 0.2)

        m = torch.mean(x, dim=[2, 3], keepdim=True)
        std = torch.sqrt(torch.mean((x - m) ** 2, dim=[2, 3], keepdim=True))
        style = torch.cat([m, std], dim=1)

        x = self.instance_norm_2(x)

        x = F.avg_pool2d(x, 2)
        w = self.style_mapping(style.view(style.size(0), style.size(1)))

        return x, w


class Encoder(nn.Module):
    def __init__(self, latent_size=256, mapping_layers=6, mapping_fmaps=256):
        super(Encoder, self).__init__()
        """
        input = b, 3, h, w
        output = b, 256, 4, 4, w_styles: b, latent_size * num_layers * 2, 1, 1
        """

        # (h,w,3) => (h,w,64)
        from_rgb = [
            nn.ReflectionPad2d(1),
            nn.utils.spectral_norm(
                nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0, bias=True)),
            nn.LeakyReLU(0.2, True)
        ]

        # (h,w,64) => (h/2,w/2,128)
        # (h/2,w/2,128) => (h/4,w/4,256)
        down_blocks = [
            nn.ReflectionPad2d(1),
            nn.utils.spectral_norm(
                nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=0, bias=True)),
            nn.LeakyReLU(0.2, True),

            nn.ReflectionPad2d(1),
            nn.utils.spectral_norm(
                nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=0, bias=True)),
            nn.LeakyReLU(0.2, True),
        ]

        encode_blocks = nn.ModuleList()

        """
        (h/4,w/4,256) = > (h/8,w/8,256)
        (h/8,w/8,256) => (h/16,w/16,256)
        (h/16,w/16,256) => (h/32,w/32,256)
        (h/32,w/32,256) => (h/64,w/64,256)
        """

        for i in range(4):
            block = EncodeBlock(256, 256, latent_size)
            encode_blocks.append(block)

        mapping_blocks = nn.ModuleList()
        input_nc = 256 * 4 * 4
        for i in range(mapping_layers):
            output_nc = 1 if i + 1 == mapping_layers else mapping_fmaps
            # block = LREQLinear(input_nc, output_nc, lrmul=0.1)
            block = nn.Linear(input_nc, output_nc)
            mapping_blocks.append(block)
            input_nc = output_nc

        self.mapping_layers = mapping_layers
        self.mapping_blocks = mapping_blocks
        self.from_rgb = nn.Sequential(*from_rgb)
        self.latent_size = latent_size
        self.encode_blocks = encode_blocks
        self.down_blocks = nn.Sequential(*down_blocks)

    def forward(self, x):
        # x.size = batch, channel=3, h, w
        # styles = torch.zeros(x.size(0), 0)
        # styles = torch.cuda.FloatTensor(x.size(0), 0).fill_(0)

        x = self.from_rgb(x)
        x = self.down_blocks(x)

        x_0 = []
        w_s = []

        for i in range(4):
            x_0.insert(0, x)
            x, w = self.encode_blocks[i](x)
            w_s.insert(0, w)
            # styles = torch.cat([w, styles], dim=1)

        z = x.view(x.size(0), -1)
        for i in range(self.mapping_layers):
            z = self.mapping_blocks[i](z)

        return x_0, w_s, z


class DecodeBlock(nn.Module):
    def __init__(self, input_nc, output_nc, latent_size=256, mapping_layers=3, mapping_fmaps=256):
        super(DecodeBlock, self).__init__()

        self.conv_1 = nn.Conv2d(input_nc, output_nc, kernel_size=3, stride=1, padding=1, bias=False)
        self.bias_1 = nn.Parameter(torch.zeros(1, output_nc, 1, 1), requires_grad=True)
        self.instance_norm_1 = nn.InstanceNorm2d(output_nc, affine=False, eps=1e-8)

        self.conv_2 = LREQConv2d(output_nc, output_nc, kernel_size=3, stride=1, padding=1, use_bias=False)
        self.bias_2 = nn.Parameter(torch.zeros(1, output_nc, 1, 1), requires_grad=True)
        self.instance_norm_2 = nn.InstanceNorm2d(output_nc, affine=False, eps=1e-8)

        mapping_blocks = nn.ModuleList()
        inc = latent_size
        for i in range(mapping_layers):
            outc = 2 * output_nc if i + 1 == mapping_layers else mapping_fmaps
            block = nn.Linear(inc, outc)
            inc = outc
            mapping_blocks.append(block)

        self.mapping_blocks = mapping_blocks
        self.mapping_layers = mapping_layers

    def forward(self, x, x_0, style):
        x = F.interpolate(x, scale_factor=2)

        for i in range(self.mapping_layers):
            style = self.mapping_blocks[i](style)

        x = self.conv_1(x) + self.bias_1
        x = F.leaky_relu(x, 0.2)
        x = self.instance_norm_1(x)
        x = x + style_mod(x, style)

        x = self.conv_2(x) + self.bias_2
        x = F.leaky_relu(x, 0.2)
        x = self.instance_norm_2(x)
        x = x + style_mod(x, style)

        return x + x_0


class Generator(nn.Module):
    def __init__(self, latent_size=256):
        super(Generator, self).__init__()

        decode_blocks = nn.ModuleList()
        for i in range(4):
            block = DecodeBlock(256, 256, latent_size)
            decode_blocks.append(block)

        up_blocks = [
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
        ]

        to_rgb = [
            nn.Conv2d(64, 3, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Tanh(),
            # LREQConv2d(input_nc, num_channels, kernel_size=1, stride=1, padding=0, gain=0.03)
        ]

        self.to_rgb = nn.Sequential(*to_rgb)
        self.decode_blocks = decode_blocks
        self.up_blocks = nn.Sequential(*up_blocks)
        self.latent_size = latent_size

    def forward(self, x, x_0, w_s):
        for i in range(4):
            x = self.decode_blocks[i](x, x_0[i], w_s[i])
        x = self.up_blocks(x)
        x = self.to_rgb(x)
        return x
