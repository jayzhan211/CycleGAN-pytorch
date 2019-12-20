import functools
import torch.nn as nn
import torch
import torchvision
from torch.nn import init
from torch.optim import lr_scheduler
from torch.nn import functional as F

###############################################################################
# Helper Functions
###############################################################################
from utils.util import toGray, to3channel, calc_mean_std


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_scheduler(optimizer, opt):
    """
    Return a learning rate scheduler
    :param optimizer: the optimizer of the network
    :param opt: stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine
    :return:
    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'linear_style':
        def lambda_rule(epoch):
            lr_l = 1.0 / float(opt.lr_decay * epoch + 1.0)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    elif opt.lr_policy == 'none':
        scheduler = None
    else:
        return NotImplementedError('learning rate policy [{}] is not implemented'.format(opt.lr_policy))
    return scheduler


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
            nn.ReLU(True)
        ]

        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim)
        ]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class ResnetBlockUGATIT(nn.Module):
    def __init__(self, dim, use_bias):
        super(ResnetBlockUGATIT, self).__init__()
        conv_block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias),
            nn.InstanceNorm2d(dim),
        ]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.
    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator
        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)

# class ResnetGenerator(nn.Module):
#     def __init__(self, input_nc,
#                  output_nc,
#                  ngf=64,
#                  norm_layer=nn.InstanceNorm2d,
#                  use_dropout=False,
#                  n_blocks=6,
#                  padding_type='reflect'):
#         """
#         Define a ResNet Block
#         :param input_nc:
#         :param output_nc:
#         :param ngf:
#         :param norm_layer:
#         :param use_dropout:
#         :param n_blocks: number of ResNet block
#         :param padding_type: padding in conv: reflect | replicate | zero
#         """
#         assert n_blocks >= 0
#         super(ResnetGenerator, self).__init__()
#
#         # batch_norm has bias term already, use_bias only if we use instance_norm
#         use_bias = norm_layer == nn.InstanceNorm2d
#
#         model = [nn.ReflectionPad2d(3),
#                  nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
#                  norm_layer(ngf),
#                  nn.ReLU(True)]
#
#         n_downsampling = 2
#         for i in range(n_downsampling):  # add downsampling layers
#             mult = 2 ** i
#             model += [
#                 nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
#                 norm_layer(ngf * mult * 2),
#                 nn.ReLU(True)
#             ]
#         mult = 2 ** n_downsampling
#         for i in range(n_blocks):  # add ResNet blocks
#
#             model += [
#                 ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
#                             use_dropout=use_dropout, use_bias=use_bias)
#             ]
#
#         for i in range(n_downsampling):  # add upsampling layers
#             mult = 2 ** (n_downsampling - i)
#             model += [
#                 nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
#                                    kernel_size=3, stride=2,
#                                    padding=1, output_padding=1,
#                                    bias=use_bias),
#                 norm_layer(int(ngf * mult / 2)),
#                 nn.ReLU(True)
#             ]
#
#         model += [nn.ReflectionPad2d(3)]
#         model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
#         model += [nn.Tanh()]
#
#         self.model = nn.Sequential(*model)
#
#     def forward(self, x):
#         return self.model(x)


class AdaLIn(nn.Module):
    """
    Adaptive Layer-Instance Normalization
    0.9 for in, 0.1 for ln in the beginning
    """

    def __init__(self, num_features, eps=1e-5):
        super(AdaLIn, self).__init__()
        self.eps = eps
        self.rho = nn.Parameter(torch.full((1, num_features, 1, 1), 0.9), requires_grad=True)

    def forward(self, x, gamma, beta):
        in_mean, in_var = torch.mean(x, dim=[2, 3], keepdim=True), torch.var(x, dim=[2, 3], keepdim=True)
        out_in = (x - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(x, dim=[1, 2, 3], keepdim=True), torch.var(x, dim=[1, 2, 3], keepdim=True)
        out_ln = (x - ln_mean) / torch.sqrt(ln_var + self.eps)
        _, c, h, w = self.rho.size()
        out = self.rho.expand(x.shape[0], -1, -1, -1) * out_in + (
                - self.rho.expand(x.shape[0], -1, -1, -1) + 1.0) * out_ln
        out = out * gamma.unsqueeze(2).unsqueeze(3) + beta.unsqueeze(2).unsqueeze(3)
        return out


class AdaIn(nn.Module):
    def __init__(self, eps=1e-5):
        super(AdaIn, self).__init__()
        self.eps = eps

    def forward(self, x, gamma, beta):
        """
        Adaptive Instance Normalization
        :param x: size = b, c, h, w
        :param gamma: b, c
        :param beta: b, c
        :return:
        """
        in_mean, in_var = torch.mean(x, dim=[2, 3], keepdim=True), torch.var(x, dim=[2, 3], keepdim=True)
        out_in = (x - in_mean) / torch.sqrt(in_var + self.eps)
        out_in = out_in.expand(x.shape[0], -1, -1, -1)
        out = out_in * gamma.unsqueeze(2).unsqueeze(3) + beta.unsqueeze(2).unsqueeze(3)
        return out


class LIn(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(LIn, self).__init__()
        self.eps = eps

        self.rho = nn.Parameter(torch.zeros(1, num_features, 1, 1), requires_grad=True)
        self.gamma = nn.Parameter(torch.ones(1, num_features, 1, 1), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(1, num_features, 1, 1), requires_grad=True)

    def forward(self, x):
        in_mean, in_var = torch.mean(x, dim=[2, 3], keepdim=True), torch.var(x, dim=[2, 3], keepdim=True)
        out_in = (x - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(x, dim=[1, 2, 3], keepdim=True), torch.var(x, dim=[1, 2, 3], keepdim=True)
        out_ln = (x - ln_mean) / torch.sqrt(ln_var + self.eps)
        _, c, h, w = self.rho.size()
        out = self.rho.expand(x.shape[0], -1, -1, -1) * out_in + (
                - self.rho.expand(x.shape[0], -1, -1, -1) + 1.0) * out_ln
        out = out * self.gamma.expand(x.shape[0], -1, -1, -1) + self.beta.expand(x.shape[0], -1, -1, -1)
        return out


class ResnetAdaLInBlock(nn.Module):
    def __init__(self, dim, use_bias):
        super(ResnetAdaLInBlock, self).__init__()
        self.pad1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias)
        self.norm1 = AdaLIn(dim)
        self.relu1 = nn.ReLU(True)

        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias)
        self.norm2 = AdaLIn(dim)

    def forward(self, x, gamma, beta):
        out = self.pad1(x)
        out = self.conv1(out)
        out = self.norm1(out, gamma, beta)
        out = self.relu1(out)
        out = self.pad2(out)
        out = self.conv2(out)
        out = self.norm2(out, gamma, beta)

        return out + x


class ResnetAdaInBlock(nn.Module):
    def __init__(self, dim, use_bias):
        super(ResnetAdaInBlock, self).__init__()
        self.pad1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias)
        self.norm1 = AdaIn()
        self.relu1 = nn.ReLU(True)

        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias)
        self.norm2 = AdaIn()

    def forward(self, x, gamma, beta):
        out = self.pad1(x)
        out = self.conv1(out)
        out = self.norm1(out, gamma, beta)
        out = self.relu1(out)
        out = self.pad2(out)
        out = self.conv2(out)
        out = self.norm2(out, gamma, beta)

        return out + x


class ResnetGeneratorUGATIT(nn.Module):
    def __init__(self,
                 input_nc,
                 output_nc,
                 ngf,
                 n_blocks=6,
                 img_size=256,
                 light=False):
        """
        Generator
        :param input_nc: (int) the number of channels in input images
        :param output_nc: (int) the number of channels in output images
        :param ngf: (int) the number of filters in the last conv layer
        :param norm_layer:
        :param n_blocks:
        :param img_size:
        :param light:
        :param padding_type:
        :param use_dropout:
        """
        super(ResnetGeneratorUGATIT, self).__init__()

        # self.norm_layer = norm_layer
        self.light = light
        self.img_size = img_size
        self.n_blocks = n_blocks
        self.ngf = ngf
        self.output_nc = output_nc
        self.input_nc = input_nc

        # down-sampling
        DownBlock = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, bias=False),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True),
        ]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            DownBlock += [
                nn.ReflectionPad2d(1),
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, bias=False),
                nn.InstanceNorm2d(ngf * mult * 2),
                nn.ReLU(True)
            ]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):  # add ResNet blocks
            DownBlock += [
                ResnetBlockUGATIT(ngf * mult, use_bias=False),
            ]

        # Class Activate Map
        # global average pooling
        self.gap_fc = nn.Linear(ngf * mult, 1, bias=False)
        # global maximum pooling
        self.gmp_fc = nn.Linear(ngf * mult, 1, bias=False)
        self.conv1x1 = nn.Conv2d(ngf * mult * 2, ngf * mult, kernel_size=1)
        self.relu = nn.ReLU(True)

        # Gamma, Beta Block
        if self.light:
            FC = [
                nn.Linear(ngf * mult, ngf * mult, bias=False),
                nn.ReLU(True),
                nn.Linear(ngf * mult, ngf * mult, bias=False),
                nn.ReLU(True),
            ]
        else:
            FC = [
                nn.Linear(img_size // mult * img_size // mult * ngf * mult, ngf * mult, bias=False),
                nn.ReLU(True),
                nn.Linear(ngf * mult, ngf * mult, bias=False),
                nn.ReLU(True),
            ]

        self.gamma = nn.Linear(ngf * mult, ngf * mult, bias=False)
        self.beta = nn.Linear(ngf * mult, ngf * mult, bias=False)

        # Up-sampling Bottleneck
        for i in range(n_blocks):
            setattr(self, 'UpBlock1_{}'.format(i + 1), ResnetAdaLInBlock(ngf * mult, use_bias=False))

        # Up-sampling
        UpBlock2 = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            UpBlock2 += [
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.ReflectionPad2d(1),
                nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, bias=False),
                LIn(int(ngf * mult / 2)),
                nn.ReLU(True),
            ]
        UpBlock2 += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, bias=False),
            nn.Tanh()
        ]

        self.DownBlock = nn.Sequential(*DownBlock)
        self.FC = nn.Sequential(*FC)
        self.UpBlock = nn.Sequential(*UpBlock2)

    def forward(self, x):
        x = self.DownBlock(x)
        gap = F.adaptive_avg_pool2d(x, 1)
        gap_logit = self.gap_fc(gap.view(gap.shape[0], -1))
        gap_weight = list(self.gap_fc.parameters())[0]
        gap = x * gap_weight.unsqueeze(2).unsqueeze(3)

        gmp = F.adaptive_max_pool2d(x, 1)
        gmp_logit = self.gmp_fc(gmp.view(gmp.shape[0], -1))
        gmp_weight = list(self.gmp_fc.parameters())[0]
        gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)

        cam_logit = torch.cat((gap_logit, gmp_logit), 1)
        x = torch.cat((gap, gmp), 1)
        x = self.relu(self.conv1x1(x))

        heatmap = torch.sum(x, dim=1, keepdim=True)

        if self.light:
            _x = F.adaptive_avg_pool2d(x, 1)
            _x = self.FC(_x.view(_x.shape[0], -1))
        else:
            _x = self.FC(x.view(x.shape[0], -1))

        gamma, beta = self.gamma(_x), self.beta(_x)

        for i in range(self.n_blocks):
            x = getattr(self, 'UpBlock1_{}'.format(i + 1))(x, gamma, beta)
        out = self.UpBlock(x)

        return out, cam_logit, heatmap


class ResnetGeneratorColorization(nn.Module):
    def __init__(self,
                 input_nc,
                 output_nc,
                 ngf=64,
                 norm_layer=nn.InstanceNorm2d,
                 use_dropout=False,
                 # img_size=256,
                 n_blocks=6,
                 padding_type='reflect'):
        """

        input: gray_scale
        output: paint to rgb, style-like the original rgb image

        :param input_nc:
        :param output_nc:
        :param ngf:
        :param n_blocks:
        """
        super(ResnetGeneratorColorization, self).__init__()

        assert input_nc == 1, 'input_nc should be 1'
        assert output_nc == 3, 'input_nc should be 3'

        use_bias = False

        # DownSample
        DownBlock = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, stride=1, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True)
        ]
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            DownBlock += [
                nn.ReflectionPad2d(1),
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=0, bias=use_bias),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True)
            ]
        # current number of filters: 4 * ngf
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            DownBlock += [
                ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                            use_bias=use_bias)
            ]

        # Up-sampling Bottleneck
        for i in range(n_blocks):
            setattr(self, 'AdaInBlock_{}'.format(i + 1), ResnetAdaInBlock(ngf * mult, use_bias=False))

        # UpSampling
        UpBlock = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            UpBlock += [
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.ReflectionPad2d(1),
                nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=0, bias=use_bias),
                norm_layer(int(ngf * mult / 2)),
                nn.ReLU(True)
            ]

        UpBlock += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, stride=1, padding=0, bias=use_bias),
            nn.Tanh()
        ]

        self.DownBlock = nn.Sequential(*DownBlock)
        self.UpBlock = nn.Sequential(*UpBlock)

    def forward(self, img):
        """
        gray_scale img
        :param img:
        :return:
        """

        img = self.DownBlock(img)

        # img_rgb = img_gray.clone()
        # for i in range(self.n_blocks):
        #     img_rgb = getattr(self, 'AdaInBlock_{}'.format(i + 1))(img_gray, gamma, beta)
        # img_rgb = self.UpSample_RBG(img_rgb)
        # img_gray = self.UpSample_Gray(img_gray)

        return img


class DiscriminatorCycleGANColorization(nn.Module):
    def __init__(self,
                 input_nc,
                 ndf=64,
                 n_layers=3):
        """

        :param input_nc:
        :param ndf:
        :param n_layers:
        """
        super(DiscriminatorCycleGANColorization, self).__init__()
        model = [
            nn.ReflectionPad2d(1),
            nn.utils.spectral_norm(
                nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, bias=True)
            ),
            nn.LeakyReLU(0.2, True)
        ]

        for i in range(1, n_layers):
            mult = 2 ** (i - 1)
            model += [
                nn.ReflectionPad2d(1),
                nn.utils.spectral_norm(
                    nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=2, bias=True)
                ),
                nn.LeakyReLU(0.2, True),
            ]

        mult = 2 ** (n_layers - 1)
        model += [
            nn.ReflectionPad2d(1),
            nn.utils.spectral_norm(
                nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=1, bias=True)
            ),
            nn.LeakyReLU(0.2, True),
        ]

        model += [
            nn.ReflectionPad2d(1),
            nn.utils.spectral_norm(
                nn.Conv2d(ndf * mult * 2, 3, kernel_size=4, stride=1, bias=False)
            ),
        ]

        self.model = nn.Sequential(*model)

    def forward(self, _input):
        return self.model(_input)


class DiscriminatorUGATIT(nn.Module):
    def __init__(self,
                 input_nc,
                 ndf=64,
                 n_layers=5):
        """
        Target Discriminator
        :param input_nc: the number of channels in input image
        :param ndf: the number of filters in the first conv layer
        :param n_layers:
        5 -> 3 for down-sampling and 2 for cam
        7 -> 5 for down-sampling and 2 for cam
        return output of encoder, cam_logit, heatmap
        """
        super(DiscriminatorUGATIT, self).__init__()
        model = [
            nn.ReflectionPad2d(1),
            nn.utils.spectral_norm(
                nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, bias=True)
            ),
            nn.LeakyReLU(0.2, True),
        ]

        for i in range(n_layers - 3):
            mult = 2 ** i
            model += [
                nn.ReflectionPad2d(1),
                nn.utils.spectral_norm(
                    nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=2, bias=True)
                ),
                nn.LeakyReLU(0.2, True),
            ]
        mult = 2 ** (n_layers - 3)
        model += [
            nn.ReflectionPad2d(1),
            nn.utils.spectral_norm(
                nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=1, bias=True)
            ),
            nn.LeakyReLU(0.2, True),
        ]

        # CAM
        mult = 2 ** (n_layers - 2)
        self.gap_fc = nn.utils.spectral_norm(
            nn.Linear(ndf * mult, 1, bias=False)
        )
        self.gmp_fc = nn.utils.spectral_norm(
            nn.Linear(ndf * mult, 1, bias=False)
        )
        self.conv1x1 = nn.Conv2d(ndf * mult * 2, ndf * mult, kernel_size=1, bias=False)
        self.relu = nn.LeakyReLU(0.2, True)
        self.model = nn.Sequential(*model)
        classify = [
            nn.ReflectionPad2d(1),
            nn.utils.spectral_norm(
                nn.Conv2d(ndf * mult, 1, kernel_size=4, bias=False)
            )
        ]

        self.classify = nn.Sequential(*classify)

    def forward(self, x):
        x = self.model(x)

        gap = F.adaptive_avg_pool2d(x, 1)
        gap_logit = self.gap_fc(gap.view(x.shape[0], -1))
        gap_weight = list(self.gap_fc.parameters())[0]
        gap = x * gap_weight.unsqueeze(2).unsqueeze(3)

        gmp = F.adaptive_max_pool2d(x, 1)
        gmp_logit = self.gmp_fc(gmp.view(x.shape[0], -1))
        gmp_weight = list(self.gmp_fc.parameters())[0]
        gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)

        cam_logit = torch.cat([gap_logit, gmp_logit], 1)
        x = torch.cat([gap, gmp], 1)
        x = self.relu(self.conv1x1(x))

        heatmap = torch.sum(x, dim=1, keepdim=True)

        out = self.classify(x)

        return out, cam_logit, heatmap



class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.
        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)



# class NLayerDiscriminator(nn.Module):
#     def __init__(self,
#                  input_nc,
#                  ndf=64,
#                  n_layers=3,
#                  norm_type='instance_norm',
#                  padding_type='reflect',
#                  relu_type='leaky'):
#         """
#         PatchGAN discriminator
#         :param input_nc:
#         :param ndf:
#         :param n_layers:
#         :param norm_layer:
#         """
#         super(NLayerDiscriminator, self).__init__()
#
#         assert relu_type == 'leaky'
#         assert padding_type == 'basic'
#         assert norm_type == 'instance_norm'
#
#
#         kw = 4
#         padw = 1
#         use_bias = norm_type != 'batch_norm'
#
#         model = []
#         model.extend(
#             self.make_layers(input_nc, ndf, kw, 2, padw, True, padding_type, 'none', relu_type))
#
#         nf = 1
#         for i in range(1, n_layers):
#             nf_prev = nf
#             nf = min(nf * 2, 8)
#             model.extend(
#                 self.make_layers(ndf * nf_prev, ndf * nf, kw, 2, padw, use_bias, padding_type, norm_type, relu_type))
#
#         nf_prev = nf
#         nf = min(nf * 2, 8)
#         model.extend(
#             self.make_layers(ndf * nf_prev, ndf * nf, kw, 1, padw, use_bias, padding_type, norm_type, relu_type))
#
#         model.extend(
#             self.make_layers(ndf * nf, 1, kw, 1, padw, norm_type != 'spectral_norm', padding_type, norm_type,
#                              relu_type))
#
#         self.model = nn.Sequential(*model)
#
#     def forward(self, x):
#         return self.model(x)
#
#     def make_layers(self, in_channels, out_channels, kernel_size, stride, padding,
#                     bias, padding_type, norm_type, relu_type):
#         model = []
#         if padding_type == 'reflect':
#             model += [nn.ReflectionPad2d(padding)]
#         conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
#         if norm_type == 'spectral_norm':
#             model += [nn.utils.spectral_norm(conv2d)]
#         elif norm_type == 'batch_norm':
#             model += [conv2d, nn.BatchNorm2d(out_channels)]
#         elif norm_type == 'instance_norm':
#             model += [conv2d, nn.InstanceNorm2d(out_channels)]
#         elif norm_type == 'none':
#             model += [conv2d]
#
#         if relu_type == 'leaky':
#             model += [nn.LeakyReLU(0.2, True)]
#         return model


def init_weights(net, init_type='normal', init_gain=0.02):
    """
    Initialize network weights.
    :param net: (net)
    :param init_type: (str) normal | xavier | kaiming | orthogonal
    :param init_gain: scaling factor for normal, xavier and orthogonal.
    :return:
    """

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [{}] is not implemented'.format(init_type))
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

        print('initialize network with {}'.format(init_type))
        net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[], init_weight=True):
    """
    Use Gpu/Cpu and initialize weight
    :param net:
    :param init_type: normal | xavier | kaiming | orthogonal
    :param init_gain: scaling factor
    :param gpu_ids:
    :return:
    """
    if len(gpu_ids) > 0:
        assert torch.cuda.is_available()
        net.to(gpu_ids[0])
    net = torch.nn.DataParallel(net, gpu_ids)

    if init_weight:
        init_weights(net, init_type, init_gain=init_gain)

    return net


# def define_G(input_nc,
#              output_nc,
#              ngf,
#              netG,
#              gpu_ids=None,
#              norm='instance_norm',
#              use_dropout=False,
#              init_type='normal',
#              init_gain=0.02,
#              light=False):
#     """
#     :param netG: Generator_name
#     :param light: light mode (UGATIT)
#     :param input_nc: number of channels in input_images
#     :param output_nc: number of channels output_images
#     :param ngf: number of filters in last conv_layer
#     :param norm: batch_norm | instance_norm | none
#     :param use_dropout:
#     :param init_type: initialize method
#     :param init_gain: scaling factor for normal, xavier, orthogonal
#     :param gpu_ids: (list or None) e.g., [0],[0, 1]
#     :return:
#
#     use ReLU for non-linearity
#     """
#
#     if gpu_ids is None:
#         gpu_ids = []
#
#     if norm == 'batch_norm':
#         norm_layer = nn.BatchNorm2d
#     elif norm == 'instance_norm':
#         norm_layer = nn.InstanceNorm2d
#     elif norm == 'none':
#         norm_layer = None
#     else:
#         raise NotImplementedError('norm_layer: {} is not implemented ... , '
#                                   'use batch_norm or instance_norm instead'.format(norm))
#
#     if netG == 'resnet_9blocks':
#         net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
#     elif netG == 'resnet_6blocks':
#         net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
#     elif netG == 'resnet_ugatit_6blocks':
#         net = ResnetGeneratorUGATIT(input_nc, output_nc, ngf=ngf, light=light)
#     elif netG == 'resnet_6blocks_colorization':
#         net = ResnetGeneratorColorization(input_nc, ngf=ngf, n_blocks=6)
#     elif netG == 'resnet_9blocks_colorization':
#         net = ResnetGeneratorColorization(input_nc, ngf=ngf, n_blocks=9)
#     else:
#         raise NotImplementedError('Generator model name: {} is not recognized,'
#                                   'use [resnet_9blocks, resnet_6blocks, resnet_ugatit_6blocks]'.format(netG))
#
#     return init_net(net, init_type, init_gain, gpu_ids)

def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a generator
    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    Returns a generator
    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597
        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).
    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)

# def define_D(input_nc,
#              ndf,
#              netD,
#              n_layers=3,
#              norm_type='instance_norm',
#              padding_type='basic',
#              relu_type='leaky',
#              init_type='normal',
#              init_gain=0.02,
#              gpu_ids=None):
#     """
#     Discriminator,  it uses leaky relu
#     :param input_nc: the number of channels in input images
#     :param ndf: number of filter in the first conv layer
#     :param netD: basic | n_layers | ugatit
#     :param n_layers_d:
#     :param norm: batch_norm | instance_norm | spectral_norm | none
#     :param use_dropout:
#     :param init_type:
#     :param init_gain:
#     :param gpu_ids:
#     :return:
#     """
# 
#     if gpu_ids is None:
#         gpu_ids = []
# 
#     if norm_type not in ['instance_norm', 'batch_norm', 'spectral_norm']:
#         raise NotImplementedError('norm_layer [{}] is not implemented ... '
#                                   'use "instance_norm", "batch_norm", "spectral_norm" instead'.format(norm_type))
# 
#     if padding_type not in ['reflect', 'basic']:
#         raise NotImplementedError('padding_type [{}] is not implemented ... , '
#                                   'use "reflect", "basic" instead'.format(padding_type))
# 
#     if netD == 'n_layers':
#         net = NLayerDiscriminator(input_nc, ndf, n_layers=n_layers, norm_type=norm_type, padding_type=padding_type,
#                                   relu_type=relu_type)
# 
#     elif netD in ['UGATIT', 'ugatit']:
#         if n_layers == 5:
#             net = DiscriminatorUGATIT(input_nc, ndf, n_layers=5)
#         elif n_layers == 7:
#             net = DiscriminatorUGATIT(input_nc, ndf, n_layers=7)
#         else:
#             raise NotImplementedError('Discriminator in UGATIT supports n_layers 5 and 7 only,'
#                                       ' you get {} instead'.format(n_layers))
#     elif netD in ['Cyclegan_colorization', 'cyclegan_colorization']:
#         net = DiscriminatorCycleGANColorization(input_nc, ndf, n_layers=n_layers)
#     else:
#         raise NotImplementedError('Discriminator model name [{}] is not recognized'.format(netD))
# 
#     return init_net(net, init_type, init_gain, gpu_ids)

def get_norm_layer(norm_type='instance'):
    """Return a normalization layer
    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none
    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator
    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    Returns a discriminator
    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.
        [n_layers]: With this mode, you cna specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)
        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.
    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)

class VGG19(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG19, self).__init__()

        layers = []
        in_channels = 3
        for v in [64, 64, 'M',
                  128, 128, 'M',
                  256, 256, 256, 256, 'M',
                  512, 512, 512, 512, 'M',
                  512, 512, 512, 512, 'M']:
            if v == 'M':
                layers += [
                    nn.MaxPool2d(kernel_size=2, stride=2)
                ]
            else:
                layers += [
                    nn.Conv2d(in_channels, v, kernel_size=3, padding=1),
                    nn.BatchNorm2d(v),
                    nn.ReLU(inplace=True)
                ]
                in_channels = v

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

        self.load_state_dict(torch.load('models/vgg19_bn-c79401a0.pth'))

        self.enc_1 = nn.Sequential(*list(self.features.children())[:3])  # input => relu 1-1
        self.enc_2 = nn.Sequential(*list(self.features.children())[3:10])  # relu 1-1 => relu 2-1
        self.enc_3 = nn.Sequential(*list(self.features.children())[10:17])  # relu 2-1 => relu 3-1
        self.enc_4 = nn.Sequential(*list(self.features.children())[17:30])  # relu 3-1 => relu 4-1
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    def forward(self, input):

        # b, _, _, _ = input.size()
        # m = torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        # for i in range(b):
        #     input[i] = m(input[i])

        results = [input]
        for i in range(4):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]


def define_Net(netType,
               init_type='normal',
               init_gain=0.02,
               gpu_ids=None,
               pretrained_encoder=None,
               pretrained_decoder=None):
    """

    """
    net = None

    if gpu_ids is None:
        gpu_ids = []

    if netType == 'adainstyletransfer':
        net = AdaInStyleTransfer(pretrained_encoder, pretrained_decoder)
    elif netType == 'vgg19':
        net = VGG19()
    else:
        raise NotImplementedError('net_type: [{}] is not implemented.'.format(netType))

    return init_net(net, init_type, init_gain, gpu_ids, False)


##############################################################################
# Helper Classes
##############################################################################


class GANLoss(nn.Module):
    def __init__(self,
                 gan_mode,
                 target_real_label=1.0,
                 target_fake_label=0.0):
        """

        :param gan_mode: lsgan
        :param target_real_label:
        :param target_fake_label:
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        else:
            raise NotImplementedError('gan_mode [{}] not implemented'.format(gan_mode))

    def get_target_tensor(self, prediction, target_is_real):
        """
        Create label tensor
        :param prediction:
        :param target_is_real: true if label is for real images, false if for fakes
        :return:
        """
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """
        calculate the loss
        :param prediction:
        :param target_is_real:
        :return:
        """
        if self.gan_mode in ['lsgan']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        else:
            raise NotImplementedError('gan_mode [{}] not implemented'.format(self.gan_mode))
        return loss


class RhoClipper:
    def __init__(self, low, high):
        self.clip_low = low
        self.clip_high = high
        assert low <= high

    def __call__(self, module):
        if hasattr(module, 'rho'):
            w = module.rho.data
            w = w.clamp(self.clip_low, self.clip_high)
            module.rho.data = w


# def calc_mean_std(feat, eps=1e-5):
#     # eps is a small value added to the variance to avoid divide-by-zero.
#     size = feat.size()
#     assert (len(size) == 4)
#     N, C = size[:2]
#     feat_var = feat.view(N, C, -1).var(dim=2) + eps
#     feat_std = feat_var.sqrt().view(N, C, 1, 1)
#     feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
#     return feat_mean, feat_std

def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean) / content_std
    return normalized_feat * style_std + style_mean


class AdaInStyleTransfer(nn.Module):
    def __init__(self, encoder_path=None, decoder_path=None):
        super(AdaInStyleTransfer, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 3, (1, 1)),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(3, 64, (3, 3)),
            nn.ReLU(),  # relu1-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),  # relu1-2
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 128, (3, 3)),
            nn.ReLU(),  # relu2-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),  # relu2-2
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 256, (3, 3)),
            nn.ReLU(),  # relu3-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),  # relu3-2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),  # relu3-3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),  # relu3-4
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 512, (3, 3)),
            nn.ReLU(),  # relu4-1, this is the last layer used
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu4-2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu4-3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu4-4
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu5-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu5-2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu5-3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU()  # relu5-4
        )

        if encoder_path is not None:
            self.encoder.load_state_dict(torch.load(encoder_path))
            self.enc_1 = nn.Sequential(*list(self.encoder.children())[:4])
            self.enc_2 = nn.Sequential(*list(self.encoder.children())[4:11])
            self.enc_3 = nn.Sequential(*list(self.encoder.children())[11:18])
            self.enc_4 = nn.Sequential(*list(self.encoder.children())[18:31])
            for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
                for param in getattr(self, name).parameters():
                    param.requires_grad = False

        self.decoder = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 256, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 128, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 64, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 3, (3, 3)),
        )

        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

        if decoder_path is not None:
            self.decoder.load_state_dict(torch.load(decoder_path))
            # for param in getattr(self, 'decoder').parameters():
            #     param.requires_grad = False

        self.encoder_path = encoder_path
        self.decoder_path = decoder_path

    # extract relu1_1, relu2_1, relu3_1, relu4_1 from input image
    def encode(self, input):
        if self.encoder_path:
            results = [input]
            for i in range(4):
                func = getattr(self, 'enc_{:d}'.format(i + 1))
                results.append(func(results[-1]))
            return results[1:]
        else:
            result = self.encoder(input)
            return result

    def forward(self, content, style, alpha=1.0):
        assert 0 <= alpha <= 1
        style_feats = self.encode(style)
        content_feats = self.encode(content)

        t = adaptive_instance_normalization(content_feats[-1], style_feats[-1])
        t = alpha * t + (1 - alpha) * content_feats[-1]

        g_t = self.decoder(t)
        g_t_feats = self.encode(g_t)

        return style_feats, content_feats, g_t_feats, g_t, t
