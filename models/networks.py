import functools
import torch.nn as nn
import torch
from torch.nn import init
from torch.optim import lr_scheduler
from torch.nn import functional as F


###############################################################################
# Helper Functions
###############################################################################


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_scheduler(optimizer, opt):
    """
    Return a learning rate scheduler
    :param optimizer:
    :param opt: opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine
    :return:
    """
    """
    For linear, we keep the same learning rate for the first <iopt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        # lambda_rule = lambda epoch: 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
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

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
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
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

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
    def __init__(self, input_nc,
                 output_nc,
                 ngf=64,
                 norm_layer=None,
                 use_dropout=False,
                 n_blocks=6,
                 padding_type='reflect'):
        """
        Define a ResNet Block
        :param input_nc:
        :param output_nc:
        :param ngf:
        :param norm_layer:
        :param use_dropout:
        :param n_blocks: number of ResNet block
        :param padding_type: padding in conv: reflect | replicate | zero
        """
        assert n_blocks >= 0, 'n_blocks: {} need to satisfy >=0'.format(n_blocks)
        super(ResnetGenerator, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        """
        batch_norm has bias term already, use_bias only if we use instance_norm
        """

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
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True)
            ]
        mult = 2 ** n_downsampling
        for i in range(n_blocks):  # add ResNet blocks

            model += [
                ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                            use_dropout=use_dropout, use_bias=use_bias)
            ]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                   kernel_size=3, stride=2,
                                   padding=1, output_padding=1,
                                   bias=use_bias),
                norm_layer(int(ngf * mult / 2)),
                nn.ReLU(True)
            ]

        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


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
                torch.ones(x.shape[0], c, h, w) - self.rho.expand(x.shape[0], -1, -1, -1)) * out_ln
        out = out * gamma.unsqueeze(2).unsqueeze(3) + beta.unsqueeze(2).unsqueeze(3)
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
                torch.ones(x.shape[0], c, h, w) - self.rho.expand(x.shape[0], -1, -1, -1)) * out_ln
        out = out * self.gamma.expand(x.shape[0], -1, -1, -1) + + self.beta.expand(x.shape[0], -1, -1, -1)
        return out


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


class ResnetGeneratorUGATIT(nn.Module):
    def __init__(self,
                 input_nc,
                 output_nc,
                 ngf=64,
                 # norm_layer=nn.InstanceNorm2d,
                 n_blocks=6,
                 img_size=256,
                 light=False,
                 padding_type='reflect',
                 use_dropout=False):
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
            _x = self.fc(_x.view(_x.shape[0], -1))
        else:
            _x = self.fc(x.view(x.shape[0], -1))

        gamma, beta = self.gamma(_x), self.beta(_x)

        for i in range(self.n_blocks):
            x = getattr(self, 'UpBlock1_{}'.format(i + 1))(x, gamma, beta)
        out = self.UpBlock(x)

        return out, cam_logit, heatmap


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


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
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
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nc,
             output_nc,
             ngf,
             netG,
             norm='none',
             use_dropout=False,
             init_type='normal',
             init_gain=0.02,
             gpu_ids=None, ):
    """

    :param input_nc: number of channels in input_images
    :param output_nc: number of channels output_images
    :param ngf: number of filters in last conv_layer
    :param net_G: resnet_9blocks | resnet_6blocks | resnet_ugatit_6blocks
    :param norm: batch_norm | instance_norm | none
    :param use_dropout:
    :param init_type: initialize method
    :param init_gain: scaling factor for normal, xavier, orthogonal
    :param gpu_ids: (list or None) e.g., [0],[0, 1]
    :return:

    use ReLU for non-linearity
    """
    if gpu_ids is None:
        gpu_ids = []

    if norm == 'batch_norm':
        norm_layer = nn.BatchNorm2d
    elif norm == 'instance_norm':
        norm_layer = nn.InstanceNorm2d
    elif norm == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('norm_layer: {} is not implemented ... , '
                                  'use batch_norm or instance_norm instead'.format(norm))

    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == 'resnet_ugatit_6blocks':
        net = ResnetGeneratorUGATIT(input_nc, output_nc, ngf)
    else:
        raise NotImplementedError('Generator model name: {} is not recognized,'
                                  'use [resnet_9blocks, resnet_6blocks, resnet_ugatit_6blocks]'.format(netG))

    return init_net(net, init_type, init_gain, gpu_ids)


class NLayerDiscriminator(nn.Module):
    def __init__(self,
                 input_nc,
                 ndf=64,
                 n_layers=3,
                 norm_layer=None):
        """
        PatchGAN discriminator
        :param input_nc:
        :param ndf:
        :param n_layers:
        :param norm_layer:
        """
        super(NLayerDiscriminator, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kernel_size = 4
        padding = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kernel_size, stride=2, padding=padding),
            nn.LeakyReLU(0.2, True)
        ]
        nf = 1
        for i in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 8)
            sequence += [
                nn.Conv2d(ndf * nf_prev, ndf * nf, kernel_size=kernel_size, stride=2, padding=padding, bias=use_bias),
                norm_layer(ndf * nf),
                nn.LeakyReLU(0.2, True)
            ]
        nf_prev = nf
        nf = min(nf * 2, 8)
        sequence += [
            nn.Conv2d(ndf * nf_prev, ndf * nf, kernel_size=kernel_size, stride=1, padding=padding, bias=use_bias),
            norm_layer(ndf * nf),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * nf, 1, kernel_size=kernel_size, stride=1, padding=padding)
        ]
        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)


def define_D(input_nc,
             ndf,
             netD,
             n_layers=3,
             norm='batch_norm',
             init_type='normal',
             init_gain=0.02,
             gpu_ids=None):
    """
    Discriminator,  it uses leaky relu
    :param input_nc:
    :param ndf: number of filter in the first conv layer
    :param netD: basic | n_layers | ugatit
    :param n_layers_d:
    :param norm:
    :param use_dropout:
    :param init_type:
    :param init_gain:
    :param gpu_ids:
    :return:
    """

    if gpu_ids is None:
        gpu_ids = []

    if norm == 'batch_norm':
        norm_layer = nn.BatchNorm2d
    elif norm == 'instance_norm':
        norm_layer = nn.InstanceNorm2d
    elif norm == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('norm_layer: {} is not implemented ... , '
                                  'use batch_norm or instance_norm instead'.format(norm))

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers=n_layers, norm_layer=norm_layer)
    elif netD in ['ugatit', 'UGATIT']:
        if n_layers == 5:
            net = DiscriminatorUGATIT(input_nc, ndf, n_layers=5)
        elif n_layers == 7:
            net = DiscriminatorUGATIT(input_nc, ndf, n_layers=7)
        else:
            raise NotImplementedError('Discriminator in UGATIT supports n_layers 5 and 7 only,'
                                      ' you get {} instead'.format(n_layers))
    else:
        raise NotImplementedError('Discriminator model name [{}] is not recognized'.format(netD))
    return init_net(net, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
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
