import functools
import math

import torch.nn as nn
import torch
import torchvision
from torch.nn import init
from torch.optim import lr_scheduler
from torch.nn import functional as F
import numpy as np

###############################################################################
# Helper Functions
###############################################################################
from utils.util import calc_mean_std
from models.efficientnet import EfficientNet
from models.bifpn import BIFPN

class EqualLinear(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True, bias_init=0, lr_mul=1, activation=None):
        super(EqualLinear, self).__init__()
        self.weight = nn.Parameter(torch.randn(output_dim, input_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(output_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(input_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)

        else:
            out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )

        return out


class PixelNorm(nn.Module):
    def __init__(self):
        super(PixelNorm, self).__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


class Identity(nn.Module):
    def forward(self, x):
        return x


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
        def norm_layer(x):
            return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler
    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine
    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
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
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02,
             gpu_ids=[]):
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
        [n_layers]: With this mode, you can specify the number of conv layers in the discriminator
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
    elif netD == 'pixel':  # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################

class GANLoss(nn.Module):
    """Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.
        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028
    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss
    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':  # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(
                *real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp  # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.
    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect'):
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
        assert (n_blocks >= 0)
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
        for i in range(n_blocks):  # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias)]

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


class UpSkipAdaINBlock(nn.Module):
    def __init__(self, num_features, norm_layer='adaILN'):
        super(UpSkipAdaINBlock, self).__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.pad1 = nn.ReflectionPad2d(1)
        self.pad2 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=0, bias=False)
        if norm_layer == 'adaIN':
            self.norm1 = adaIN(num_features)
            self.norm2 = adaIN(num_features)
        elif norm_layer == 'adaILN':
            self.norm1 = adaILN(num_features)
            self.norm2 = adaILN(num_features)
        else:
            raise NotImplementedError('norm: [{}] is not implemented.'.format(norm_layer))

        self.relu1 = nn.ReLU(True)

        self.pad = nn.ReflectionPad2d(1)
        self.conv = nn.Conv2d(num_features + num_features // 2, num_features // 2, kernel_size=3, stride=1, padding=0,
                              bias=False)

        # self.norm = nn.InstanceNorm2d(num_features // 2)
        self.norm = ILN(num_features // 2)
        self.relu = nn.ReLU(True)

    def forward(self, x, gamma, beta, skip):
        out = self.pad1(x)
        out = self.conv1(out)
        out = self.norm1(out, gamma, beta)
        out = self.relu1(out)
        out = self.pad2(out)
        out = self.conv2(out)
        out = self.norm2(out, gamma, beta)

        # x = self.upsample(out + x) + skip
        x = torch.cat([self.upsample(out + x), skip], 1)
        x = self.pad(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)

        return x


class UStyleGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64):
        super(UStyleGenerator, self).__init__()

        # h,w,3 -> h,w,64
        downblock1 = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
        ]

        # h,w,64, h/2,w/2,128
        downblock2 = [

            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=0, bias=False),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(inplace=True),
        ]
        # h/2,w/2,128 => h/4,w/4,256
        downblock3 = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=0, bias=False),
            nn.InstanceNorm2d(ngf * 4),
            nn.ReLU(inplace=True),
        ]
        # h/4,w/4,256 => h/8,w/8,512
        downblock4 = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf * 4, ngf * 8, kernel_size=3, stride=2, padding=0, bias=False),
            nn.InstanceNorm2d(ngf * 8),
            nn.ReLU(inplace=True),
        ]
        # h/8,w/8,512 => h/16,w/16,1024
        downblock5 = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf * 8, ngf * 16, kernel_size=3, stride=2, padding=0, bias=False),
            nn.InstanceNorm2d(ngf * 16),
            nn.ReLU(inplace=True),
        ]

        self.downblock1 = nn.Sequential(*downblock1)
        self.downblock2 = nn.Sequential(*downblock2)
        self.downblock3 = nn.Sequential(*downblock3)
        self.downblock4 = nn.Sequential(*downblock4)
        self.downblock5 = nn.Sequential(*downblock5)

        # self.gap_fc_5 = nn.Linear(ngf * 16, 1, bias=False)
        # self.gap_fc_4 = nn.Linear(ngf * 8, 1, bias=False)
        # self.gap_fc_3 = nn.Linear(ngf * 4, 1, bias=False)
        # self.gap_fc_2 = nn.Linear(ngf * 2, 1, bias=False)
        #
        # self.gmp_fc_5 = nn.Linear(ngf * 16, 1, bias=False)
        # self.gmp_fc_4 = nn.Linear(ngf * 8, 1, bias=False)
        # self.gmp_fc_3 = nn.Linear(ngf * 4, 1, bias=False)
        # self.gmp_fc_2 = nn.Linear(ngf * 2, 1, bias=False)

        self.relu = nn.ReLU(True)

        for i in range(5, 1, -1):
            setattr(self, 'gap_fc_{}'.format(i), nn.Linear(ngf * 2 ** (i - 1), 1, bias=False))
            setattr(self, 'gmp_fc_{}'.format(i), nn.Linear(ngf * 2 ** (i - 1), 1, bias=False))

            setattr(self, 'conv1x1_{}'.format(i),
                    nn.Conv2d(ngf * 2 ** (i - 1) * 3, ngf * 2 ** (i - 1), kernel_size=1, stride=1, bias=True))
            setattr(self, 'fc{}'.format(i),
                    nn.Sequential(nn.Linear(ngf * 2 ** (i - 1), ngf * 2 ** (i - 1)), nn.ReLU(True)))
            setattr(self, 'gamma{}'.format(i), nn.Linear(ngf * 2 ** (i - 1), ngf * 2 ** (i - 1)))
            setattr(self, 'beta{}'.format(i), nn.Linear(ngf * 2 ** (i - 1), ngf * 2 ** (i - 1)))

            setattr(self, 'upblock{}'.format(i), UpSkipAdaINBlock(ngf * 2 ** (i - 1)))

        self.upblock = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, stride=1, padding=0, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        x1 = self.downblock1(x)
        x2 = self.downblock2(x1)
        x3 = self.downblock3(x2)
        x4 = self.downblock4(x3)
        x5 = self.downblock5(x4)

        # x5-gamp
        gap = torch.nn.functional.adaptive_avg_pool2d(x5, 1)
        gap_logit = self.gap_fc_5(gap.view(x5.shape[0], -1))
        gap_weight = list(self.gap_fc_5.parameters())[0]
        gap = x5 * gap_weight.unsqueeze(2).unsqueeze(3)

        gmp = torch.nn.functional.adaptive_max_pool2d(x5, 1)
        gmp_logit = self.gmp_fc_5(gmp.view(x5.shape[0], -1))
        gmp_weight = list(self.gmp_fc_5.parameters())[0]
        gmp = x5 * gmp_weight.unsqueeze(2).unsqueeze(3)

        cam_logit = torch.cat([gap_logit, gmp_logit], 1)
        x5 = torch.cat([gap, gmp, x5], 1)
        x5 = self.relu(self.conv1x1_5(x5))
        # x5-gamp

        # x4-gamp
        gap = torch.nn.functional.adaptive_avg_pool2d(x4, 1)
        gap_logit = self.gap_fc_4(gap.view(x4.shape[0], -1))
        gap_weight = list(self.gap_fc_4.parameters())[0]
        gap = x4 * gap_weight.unsqueeze(2).unsqueeze(3)

        gmp = torch.nn.functional.adaptive_max_pool2d(x4, 1)

        gmp_logit = self.gmp_fc_4(gmp.view(x4.shape[0], -1))
        gmp_weight = list(self.gmp_fc_4.parameters())[0]
        gmp = x4 * gmp_weight.unsqueeze(2).unsqueeze(3)

        cam_logit += torch.cat([gap_logit, gmp_logit], 1)
        x4 = torch.cat([gap, gmp, x4], 1)
        x4 = self.relu(self.conv1x1_4(x4))
        # x4-gamp

        # x3-gamp
        gap = torch.nn.functional.adaptive_avg_pool2d(x3, 1)
        gap_logit = self.gap_fc_3(gap.view(x3.shape[0], -1))
        gap_weight = list(self.gap_fc_3.parameters())[0]
        gap = x3 * gap_weight.unsqueeze(2).unsqueeze(3)

        gmp = torch.nn.functional.adaptive_max_pool2d(x3, 1)

        gmp_logit = self.gmp_fc_3(gmp.view(x3.shape[0], -1))
        gmp_weight = list(self.gmp_fc_3.parameters())[0]
        gmp = x3 * gmp_weight.unsqueeze(2).unsqueeze(3)

        cam_logit += torch.cat([gap_logit, gmp_logit], 1)
        x3 = torch.cat([gap, gmp, x3], 1)
        x3 = self.relu(self.conv1x1_3(x3))
        # x3-gamp

        # x2-gamp
        gap = torch.nn.functional.adaptive_avg_pool2d(x2, 1)
        gap_logit = self.gap_fc_2(gap.view(x2.shape[0], -1))
        gap_weight = list(self.gap_fc_2.parameters())[0]
        gap = x2 * gap_weight.unsqueeze(2).unsqueeze(2)

        gmp = torch.nn.functional.adaptive_max_pool2d(x2, 1)

        gmp_logit = self.gmp_fc_2(gmp.view(x2.shape[0], -1))
        gmp_weight = list(self.gmp_fc_2.parameters())[0]
        gmp = x2 * gmp_weight.unsqueeze(2).unsqueeze(2)

        cam_logit += torch.cat([gap_logit, gmp_logit], 1)
        x2 = torch.cat([gap, gmp, x2], 1)
        x2 = self.relu(self.conv1x1_2(x2))
        # x2-gamp

        # up5
        x5_ = torch.nn.functional.adaptive_avg_pool2d(x5, 1)
        x5_ = self.fc5(x5_.view(x5_.size(0), -1))
        gamma, beta = self.gamma5(x5_), self.beta5(x5_)
        x4 = self.upblock5(x5, gamma, beta, x4)
        # up5

        # up4
        x4_ = torch.nn.functional.adaptive_avg_pool2d(x4, 1)
        x4_ = self.fc4(x4_.view(x4_.size(0), -1))
        gamma, beta = self.gamma4(x4_), self.beta4(x4_)
        x3 = self.upblock4(x4, gamma, beta, x3)
        # up4

        # up3
        x3_ = torch.nn.functional.adaptive_avg_pool2d(x3, 1)
        x3_ = self.fc3(x3_.view(x3_.size(0), -1))
        gamma, beta = self.gamma3(x3_), self.beta3(x3_)
        x2 = self.upblock3(x3, gamma, beta, x2)
        # up3

        # up2
        x2_ = torch.nn.functional.adaptive_avg_pool2d(x2, 1)
        x2_ = self.fc2(x2_.view(x2_.size(0), -1))
        gamma, beta = self.gamma2(x2_), self.beta2(x2_)
        x1 = self.upblock2(x2, gamma, beta, x1)
        # up2

        out = self.upblock(x1)
        return out, cam_logit


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
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer,
                                             innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):  # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                 norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True,
                                             norm_layer=norm_layer)  # add the outermost layer

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
        else:  # add skip connections
            return torch.cat([x, self.model(x)], 1)


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

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)


class ResnetBlockUGATIT(nn.Module):
    def __init__(self, dim, use_bias):
        super(ResnetBlockUGATIT, self).__init__()
        conv_block = []
        conv_block += [nn.ReflectionPad2d(1),
                       nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias),
                       nn.InstanceNorm2d(dim),
                       nn.ReLU(True)]

        conv_block += [nn.ReflectionPad2d(1),
                       nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias),
                       nn.InstanceNorm2d(dim)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class adaIN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(adaIN, self).__init__()
        self.eps = eps

    def forward(self, x, gamma, beta):
        in_mean, in_var = torch.mean(x, dim=[2, 3], keepdim=True), torch.var(x, dim=[2, 3], keepdim=True)
        out_in = (x - in_mean) / torch.sqrt(in_var + self.eps)
        out = out_in * gamma.unsqueeze(2).unsqueeze(3) + beta.unsqueeze(
            2).unsqueeze(3)
        return out


class adaILN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(adaILN, self).__init__()
        self.eps = eps
        self.rho = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        self.rho.data.fill_(0.9)

    def forward(self, input, gamma, beta):
        in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(input, dim=[2, 3], keepdim=True)
        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(input, dim=[1, 2, 3], keepdim=True), torch.var(input, dim=[1, 2, 3], keepdim=True)
        out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)
        out = self.rho.expand(input.shape[0], -1, -1, -1) * out_in + (
                1 - self.rho.expand(input.shape[0], -1, -1, -1)) * out_ln
        out = out * gamma.unsqueeze(2).unsqueeze(3) + beta.unsqueeze(2).unsqueeze(3)

        return out


class ILN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(ILN, self).__init__()
        self.eps = eps
        self.rho = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        self.gamma = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        self.beta = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        self.rho.data.fill_(0.0)
        self.gamma.data.fill_(1.0)
        self.beta.data.fill_(0.0)

    def forward(self, input):
        in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(input, dim=[2, 3], keepdim=True)
        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(input, dim=[1, 2, 3], keepdim=True), torch.var(input, dim=[1, 2, 3], keepdim=True)
        out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)
        out = self.rho.expand(input.shape[0], -1, -1, -1) * out_in + (
                1 - self.rho.expand(input.shape[0], -1, -1, -1)) * out_ln
        out = out * self.gamma.expand(input.shape[0], -1, -1, -1) + self.beta.expand(input.shape[0], -1, -1, -1)

        return out


class ResnetAdaILNBlock(nn.Module):
    def __init__(self, dim, use_bias):
        super(ResnetAdaILNBlock, self).__init__()
        self.pad1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias)
        self.norm1 = adaILN(dim)
        self.relu1 = nn.ReLU(True)

        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias)
        self.norm2 = adaILN(dim)

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
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=6, img_size=256, light=False):
        assert (n_blocks >= 0)
        super(ResnetGeneratorUGATIT, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.n_blocks = n_blocks
        self.img_size = img_size
        self.light = light

        DownBlock = []
        DownBlock += [nn.ReflectionPad2d(3),
                      nn.Conv2d(input_nc, ngf, kernel_size=7, stride=1, padding=0, bias=False),
                      nn.InstanceNorm2d(ngf),
                      nn.ReLU(True)]

        # Down-Sampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            DownBlock += [nn.ReflectionPad2d(1),
                          nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=0, bias=False),
                          nn.InstanceNorm2d(ngf * mult * 2),
                          nn.ReLU(True)]

        # Down-Sampling Bottleneck
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            DownBlock += [ResnetBlockUGATIT(ngf * mult, use_bias=False)]

        # Class Activation Map
        self.gap_fc = nn.Linear(ngf * mult, 1, bias=False)
        self.gmp_fc = nn.Linear(ngf * mult, 1, bias=False)
        self.conv1x1 = nn.Conv2d(ngf * mult * 2, ngf * mult, kernel_size=1, stride=1, bias=True)
        self.relu = nn.ReLU(True)

        # Gamma, Beta block
        if self.light:
            FC = [nn.Linear(ngf * mult, ngf * mult, bias=False),
                  nn.ReLU(True),
                  nn.Linear(ngf * mult, ngf * mult, bias=False),
                  nn.ReLU(True)]
        else:
            FC = [nn.Linear(img_size // mult * img_size // mult * ngf * mult, ngf * mult, bias=False),
                  nn.ReLU(True),
                  nn.Linear(ngf * mult, ngf * mult, bias=False),
                  nn.ReLU(True)]
        self.gamma = nn.Linear(ngf * mult, ngf * mult, bias=False)
        self.beta = nn.Linear(ngf * mult, ngf * mult, bias=False)

        # Up-Sampling Bottleneck
        for i in range(n_blocks):
            setattr(self, 'UpBlock1_' + str(i + 1), ResnetAdaILNBlock(ngf * mult, use_bias=False))

        # Up-Sampling
        UpBlock2 = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            UpBlock2 += [nn.Upsample(scale_factor=2, mode='nearest'),
                         nn.ReflectionPad2d(1),
                         nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=0, bias=False),
                         ILN(int(ngf * mult / 2)),
                         nn.ReLU(True)]

        UpBlock2 += [nn.ReflectionPad2d(3),
                     nn.Conv2d(ngf, output_nc, kernel_size=7, stride=1, padding=0, bias=False),
                     nn.Tanh()]

        self.DownBlock = nn.Sequential(*DownBlock)
        self.FC = nn.Sequential(*FC)
        self.UpBlock2 = nn.Sequential(*UpBlock2)

    def forward(self, input):
        x = self.DownBlock(input)

        gap = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        gap_logit = self.gap_fc(gap.view(x.shape[0], -1))
        gap_weight = list(self.gap_fc.parameters())[0]
        gap = x * gap_weight.unsqueeze(2).unsqueeze(3)

        gmp = torch.nn.functional.adaptive_max_pool2d(x, 1)
        gmp_logit = self.gmp_fc(gmp.view(x.shape[0], -1))
        gmp_weight = list(self.gmp_fc.parameters())[0]
        gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)

        cam_logit = torch.cat([gap_logit, gmp_logit], 1)
        x = torch.cat([gap, gmp], 1)
        x = self.relu(self.conv1x1(x))

        heatmap = torch.sum(x, dim=1, keepdim=True)

        if self.light:
            x_ = torch.nn.functional.adaptive_avg_pool2d(x, 1)
            x_ = self.FC(x_.view(x_.shape[0], -1))
        else:
            x_ = self.FC(x.view(x.shape[0], -1))
        gamma, beta = self.gamma(x_), self.beta(x_)

        for i in range(self.n_blocks):
            x = getattr(self, 'UpBlock1_' + str(i + 1))(x, gamma, beta)
        out = self.UpBlock2(x)

        return out, cam_logit, heatmap


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


class NICEDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=7):
        super(NICEDiscriminator, self).__init__()

        model = [nn.ReflectionPad2d(1),
                 nn.utils.spectral_norm(
                     nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=0, bias=True)),
                 nn.LeakyReLU(0.2, True)]  # 1+3*2^0 =4

        for i in range(1, 2):  # 1+3*2^0 + 3*2^1 =10
            mult = 2 ** (i - 1)
            model += [nn.ReflectionPad2d(1),
                      nn.utils.spectral_norm(
                          nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=2, padding=0, bias=True)),
                      nn.LeakyReLU(0.2, True)]

            # Class Activation Map
        mult = 2 ** (1)
        self.fc = nn.utils.spectral_norm(nn.Linear(ndf * mult * 2, 1, bias=False))
        self.conv1x1 = nn.Conv2d(ndf * mult * 2, ndf * mult, kernel_size=1, stride=1, bias=True)
        self.leaky_relu = nn.LeakyReLU(0.2, True)
        self.lamda = nn.Parameter(torch.zeros(1))

        Dis0_0 = []
        for i in range(2, n_layers - 4):  # 1+3*2^0 + 3*2^1 + 3*2^2 =22
            mult = 2 ** (i - 1)
            Dis0_0 += [nn.ReflectionPad2d(1),
                       nn.utils.spectral_norm(
                           nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=2, padding=0, bias=True)),
                       nn.LeakyReLU(0.2, True)]

        mult = 2 ** (n_layers - 4 - 1)
        Dis0_1 = [nn.ReflectionPad2d(1),  # 1+3*2^0 + 3*2^1 + 3*2^2 +3*2^3 = 46
                  nn.utils.spectral_norm(
                      nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=1, padding=0, bias=True)),
                  nn.LeakyReLU(0.2, True)]
        mult = 2 ** (n_layers - 4)
        self.conv0 = nn.utils.spectral_norm(  # 1+3*2^0 + 3*2^1 + 3*2^2 +3*2^3 + 3*2^3= 70
            nn.Conv2d(ndf * mult, 1, kernel_size=4, stride=1, padding=0, bias=False))

        Dis1_0 = []
        for i in range(n_layers - 4,
                       n_layers - 2):  # 1+3*2^0 + 3*2^1 + 3*2^2 + 3*2^3=46, 1+3*2^0 + 3*2^1 + 3*2^2 +3*2^3 +3*2^4 = 190
            mult = 2 ** (i - 1)
            Dis1_0 += [nn.ReflectionPad2d(1),
                       nn.utils.spectral_norm(
                           nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=2, padding=0, bias=True)),
                       nn.LeakyReLU(0.2, True)]

        mult = 2 ** (n_layers - 2 - 1)
        Dis1_1 = [nn.ReflectionPad2d(1),  # 1+3*2^0 + 3*2^1 + 3*2^2 +3*2^3 +3*2^4 + 3*2^5= 190 + 96 = 190
                  nn.utils.spectral_norm(
                      nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=1, padding=0, bias=True)),
                  nn.LeakyReLU(0.2, True)]
        mult = 2 ** (n_layers - 2)
        self.conv1 = nn.utils.spectral_norm(  # 1+3*2^0 + 3*2^1 + 3*2^2 +3*2^3 +3*2^4 + 3*2^5 + 3*2^5 = 286
            nn.Conv2d(ndf * mult, 1, kernel_size=4, stride=1, padding=0, bias=False))

        # self.attn = Self_Attn( ndf * mult)
        self.pad = nn.ReflectionPad2d(1)

        self.model = nn.Sequential(*model)
        self.Dis0_0 = nn.Sequential(*Dis0_0)
        self.Dis0_1 = nn.Sequential(*Dis0_1)
        self.Dis1_0 = nn.Sequential(*Dis1_0)
        self.Dis1_1 = nn.Sequential(*Dis1_1)

    def forward(self, input):

        # TODO add cam to 10x10, 70x70, 286x286?

        x = self.model(input)

        x_0 = x

        gap = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        gmp = torch.nn.functional.adaptive_max_pool2d(x, 1)
        x = torch.cat([x, x], 1)
        cam_logit = torch.cat([gap, gmp], 1)
        cam_logit = self.fc(cam_logit.view(cam_logit.shape[0], -1))
        weight = list(self.fc.parameters())[0]
        x = x * weight.unsqueeze(2).unsqueeze(3)
        x = self.conv1x1(x)

        x = self.lamda * x + x_0
        # print("lamda:",self.lamda)

        x = self.leaky_relu(x)

        heatmap = torch.sum(x, dim=1, keepdim=True)

        z = x

        x0 = self.Dis0_0(x)
        x1 = self.Dis1_0(x0)
        x0 = self.Dis0_1(x0)
        x1 = self.Dis1_1(x1)
        x0 = self.pad(x0)
        x1 = self.pad(x1)
        out0 = self.conv0(x0)
        out1 = self.conv1(x1)

        return out0, out1, cam_logit, heatmap, z


class NICESADiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=7):
        super(NICESADiscriminator, self).__init__()

        # 3, h, w => 64, h/2, w/2, rf=4
        model = [nn.ReflectionPad2d(1),
                 nn.utils.spectral_norm(
                     nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=0, bias=True)),
                 nn.LeakyReLU(0.2, True)]
        # 64, h/2, w/2 => 128, h/4, w/4
        model += [nn.ReflectionPad2d(1),
                  nn.utils.spectral_norm(
                      nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=0, bias=True)),
                  nn.LeakyReLU(0.2, True)]

        # SA module

        self.query_conv = nn.utils.spectral_norm(nn.Conv2d(ndf * 2, ndf * 2, kernel_size=1, stride=1, bias=False))
        self.key_conv = nn.utils.spectral_norm(nn.Conv2d(ndf * 2, ndf * 2, kernel_size=1, stride=1, bias=False))
        self.value_conv = nn.utils.spectral_norm(nn.Conv2d(ndf * 2, ndf * 2, kernel_size=1, stride=1, bias=False))

        self.conv1x1 = nn.utils.spectral_norm(nn.Conv2d(ndf * 2, ndf * 2, kernel_size=1, stride=1, bias=True))
        self.leaky_relu = nn.LeakyReLU(0.2, True)

        self.softmax = nn.Softmax(dim=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        Dis0_0 = []
        for i in range(2, n_layers - 4):  # 1+3*2^0 + 3*2^1 + 3*2^2 =22
            mult = 2 ** (i - 1)
            Dis0_0 += [nn.ReflectionPad2d(1),
                       nn.utils.spectral_norm(
                           nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=2, padding=0, bias=True)),
                       nn.LeakyReLU(0.2, True)]

        mult = 2 ** (n_layers - 4 - 1)
        Dis0_1 = [nn.ReflectionPad2d(1),  # 1+3*2^0 + 3*2^1 + 3*2^2 +3*2^3 = 46
                  nn.utils.spectral_norm(
                      nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=1, padding=0, bias=True)),
                  nn.LeakyReLU(0.2, True)]
        mult = 2 ** (n_layers - 4)
        self.conv0 = nn.utils.spectral_norm(  # 1+3*2^0 + 3*2^1 + 3*2^2 +3*2^3 + 3*2^3= 70
            nn.Conv2d(ndf * mult, 1, kernel_size=4, stride=1, padding=0, bias=False))

        Dis1_0 = []
        for i in range(n_layers - 4,
                       n_layers - 2):  # 1+3*2^0 + 3*2^1 + 3*2^2 + 3*2^3=46, 1+3*2^0 + 3*2^1 + 3*2^2 +3*2^3 +3*2^4 = 190
            mult = 2 ** (i - 1)
            Dis1_0 += [nn.ReflectionPad2d(1),
                       nn.utils.spectral_norm(
                           nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=2, padding=0, bias=True)),
                       nn.LeakyReLU(0.2, True)]

        mult = 2 ** (n_layers - 2 - 1)
        Dis1_1 = [nn.ReflectionPad2d(1),  # 1+3*2^0 + 3*2^1 + 3*2^2 +3*2^3 +3*2^4 + 3*2^5= 190 + 96 = 190
                  nn.utils.spectral_norm(
                      nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=1, padding=0, bias=True)),
                  nn.LeakyReLU(0.2, True)]
        mult = 2 ** (n_layers - 2)
        self.conv1 = nn.utils.spectral_norm(  # 1+3*2^0 + 3*2^1 + 3*2^2 +3*2^3 +3*2^4 + 3*2^5 + 3*2^5 = 286
            nn.Conv2d(ndf * mult, 1, kernel_size=4, stride=1, padding=0, bias=False))

        # self.attn = Self_Attn( ndf * mult)
        self.pad = nn.ReflectionPad2d(1)

        self.model = nn.Sequential(*model)
        self.Dis0_0 = nn.Sequential(*Dis0_0)
        self.Dis0_1 = nn.Sequential(*Dis0_1)
        self.Dis1_0 = nn.Sequential(*Dis1_0)
        self.Dis1_1 = nn.Sequential(*Dis1_1)



    def forward(self, x):
        x = self.model(x)

        x_0 = x
        # x = (1, 128, h/4, w/4)
        b, c, h, w = x.size()
        proj_query = self.query_conv(x).view(b, c, -1).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(b, c, -1)

        # energy = (b, hw, hw)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(b, c, -1)
        x = torch.bmm(proj_value, attention).view(b, c, h, w)
        x = self.leaky_relu(self.conv1x1(x))
        x = self.gamma * x + x_0
        heatmap = torch.sum(x, dim=1, keepdim=True)

        z = x

        x0 = self.Dis0_0(x)
        x1 = self.Dis1_0(x0)
        x0 = self.Dis0_1(x0)
        x1 = self.Dis1_1(x1)
        x0 = self.pad(x0)
        x1 = self.pad(x1)
        out0 = self.conv0(x0)
        out1 = self.conv1(x1)

        return out0, out1, heatmap, z

class EfficientDetGenerator(nn.Module):

    def __init__(self, output_nc, bifpn_in_channels, bifpn_out_channels, num_stacks, num_outs):
        super(EfficientDetGenerator, self).__init__()
        
        self.model = BIFPN(in_channels=bifpn_in_channels,
                      out_channels=bifpn_out_channels,
                      stack=num_stacks,
                      num_outs=num_outs)
        # 1 224 32 32
        # 1 112 64 64
        # 1 56 128 128
        # 1 28 256 256
        self.upblock = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(224, 56, kernel_size=3, stride=1, padding=0, bias=False),
            ILN(56),
            nn.ReLU(True),
            nn.Conv2d(56, 112 * 4, kernel_size=1, stride=1, padding=0, bias=True),
            nn.PixelShuffle(2),
            ILN(112),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(112, 28, kernel_size=3, stride=1, padding=0, bias=False),
            ILN(28),
            nn.ReLU(True),
            nn.Conv2d(28, 56 * 4, kernel_size=1, stride=1, padding=0, bias=True),
            nn.PixelShuffle(2),
            ILN(56),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(56, 14, kernel_size=3, stride=1, padding=0, bias=False),
            ILN(14),
            nn.ReLU(True),
            nn.Conv2d(14, 28 * 4, kernel_size=1, stride=1, padding=0, bias=True),
            nn.PixelShuffle(2),
            ILN(28),
            nn.ReLU(True),

            nn.Conv2d(28, 64, kernel_size=1, stride=1, padding=0, bias=False),

            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_nc, kernel_size=7, stride=1, padding=0, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        outs = self.model(x)
        # torch.Size([1, 112, 32, 32])
        # torch.Size([1, 112, 16, 16])
        # torch.Size([1, 112, 8, 8])
        # torch.Size([1, 112, 4, 4])
        # torch.Size([1, 112, 2, 2])
        out = self.upblock(outs[0])

        return out
    
class EfficientDetDiscrminator(nn.Module):
    # EfficientDet Discriminator
    def __init__(self, network='efficientnet-b4',):
        super(EfficientDetDiscrminator, self).__init__()
        self.backbone = EfficientNet.from_pretrained(network)
        
        self.fc = nn.Linear(1000, 1)
        
    def forward(self, x):
        feats, x = self.backbone(x)
        x = self.fc(x)
        return feats[-5:], x

class ENSADiscriminator(nn.Module):
    # Efficient Nice Self Attention Discriminator
    # ENSAD
    def __init__(self, input_nc, ndf=64, n_layers=7):
        super(ENSADiscriminator, self).__init__()

        # 3, h, w => 64, h/2, w/2, rf=4
        model = [nn.ReflectionPad2d(1),
                 nn.utils.spectral_norm(
                     nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=0, bias=True)),
                 nn.LeakyReLU(0.2, True)]
        # 64, h/2, w/2 => 128, h/4, w/4
        model += [nn.ReflectionPad2d(1),
                  nn.utils.spectral_norm(
                      nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=0, bias=True)),
                  nn.LeakyReLU(0.2, True)]

        # SA module

        self.query_conv = nn.utils.spectral_norm(nn.Conv2d(ndf * 2, ndf * 2, kernel_size=1, stride=1, bias=False))
        self.key_conv = nn.utils.spectral_norm(nn.Conv2d(ndf * 2, ndf * 2, kernel_size=1, stride=1, bias=False))
        self.value_conv = nn.utils.spectral_norm(nn.Conv2d(ndf * 2, ndf * 2, kernel_size=1, stride=1, bias=False))

        self.conv1x1 = nn.utils.spectral_norm(nn.Conv2d(ndf * 2, ndf * 2, kernel_size=1, stride=1, bias=True))
        self.leaky_relu = nn.LeakyReLU(0.2, True)

        self.softmax = nn.Softmax(dim=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        Dis0_0 = []
        for i in range(2, n_layers - 4):  # 1+3*2^0 + 3*2^1 + 3*2^2 =22
            mult = 2 ** (i - 1)
            Dis0_0 += [nn.ReflectionPad2d(1),
                       nn.utils.spectral_norm(
                           nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=2, padding=0, bias=True)),
                       nn.LeakyReLU(0.2, True)]

        mult = 2 ** (n_layers - 4 - 1)
        Dis0_1 = [nn.ReflectionPad2d(1),  # 1+3*2^0 + 3*2^1 + 3*2^2 +3*2^3 = 46
                  nn.utils.spectral_norm(
                      nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=1, padding=0, bias=True)),
                  nn.LeakyReLU(0.2, True)]
        mult = 2 ** (n_layers - 4)
        self.conv0 = nn.utils.spectral_norm(  # 1+3*2^0 + 3*2^1 + 3*2^2 +3*2^3 + 3*2^3= 70
            nn.Conv2d(ndf * mult, 1, kernel_size=4, stride=1, padding=0, bias=False))

        Dis1_0 = []
        for i in range(n_layers - 4,
                       n_layers - 2):  # 1+3*2^0 + 3*2^1 + 3*2^2 + 3*2^3=46, 1+3*2^0 + 3*2^1 + 3*2^2 +3*2^3 +3*2^4 = 190
            mult = 2 ** (i - 1)
            Dis1_0 += [nn.ReflectionPad2d(1),
                       nn.utils.spectral_norm(
                           nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=2, padding=0, bias=True)),
                       nn.LeakyReLU(0.2, True)]

        mult = 2 ** (n_layers - 2 - 1)
        Dis1_1 = [nn.ReflectionPad2d(1),  # 1+3*2^0 + 3*2^1 + 3*2^2 +3*2^3 +3*2^4 + 3*2^5= 190 + 96 = 190
                  nn.utils.spectral_norm(
                      nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=1, padding=0, bias=True)),
                  nn.LeakyReLU(0.2, True)]
        mult = 2 ** (n_layers - 2)
        self.conv1 = nn.utils.spectral_norm(  # 1+3*2^0 + 3*2^1 + 3*2^2 +3*2^3 +3*2^4 + 3*2^5 + 3*2^5 = 286
            nn.Conv2d(ndf * mult, 1, kernel_size=4, stride=1, padding=0, bias=False))

        # self.attn = Self_Attn( ndf * mult)
        self.pad = nn.ReflectionPad2d(1)

        self.model = nn.Sequential(*model)
        self.Dis0_0 = nn.Sequential(*Dis0_0)
        self.Dis0_1 = nn.Sequential(*Dis0_1)
        self.Dis1_0 = nn.Sequential(*Dis1_0)
        self.Dis1_1 = nn.Sequential(*Dis1_1)



    def forward(self, x):
        x = self.model(x)

        x_0 = x
        # x = (1, 128, h/4, w/4)
        b, c, h, w = x.size()
        proj_query = self.query_conv(x).view(b, c, -1).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(b, c, -1)

        # energy = (b, hw, hw)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(b, c, -1)
        x = torch.bmm(proj_value, attention).view(b, c, h, w)
        x = self.leaky_relu(self.conv1x1(x))
        x = self.gamma * x + x_0
        heatmap = torch.sum(x, dim=1, keepdim=True)

        z = x

        x0 = self.Dis0_0(x)
        x1 = self.Dis1_0(x0)
        x0 = self.Dis0_1(x0)
        x1 = self.Dis1_1(x1)
        x0 = self.pad(x0)
        x1 = self.pad(x1)
        out0 = self.conv0(x0)
        out1 = self.conv1(x1)

        return out0, out1, heatmap, z

class DiscriminatorUGATIT(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=5):
        super(DiscriminatorUGATIT, self).__init__()

        # rf:4
        model = [nn.ReflectionPad2d(1),
                 nn.utils.spectral_norm(
                     nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=0, bias=True)),
                 nn.LeakyReLU(0.2, True)]
        # local rf: 10, 22
        # global rf: 10, 22, 46, 190
        for i in range(1, n_layers - 2):
            mult = 2 ** (i - 1)
            model += [nn.ReflectionPad2d(1),
                      nn.utils.spectral_norm(
                          nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=2, padding=0, bias=True)),
                      nn.LeakyReLU(0.2, True)]

        mult = 2 ** (n_layers - 2 - 1)
        # local rf: 25
        # global rf: 97
        model += [nn.ReflectionPad2d(1),
                  nn.utils.spectral_norm(
                      nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=1, padding=0, bias=True)),
                  nn.LeakyReLU(0.2, True)]

        # Class Activation Map
        mult = 2 ** (n_layers - 2)
        self.gap_fc = nn.utils.spectral_norm(nn.Linear(ndf * mult, 1, bias=False))
        self.gmp_fc = nn.utils.spectral_norm(nn.Linear(ndf * mult, 1, bias=False))
        self.conv1x1 = nn.Conv2d(ndf * mult * 2, ndf * mult, kernel_size=1, stride=1, bias=True)
        self.leaky_relu = nn.LeakyReLU(0.2, True)

        # local rf: 28
        # global rf: 100
        self.pad = nn.ReflectionPad2d(1)
        self.conv = nn.utils.spectral_norm(
            nn.Conv2d(ndf * mult, 1, kernel_size=4, stride=1, padding=0, bias=False))

        self.model = nn.Sequential(*model)

    def forward(self, input):
        x = self.model(input)

        gap = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        gap_logit = self.gap_fc(gap.view(x.shape[0], -1))
        gap_weight = list(self.gap_fc.parameters())[0]
        gap = x * gap_weight.unsqueeze(2).unsqueeze(3)

        gmp = torch.nn.functional.adaptive_max_pool2d(x, 1)
        gmp_logit = self.gmp_fc(gmp.view(x.shape[0], -1))
        gmp_weight = list(self.gmp_fc.parameters())[0]
        gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)

        cam_logit = torch.cat([gap_logit, gmp_logit], 1)
        x = torch.cat([gap, gmp], 1)
        x = self.leaky_relu(self.conv1x1(x))

        heatmap = torch.sum(x, dim=1, keepdim=True)

        x = self.pad(x)
        out = self.conv(x)

        return out, cam_logit, heatmap


# class VGG19(nn.Module):
#     def __init__(self, num_classes=1000):
#         super(VGG19, self).__init__()
#
#         layers = []
#         in_channels = 3
#         for v in [64, 64, 'M',
#                   128, 128, 'M',
#                   256, 256, 256, 256, 'M',
#                   512, 512, 512, 512, 'M',
#                   512, 512, 512, 512, 'M']:
#             if v == 'M':
#                 layers += [
#                     nn.MaxPool2d(kernel_size=2, stride=2)
#                 ]
#             else:
#                 layers += [
#                     nn.Conv2d(in_channels, v, kernel_size=3, padding=1),
#                     nn.BatchNorm2d(v),
#                     nn.ReLU(inplace=True)
#                 ]
#                 in_channels = v
#
#         self.features = nn.Sequential(*layers)
#         self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
#         self.classifier = nn.Sequential(
#             nn.Linear(512 * 7 * 7, 4096),
#             nn.ReLU(True),
#             nn.Dropout(),
#             nn.Linear(4096, 4096),
#             nn.ReLU(True),
#             nn.Dropout(),
#             nn.Linear(4096, num_classes),
#         )
#
#         self.load_state_dict(torch.load('models/vgg19_bn-c719001a0.pth'))
#
#         self.enc_1 = nn.Sequential(*list(self.features.children())[:3])  # input => relu 1-1
#         self.enc_2 = nn.Sequential(*list(self.features.children())[3:10])  # relu 1-1 => relu 2-1
#         self.enc_3 = nn.Sequential(*list(self.features.children())[10:17])  # relu 2-1 => relu 3-1
#         self.enc_4 = nn.Sequential(*list(self.features.children())[17:30])  # relu 3-1 => relu 4-1
#         for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
#             for param in getattr(self, name).parameters():
#                 param.requires_grad = False
#
#     def forward(self, input):
#
#         # b, _, _, _ = input.size()
#         # m = torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
#         # for i in range(b):
#         #     input[i] = m(input[i])
#
#         results = [input]
#         for i in range(4):
#             func = getattr(self, 'enc_{:d}'.format(i + 1))
#             results.append(func(results[-1]))
#         return results[1:]


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


# class ModulatedConv2d(nn.Module):
#     def __init__(
#             self,
#             in_channel,
#             out_channel,
#             kernel_size,
#             style_dim,
#             demodulate=True,
#             upsample=False,
#             downsample=False,
#             blur_kernel=[1, 3, 3, 1],
#     ):
#         super().__init__()
#
#         self.eps = 1e-8
#         self.kernel_size = kernel_size
#         self.in_channel = in_channel
#         self.out_channel = out_channel
#         self.upsample = upsample
#         self.downsample = downsample
#
#         if upsample:
#             factor = 2
#             p = (len(blur_kernel) - factor) - (kernel_size - 1)
#             pad0 = (p + 1) // 2 + factor - 1
#             pad1 = p // 2 + 1
#
#             self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)
#
#         if downsample:
#             factor = 2
#             p = (len(blur_kernel) - factor) + (kernel_size - 1)
#             pad0 = (p + 1) // 2
#             pad1 = p // 2
#
#             self.blur = Blur(blur_kernel, pad=(pad0, pad1))
#
#         fan_in = in_channel * kernel_size ** 2
#         self.scale = 1 / math.sqrt(fan_in)
#         self.padding = kernel_size // 2
#
#         self.weight = nn.Parameter(
#             torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
#         )
#
#         self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)
#
#         self.demodulate = demodulate
#
#     def __repr__(self):
#         return (
#             f'{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, '
#             f'upsample={self.upsample}, downsample={self.downsample})'
#         )
#
#     def forward(self, input, style):
#         batch, in_channel, height, width = input.shape
#
#         style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
#         weight = self.scale * self.weight * style
#
#         if self.demodulate:
#             demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
#             weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)
#
#         weight = weight.view(
#             batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
#         )
#
#         if self.upsample:
#             input = input.view(1, batch * in_channel, height, width)
#             weight = weight.view(
#                 batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
#             )
#             weight = weight.transpose(1, 2).reshape(
#                 batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
#             )
#             out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
#             _, _, height, width = out.shape
#             out = out.view(batch, self.out_channel, height, width)
#             out = self.blur(out)
#
#         elif self.downsample:
#             input = self.blur(input)
#             _, _, height, width = input.shape
#             input = input.view(1, batch * in_channel, height, width)
#             out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
#             _, _, height, width = out.shape
#             out = out.view(batch, self.out_channel, height, width)
#
#         else:
#             input = input.view(1, batch * in_channel, height, width)
#             out = F.conv2d(input, weight, padding=self.padding, groups=batch)
#             _, _, height, width = out.shape
#             out = out.view(batch, self.out_channel, height, width)
#
#         return out


class NICEResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=6, img_size=256, light=False):
        assert (n_blocks >= 0)
        super(NICEResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.n_blocks = n_blocks
        self.img_size = img_size
        self.light = light

        n_downsampling = 2

        mult = 2 ** n_downsampling
        UpBlock0 = [nn.ReflectionPad2d(1),
                    nn.Conv2d(int(ngf * mult / 2), ngf * mult, kernel_size=3, stride=1, padding=0, bias=True),
                    ILN(ngf * mult),
                    nn.ReLU(True)]

        self.relu = nn.ReLU(True)

        # Gamma, Beta block
        if self.light:
            FC = [nn.Linear(ngf * mult, ngf * mult, bias=False),
                  nn.ReLU(True),
                  nn.Linear(ngf * mult, ngf * mult, bias=False),
                  nn.ReLU(True)]
        else:
            FC = [nn.Linear(img_size // mult * img_size // mult * ngf * mult, ngf * mult, bias=False),
                  nn.ReLU(True),
                  nn.Linear(ngf * mult, ngf * mult, bias=False),
                  nn.ReLU(True)]
        self.gamma = nn.Linear(ngf * mult, ngf * mult, bias=False)
        self.beta = nn.Linear(ngf * mult, ngf * mult, bias=False)

        # Up-Sampling Bottleneck
        for i in range(n_blocks):
            setattr(self, 'UpBlock1_' + str(i + 1), ResnetAdaILNBlock(ngf * mult, use_bias=False))

        # Up-Sampling
        UpBlock2 = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            # Experiments show that the performance of Up-sample and Sub-pixel is similar,
            #  although theoretically Sub-pixel has more parameters and less FLOPs.
            # UpBlock2 += [nn.Upsample(scale_factor=2, mode='nearest'),
            #              nn.ReflectionPad2d(1),
            #              nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=0, bias=False),
            #              ILN(int(ngf * mult / 2)),
            #              nn.ReLU(True)]
            UpBlock2 += [nn.ReflectionPad2d(1),
                         nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=0, bias=False),
                         ILN(int(ngf * mult / 2)),
                         nn.ReLU(True),
                         nn.Conv2d(int(ngf * mult / 2), int(ngf * mult / 2) * 4, kernel_size=1, stride=1, bias=True),
                         nn.PixelShuffle(2),
                         ILN(int(ngf * mult / 2)),
                         nn.ReLU(True)
                         ]

        UpBlock2 += [nn.ReflectionPad2d(3),
                     nn.Conv2d(ngf, output_nc, kernel_size=7, stride=1, padding=0, bias=False),
                     nn.Tanh()]

        self.FC = nn.Sequential(*FC)
        self.UpBlock0 = nn.Sequential(*UpBlock0)
        self.UpBlock2 = nn.Sequential(*UpBlock2)

    def forward(self, z):
        x = z
        x = self.UpBlock0(x)

        if self.light:
            x_ = torch.nn.functional.adaptive_avg_pool2d(x, 1)
            x_ = self.FC(x_.view(x_.shape[0], -1))
        else:
            x_ = self.FC(x.view(x.shape[0], -1))
        gamma, beta = self.gamma(x_), self.beta(x_)

        for i in range(self.n_blocks):
            x = getattr(self, 'UpBlock1_' + str(i + 1))(x, gamma, beta)

        out = self.UpBlock2(x)

        return out


class NICEV2ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=6, img_size=256):
        assert (n_blocks >= 0)
        super(NICEV2ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.n_blocks = n_blocks
        self.img_size = img_size

        n_downsampling = 2

        mult = 2 ** n_downsampling
        UpBlock0 = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(int(ngf * mult / 2), ngf * mult, kernel_size=3, stride=1, padding=0, bias=True),
            ILN(ngf * mult),
            nn.ReLU(True)
        ]

        self.relu = nn.ReLU(True)

        # # Gamma, Beta block
        # if self.light:
        #     FC = [nn.Linear(ngf * mult, ngf * mult, bias=False),
        #           nn.ReLU(True),
        #           nn.Linear(ngf * mult, ngf * mult, bias=False),
        #           nn.ReLU(True)]
        # else:
        #     FC = [nn.Linear(img_size // mult * img_size // mult * ngf * mult, ngf * mult, bias=False),
        #           nn.ReLU(True),
        #           nn.Linear(ngf * mult, ngf * mult, bias=False),
        #           nn.ReLU(True)]

        conv_style = [
            nn.Conv2d(ngf * mult, ngf * mult, kernel_size=img_size // mult, stride=1, padding=0,
                      groups=img_size // mult, bias=False),
            nn.ReLU(True),
        ]
        fc_style = [
            nn.Linear(ngf * mult, ngf * mult, bias=False),
            nn.ReLU(True),
            nn.Linear(ngf * mult, ngf * mult, bias=False),
            nn.ReLU(True),
        ]

        self.gamma = nn.Linear(ngf * mult, ngf * mult, bias=False)
        self.beta = nn.Linear(ngf * mult, ngf * mult, bias=False)

        # Up-Sampling Bottleneck
        for i in range(n_blocks):
            setattr(self, 'UpBlock1_' + str(i + 1), ResnetAdaILNBlock(ngf * mult, use_bias=False))

        # Up-Sampling
        UpBlock2 = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            # Experiments show that the performance of Up-sample and Sub-pixel is similar,
            #  although theoretically Sub-pixel has more parameters and less FLOPs.
            UpBlock2 += [nn.Upsample(scale_factor=2, mode='nearest'),
                         nn.ReflectionPad2d(1),
                         nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=0, bias=False),
                         ILN(int(ngf * mult / 2)),
                         nn.ReLU(True)]
            # UpBlock2 += [nn.ReflectionPad2d(1),
            #              nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=0, bias=False),
            #              ILN(int(ngf * mult / 2)),
            #              nn.ReLU(True),
            #              nn.Conv2d(int(ngf * mult / 2), int(ngf * mult / 2) * 4, kernel_size=1, stride=1, bias=True),
            #              nn.PixelShuffle(2),
            #              ILN(int(ngf * mult / 2)),
            #              nn.ReLU(True)
            #              ]

        UpBlock2 += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, stride=1, padding=0, bias=False),
            nn.Tanh()
        ]

        # self.FC = nn.Sequential(*FC)
        self.conv_style = nn.Sequential(*conv_style)
        self.fc_style = nn.Sequential(*fc_style)
        self.UpBlock0 = nn.Sequential(*UpBlock0)
        self.UpBlock2 = nn.Sequential(*UpBlock2)

    def forward(self, z):
        x = z
        x = self.UpBlock0(x)

        # if self.light:
        #     x_ = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        #     x_ = self.FC(x_.view(x_.shape[0], -1))
        # else:
        #     x_ = self.FC(x.view(x.shape[0], -1))

        x_ = self.fc_style(self.conv_style(x).view(x.size(0), -1))
        gamma, beta = self.gamma(x_), self.beta(x_)

        for i in range(self.n_blocks):
            x = getattr(self, 'UpBlock1_' + str(i + 1))(x, gamma, beta)

        out = self.UpBlock2(x)

        return out


class NICE3SResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=4):
        super(NICE3SResnetGenerator, self).__init__()
        # self.input_nc = input_nc
        # self.output_nc = output_nc
        # self.ngf = ngf
        self.n_blocks = n_blocks
        # self.img_size = img_size
        # self.light = light

        # h/64, w/64, 2048 = > h/32, w/32, 1024
        upblock0_190x = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf * 32, ngf * 16, kernel_size=3, stride=1, padding=0, bias=False),
            ILN(ngf * 16),
            nn.ReLU(True),
            nn.Conv2d(ngf * 16, ngf * 16 * 4, kernel_size=1, stride=1, padding=0, bias=True),
            nn.PixelShuffle(2),
            ILN(ngf * 16),
            nn.ReLU(True)
        ]

        fc190 = [
            nn.Linear(ngf * 16, ngf * 16, bias=False),
            nn.ReLU(True),
            nn.Linear(ngf * 16, ngf * 16, bias=False),
            nn.ReLU(True),
        ]

        self.fc190 = nn.Sequential(*fc190)
        self.gamma190 = nn.Linear(ngf * 16, ngf * 16, bias=False)
        self.beta190 = nn.Linear(ngf * 16, ngf * 16, bias=False)

        for i in range(n_blocks):
            setattr(self, 'upblock1-{}_190x'.format(i), ResnetAdaILNBlock(ngf * 16, use_bias=False))

        # h/32, w/32, 1024 => h/16, w/16, 512
        upblock2_190x = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf * 16, ngf * 8, kernel_size=3, stride=1, padding=0, bias=False),
            ILN(ngf * 8),
            nn.ReLU(True),
            nn.Conv2d(ngf * 8, ngf * 8 * 4, kernel_size=1, stride=1, padding=0, bias=True),
            nn.PixelShuffle(2),
            ILN(ngf * 8),
            nn.ReLU(True)
        ]

        self.conv190_46 = nn.Conv2d(ngf * 8 * 2, ngf * 8, kernel_size=1, stride=1, padding=0, bias=False)

        # 46
        # h/16, w/16, 512 = > h/8, w/8, 256
        upblock0_46x = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf * 8, ngf * 4, kernel_size=3, stride=1, padding=0, bias=False),
            ILN(ngf * 4),
            nn.ReLU(True),
            nn.Conv2d(ngf * 4, ngf * 4 * 4, kernel_size=1, stride=1, padding=0, bias=True),
            nn.PixelShuffle(2),
            ILN(ngf * 4),
            nn.ReLU(True)
        ]

        fc46 = [
            nn.Linear(ngf * 4, ngf * 4, bias=False),
            nn.ReLU(True),
            nn.Linear(ngf * 4, ngf * 4, bias=False),
            nn.ReLU(True),
        ]
        self.fc46 = nn.Sequential(*fc46)
        self.gamma46 = nn.Linear(ngf * 4, ngf * 4, bias=False)
        self.beta46 = nn.Linear(ngf * 4, ngf * 4, bias=False)

        for i in range(n_blocks):
            setattr(self, 'upblock1-{}_46x'.format(i), ResnetAdaILNBlock(ngf * 4, use_bias=False))

        # h/8, w/8, 256 => h/4, w/4, 128
        upblock2_46x = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf * 4, ngf * 2, kernel_size=3, stride=1, padding=0, bias=False),
            ILN(ngf * 2),
            nn.ReLU(True),
            nn.Conv2d(ngf * 2, ngf * 2 * 4, kernel_size=1, stride=1, padding=0, bias=True),
            nn.PixelShuffle(2),
            ILN(ngf * 2),
            nn.ReLU(True)
        ]

        self.conv46_10 = nn.Conv2d(ngf * 2 * 2, ngf * 2, kernel_size=1, stride=1, padding=0, bias=False)
        # self.fc46_10 = nn.Linear(ngf * 2 * 2, ngf * 2, bias=False)

        # h/4, w/4, 128 => h/2, w/2, 64
        upblock0_10x = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf * 2, ngf, kernel_size=3, stride=1, padding=0, bias=False),
            ILN(ngf),
            nn.ReLU(True),
            nn.Conv2d(ngf, ngf * 4, kernel_size=1, stride=1, padding=0, bias=True),
            nn.PixelShuffle(2),
            ILN(ngf),
            nn.ReLU(True)
        ]

        fc10 = [
            nn.Linear(ngf, ngf, bias=False),
            nn.ReLU(True),
            nn.Linear(ngf, ngf, bias=False),
            nn.ReLU(True),
        ]
        self.fc10 = nn.Sequential(*fc10)
        self.gamma10 = nn.Linear(ngf, ngf, bias=False)
        self.beta10 = nn.Linear(ngf, ngf, bias=False)

        for i in range(n_blocks):
            setattr(self, 'upblock1-{}_10x'.format(i), ResnetAdaILNBlock(ngf, use_bias=False))

        # h/2, w/2, 64 => h, w, 64
        upblock2_10x = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf, ngf // 2, kernel_size=3, stride=1, padding=0, bias=False),
            ILN(ngf // 2),
            nn.ReLU(True),
            nn.Conv2d(ngf // 2, ngf * 4, kernel_size=1, stride=1, padding=0, bias=True),
            nn.PixelShuffle(2),
            ILN(ngf),
            nn.ReLU(True),
        ]

        upblock3 = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, stride=1, padding=0, bias=False),
            nn.Tanh(),
        ]

        self.upblock0_190x = nn.Sequential(*upblock0_190x)
        self.upblock2_190x = nn.Sequential(*upblock2_190x)
        self.upblock0_46x = nn.Sequential(*upblock0_46x)
        self.upblock2_46x = nn.Sequential(*upblock2_46x)
        self.upblock0_10x = nn.Sequential(*upblock0_10x)
        self.upblock2_10x = nn.Sequential(*upblock2_10x)
        self.upblock3 = nn.Sequential(*upblock3)

    def forward(self, x10, x46, x190):
        # x10, h/4, w/4, 128
        # x46, h/16, w/16, 512
        # x190 h / 64, w / 64, 2048

        x = x190
        x = self.upblock0_190x(x)
        x_ = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        x_ = self.fc190(x_.view(x_.size(0), -1))
        gamma, beta = self.gamma190(x_), self.beta190(x_)
        for i in range(self.n_blocks):
            x = getattr(self, 'upblock1-{}_190x'.format(i))(x, gamma, beta)
        x = self.upblock2_190x(x)

        x_0 = x
        x = x46
        # print(x.size(), x_0.size())
        x = torch.cat([x, x_0], 1)
        print(x.size())
        # x = self.fc190_46(x)
        x = self.conv190_46(x)
        x = self.upblock0_46x(x)
        x_ = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        x_ = self.fc46(x_.view(x_.size(0), -1))
        gamma, beta = self.gamma46(x_), self.beta46(x_)
        for i in range(self.n_blocks):
            x = getattr(self, 'upblock1-{}_46x'.format(i))(x, gamma, beta)
        x = self.upblock2_46x(x)

        x_0 = x
        x = x10
        x = torch.cat([x, x_0], 1)
        # x = self.fc46_10(x)
        x = self.conv46_10(x)
        x = self.upblock0_10x(x)
        x_ = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        x_ = self.fc10(x_.view(x_.size(0), -1))
        gamma, beta = self.gamma10(x_), self.beta10(x_)
        for i in range(self.n_blocks):
            x = getattr(self, 'upblock1-{}_10x'.format(i))(x, gamma, beta)
        x = self.upblock2_10x(x)

        out = self.upblock3(x)

        return out


class NICE3SDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64):
        super(NICE3SDiscriminator, self).__init__()

        dis4_10 = [
            # h, w, 3 => h/2, w/2, 64, rf = 4
            nn.ReflectionPad2d(1),
            nn.utils.spectral_norm(
                nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=0, bias=True)),
            nn.LeakyReLU(0.2, True),

            # h/2, w/2, 64 => h/4, w/4, 128, rf = 10
            nn.ReflectionPad2d(1),
            nn.utils.spectral_norm(
                nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=0, bias=True)),
            nn.LeakyReLU(0.2, True),
        ]

        self.fc10 = nn.utils.spectral_norm(
            nn.Linear(ndf * 2 * 2, 1, bias=False))
        self.conv1x1_10x = nn.Conv2d(ndf * 2 * 3, ndf * 2, kernel_size=1, stride=1, bias=True)
        self.leaky_relu_10x = nn.LeakyReLU(0.2, True)

        # h/4, w/4, 128 => h/8, w/8, 256,  rf = 22
        dis22 = [
            nn.ReflectionPad2d(1),
            nn.utils.spectral_norm(
                nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=0, bias=True)),
            nn.LeakyReLU(0.2, True),
        ]

        # h/8, w/8, 256 => h/16, w/16, 512,  rf = 46
        dis46 = [
            nn.ReflectionPad2d(1),
            nn.utils.spectral_norm(
                nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=0, bias=True)),
            nn.LeakyReLU(0.2, True),
        ]

        self.fc46 = nn.utils.spectral_norm(
            nn.Linear(ndf * 8 * 2, 1, bias=False))
        self.conv1x1_46x = nn.Conv2d(ndf * 8 * 3, ndf * 8, kernel_size=1, stride=1, bias=True)
        self.leaky_relu_46x = nn.LeakyReLU(0.2, True)

        # 70x70 classifier
        cls70x = [
            # h/8, w/8, 256 => h/8, w/8, 512,  rf = 46
            nn.ReflectionPad2d(1),
            nn.utils.spectral_norm(
                nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=1, padding=0, bias=True)),
            nn.LeakyReLU(0.2, True),

            # h/8, w/8, 512 => h/8, w/8, 1,  rf = 70
            nn.ReflectionPad2d(1),
            nn.utils.spectral_norm(
                nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=0, bias=True)),
        ]

        # h/16, w/16, 512 => h/32, w/32, 1024,  rf = 94
        dis94 = [
            nn.ReflectionPad2d(1),
            nn.utils.spectral_norm(
                nn.Conv2d(ndf * 8, ndf * 16, kernel_size=4, stride=2, padding=0, bias=True)),
            nn.LeakyReLU(0.2, True),
        ]

        # 286x286 classifier
        cls286x = [
            # h/32, w/32, 1024 => h/32, w/32, 2048,  rf = 190
            nn.ReflectionPad2d(1),
            nn.utils.spectral_norm(
                nn.Conv2d(ndf * 16, ndf * 32, kernel_size=4, stride=1, padding=0, bias=True)),
            nn.LeakyReLU(0.2, True),

            # h/32, w/32, 1,  rf = 286
            nn.ReflectionPad2d(1),
            nn.utils.spectral_norm(
                nn.Conv2d(ndf * 32, 1, kernel_size=4, stride=1, padding=0, bias=True)),
        ]

        # h/32, w/32, 1024 => h/64, w/64, 2048,  rf = 190
        dis190 = [
            nn.ReflectionPad2d(1),
            nn.utils.spectral_norm(
                nn.Conv2d(ndf * 16, ndf * 32, kernel_size=4, stride=2, padding=0, bias=True)),
            nn.LeakyReLU(0.2, True),
        ]

        self.fc190 = nn.utils.spectral_norm(
            nn.Linear(ndf * 32 * 2, 1, bias=False))
        self.conv1x1_190x = nn.Conv2d(ndf * 32 * 3, ndf * 32, kernel_size=1, stride=1, bias=True)
        self.leaky_relu_190x = nn.LeakyReLU(0.2, True)

        self.dis4_10 = nn.Sequential(*dis4_10)
        self.dis22 = nn.Sequential(*dis22)
        self.dis46 = nn.Sequential(*dis46)
        self.dis94 = nn.Sequential(*dis94)
        self.dis190 = nn.Sequential(*dis190)
        self.cls70x = nn.Sequential(*cls70x)
        self.cls286x = nn.Sequential(*cls286x)

    def forward(self, x):
        # gen10x
        x = self.dis4_10(x)

        x10 = x
        gap = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        gmp = torch.nn.functional.adaptive_max_pool2d(x, 1)
        cam_logit = torch.cat([gap, gmp], 1)
        cam_logit_10 = self.fc10(cam_logit.view(cam_logit.shape[0], -1))
        weight = list(self.fc10.parameters())[0]
        x = torch.cat([x, x], 1)
        x = x * weight.unsqueeze(2).unsqueeze(3)
        x = torch.cat([x, x10], 1)
        x = self.leaky_relu_10x(self.conv1x1_10x(x))

        heatmap10x = torch.sum(x, dim=1, keepdim=True)

        x10 = x

        x = self.dis22(x)

        # cls70x
        cls70 = self.cls70x(x)

        # gen46x
        x = self.dis46(x)
        x46 = x
        gap = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        gmp = torch.nn.functional.adaptive_max_pool2d(x, 1)
        cam_logit = torch.cat([gap, gmp], 1)
        cam_logit_46 = self.fc46(cam_logit.view(cam_logit.shape[0], -1))
        weight = list(self.fc46.parameters())[0]
        x = torch.cat([x, x], 1)
        x = x * weight.unsqueeze(2).unsqueeze(3)
        x = torch.cat([x, x46], 1)
        x = self.leaky_relu_46x(self.conv1x1_46x(x))

        heatmap46x = torch.sum(x, dim=1, keepdim=True)

        x46 = x

        x = self.dis94(x)
        # cls286x
        cls286 = self.cls286x(x)

        # gen190x
        x = self.dis190(x)
        x190 = x
        gap = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        gmp = torch.nn.functional.adaptive_max_pool2d(x, 1)
        cam_logit = torch.cat([gap, gmp], 1)
        cam_logit_190 = self.fc190(cam_logit.view(cam_logit.shape[0], -1))
        weight = list(self.fc190.parameters())[0]
        x = torch.cat([x, x], 1)
        x = x * weight.unsqueeze(2).unsqueeze(3)
        x = torch.cat([x, x190], 1)
        x = self.leaky_relu_190x(self.conv1x1_190x(x))

        heatmap190x = torch.sum(x, dim=1, keepdim=True)

        x190 = x

        return cls70, cls286, x10, x46, x190, cam_logit_10, cam_logit_46, cam_logit_190, heatmap10x, heatmap46x, heatmap190x
