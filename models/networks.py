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
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [{}] is not implemented'.format(opt.lr_policy))
    return scheduler


def get_norm_layer(norm_type='instance'):
    """
    :param norm_type: instance | batchnorm | none
    :return:

    for batchnorm, we use learnable affine parameters
    for instancenorm, we dont use learnalbe affine paramters
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = Identity()
    else:
        raise NotImplementedError('normalization layer [{}] is not found'.format(norm_type))
    return norm_layer


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [{}] is not implemented'.format(padding_type))

        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
            nn.ReLU(True),
        ]

        if use_dropout:
            conv_block += [
                nn.Dropout(0.5),
            ]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [{}] is not implemented'.format(padding_type))

        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
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
        batchnorm has bias term already, use_bias only if we use instancenorm
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
    """
    def __init__(self, num_features, eps=1e-5):
        super(AdaLIn, self).__init__()
        self.eps = eps
        self.rho = nn.Parameter(torch.tensor(1, num_features, 1, 1))
        self.rho.data.fill_(0.9)

    def forward(self, x, gamma, beta):
        in_mean, in_var = torch.mean(x, dim=(2, 3), keepdim=True), torch.var(x, dim=(2, 3), keepdim=True)
        out_in = (x - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(x, dim=(1, 2, 3), keepdim=True), torch.var(x, dim=(1, 2, 3), keepdim=True)
        out_ln = (x - ln_mean) / torch.sqrt(ln_var + self.eps)
        out = self.rho.expand(x.shape[0], -1, -1, -1) * out_in + (
                    1 - self.rho.expand(x.shape[0], -1, -1, -1)) * out_ln
        out = out * gamma.unsqueeze(2).unsqueeze(3) + beta.unsqueeze(2).unsqueeze(3)
        # TODO how to remain the value of rho in range [0, 1]



class ResnetAdaLInBlock(nn.Module):
    def __init__(self, dim, use_bias):
        super(ResnetAdaLInBlock, self).__init__()



class ResnetGenerator_UGATIT(nn.Module):
    def __init__(self,
                 input_nc,
                 output_nc,
                 ngf=64,
                 norm_layer=nn.InstanceNorm2d,
                 n_blocks=6,
                 img_size=256,
                 light=False,
                 padding_type='reflect',
                 use_dropout=False):
        super(ResnetGenerator_UGATIT, self).__init__()

        self.norm_layer = norm_layer
        self.light = light
        self.img_size = img_size
        self.n_blocks = n_blocks
        self.ngf = ngf
        self.output_nc = output_nc
        self.input_nc = input_nc

        # TODO maybe try leakyrelu instead
        downsample_block = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, output_nc, kernel_size=7, bias=False),
            norm_layer(ngf),
            nn.ReLU(True),
        ]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            downsample_block += [
                nn.ReflectionPad2d(1),
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, bias=False),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True)
            ]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):  # add ResNet blocks
            downsample_block += [
                ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                            use_dropout=use_dropout, use_bias=False),
            ]


        # Class Activate Map
        ## global averge pooling
        self.gap_fc = nn.Linear(ngf * mult, 1, bias=False)
        ## global maximum pooling
        self.gmp_fc = nn.Linear(ngf * mult, 1, bias=False)
        self.conv1x1 = nn.Conv2d(ngf * mult * 2, ngf * mult, kernel_size=1)
        self.relu = nn.ReLU(True)

        # Gamma, Beta Block
        if self.light:
            self.fc = nn.Sequential([
                nn.Linear(ngf * mult, ngf * mult, bias=False),
                nn.ReLU(True),
                nn.Linear(ngf * mult, ngf * mult, bias=False),
                nn.ReLU(True),
            ])
        else:
            self.fc = nn.Sequential([
                nn.Linear(img_size // mult * img_size // mult * ngf * mult, ngf * mult, bias=False),
                nn.ReLU(True),
                nn.Linear(ngf * mult, ngf * mult, bias=False),
                nn.ReLU(True),
            ])

        self.gamma = nn.Linear(ngf * mult, ngf * mult, bias=False)
        self.beta = nn.Linear(ngf * mult, ngf * mult, bias=False)

        # Upsample Bottleneck
        for i in range(n_blocks):
            setattr(self, 'UpBlock1_{}'.format(i+1), ResnetAdaLInBlock(ngf * mult, use_bias=False))


        self.downsample_block = nn.Sequential(*downsample_block)

    def forward(self, x):
        x = self.downsample_block(x)

        gap = F.adaptive_avg_pool2d(x, 1)
        gap_logit = self.gap_fc(gap.view(gap.shape[0], -1))
        gap_weight = list(self.gap_fc.parameters())[0]
        gap = x * gap_weight.unsqueeze(2).unsqueeze(3)

        gmp = F.adaptive_max_pool2d(x, 1)
        gmp_logit = self.gmp_fc(gap.view(gap.shape[0], -1))
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
                raise NotImplementedError('initialization method [{}] is not implemented' .format(init_type))
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
             norm='batch',
             use_dropout=False,
             init_type='normal',
             init_gain=0.02,
             gpu_ids=None):
    """

    :param input_nc: number of channels in input_images
    :param output_nc: ouput_images
    :param ngf: number of filters in last conv_layer
    :param net_G: resnet_6 | resnet_9
    :param norm: batch_norm | instance_norm | none
    :param use_dropout:
    :param init_type: initialize method
    :param init_gain: scaling factor for normal, xavier, orthogonal
    :param gpu_ids: e.g., 0,1,2
    :return:

    use ReLU for non-linearity
    """
    if gpu_ids is None:
        gpu_ids = []

    norm_layer = get_norm_layer(norm_type=norm)
    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    else:
        raise NotImplementedError('Generator model name [{}] is not recognized'.format(netG))
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
             n_layers_D=3,
             norm='batch',
             init_type='normal',
             init_gain=0.02,
             gpu_ids=None):
    """
    Discriminator,  it uses leaky relu
    :param input_nc:
    :param ndf: number of filter in the first conv layer
    :param netD: basic | pixel | n_layers
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
    norm_layer = get_norm_layer(norm_type=norm)
    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
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
