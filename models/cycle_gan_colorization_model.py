import itertools

from models import networks
from models.networks import define_G, define_D
from utils.util import to3channel, calc_mean_std
from .base_model import BaseModel
import torch.nn as nn
import torch


class CycleGANColorizationModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True)
        if is_train:
            parser.add_argument('--adv_weight', type=float, default=1.0, help='weight for adversarial loss')
            parser.add_argument('--cycle_weight', type=float, default=10.0, help='weight for cycle loss')
            parser.add_argument('--identity_weight', type=float, default=5.0, help='weight for identity loss')
            parser.add_argument('--content_weight', type=float, default=0.2, help='weight for content loss')
            parser.add_argument('--style_weight', type=float, default=0.2, help='weight for style loss')
            parser.add_argument('--rec_weight', type=float, default=1.0, help='weight for style loss')

            parser.add_argument('--img_size', type=int, default=256, help='size of image')
            parser.add_argument('--netG', type=str, default='resnet_6blocks',
                                help='specify generator architecture')
            parser.add_argument('--netD', type=str, default='basic',
                                help='specify discriminator architecture')

            parser.add_argument('--input_nc_color', type=int, default=1,
                                help='# of input image channels: 3 for RGB and 1 for grayscale')
            parser.add_argument('--output_nc_color', type=int, default=3,
                                help='# of output image channels: 3 for RGB and 1 for grayscale')

            parser.add_argument('--vgg_model_path', type=str, default='models/vgg_normalised.pth',
                                help='path of pretrained_vgg_model')

        return parser

    def __init__(self, opt):
        super(CycleGANColorizationModel, self).__init__(opt)

        opt.input_nc = 1
        opt.output_nc = 1

        self.opt = opt
        self.loss_names = [
            'adv_G_A',
            'adv_G_B',
            'rec_G_A',
            'rec_G_B',
            'idt_G_A',
            'idt_G_B',
        ]
        visual_names_A = [
            'real_A_RGB',
            'real_A_Gray',
            'fake_A2B_RGB',
            'fake_A2B_Gray',
            # 'fake_A2B2A_RGB',
            'fake_A2B2A_Gray',
            # 'fake_A2A_RGB',
            'fake_A2A_Gray',
        ]
        visual_names_B = [
            'real_B_RGB',
            'real_B_Gray',
            'fake_B2A_RGB',
            'fake_B2A_Gray',
            # 'fake_B2A2B_RGB',
            'fake_B2A2B_Gray',
            # 'fake_B2B_RGB',
            'fake_B2B_Gray',
        ]
        self.visual_names = visual_names_A + visual_names_B
        self.model_names = ['genA2B', 'genB2A', 'genColorA2B', 'genColorB2A']
        if self.isTrain:
            # self.model_names.extend(['disGA', 'disGB', 'disLA', 'disLB'])
            self.model_names.extend(['disA', 'disB', ])

        # define networks
        self.genA2B = define_G(opt.input_nc, opt.output_nc, ngf=opt.ngf, netG=opt.netG, gpu_ids=self.gpu_ids,
                               norm=opt.norm, use_dropout=not opt.no_dropout,
                               init_type=opt.init_type, init_gain=opt.init_gain)
        self.genB2A = define_G(opt.output_nc, opt.input_nc, ngf=opt.ngf, netG=opt.netG, gpu_ids=self.gpu_ids,
                               norm=opt.norm, use_dropout=not opt.no_dropout,
                               init_type=opt.init_type, init_gain=opt.init_gain)

        self.genColorA2B = networks.define_Net(net_type='adainstyletransfer', gpu_ids=self.gpu_ids, pretrained_model_path=opt.vgg_model_path)
        self.genColorB2A = networks.define_Net(net_type='adainstyletransfer', gpu_ids=self.gpu_ids, pretrained_model_path=opt.vgg_model_path)
        # self.disGA = define_D(opt.input_nc, opt.ndf, opt.netD, norm='spectral_norm', n_layers=7, gpu_ids=self.gpu_ids)
        # self.disGB = define_D(opt.input_nc, opt.ndf, opt.netD, norm='spectral_norm', n_layers=7, gpu_ids=self.gpu_ids)
        # self.disLA = define_D(opt.input_nc, opt.ndf, opt.netD, norm='spectral_norm', n_layers=5, gpu_ids=self.gpu_ids)
        # self.disLB = define_D(opt.input_nc, opt.ndf, opt.netD, norm='spectral_norm', n_layers=5, gpu_ids=self.gpu_ids)

        if self.isTrain:
            self.disA = define_D(opt.output_nc, opt.ndf, opt.netD, gpu_ids=self.gpu_ids)
            self.disB = define_D(opt.input_nc, opt.ndf, opt.netD, gpu_ids=self.gpu_ids)

            self.optimizer_G = torch.optim.Adam(
                itertools.chain(self.genA2B.parameters(), self.genB2A.parameters()),
                lr=opt.lr,
                betas=(opt.beta1, 0.999),
                weight_decay=opt.weight_decay)

            self.optimizer_D = torch.optim.Adam(
                itertools.chain(self.disA.parameters(), self.disB.parameters()),
                lr=opt.lr,
                betas=(opt.beta1, 0.999),
                weight_decay=opt.weight_decay)

            self.optimizer_GC = torch.optim.Adam(
                itertools.chain(self.genColorA2B.module.decoder.parameters(), self.genColorB2A.module.decoder.parameters()),
                lr=opt.lr,
                betas=(opt.beta1, 0.999))

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_GC)

            self.L1_loss = nn.L1Loss().to(self.device)
            self.MSE_loss = nn.MSELoss().to(self.device)

        self.adv_weight = opt.adv_weight
        self.cycle_weight = opt.cycle_weight
        self.identity_weight = opt.identity_weight
        self.content_weight = opt.content_weight
        self.style_weight = opt.style_weight
        self.rec_weight = opt.rec_weight

        self.real_A_Gray = None
        self.real_B_Gray = None

        self.fake_A2B_Gray = None
        self.fake_A2B2A_Gray = None
        self.fake_B2A_Gray = None
        self.fake_B2A2B_Gray = None
        self.fake_A2A_Gray = None
        self.fake_B2B_Gray = None
        self.fake_A2B_RGB = None
        self.fake_B2A_RGB = None

        self.loss_adv_G_A = None
        self.loss_adv_G_B = None
        self.loss_rec_G_A = None
        self.loss_rec_G_B = None
        self.loss_idt_G_A = None
        self.loss_idt_G_B = None
        self.loss_content_G_A = None
        self.loss_style_G_A = None
        self.loss_color_rec_G_A = None
        self.loss_content_G_B = None
        self.loss_style_G_B = None
        self.loss_color_rec_G_B = None

    def set_input(self, _input):
        A2B = self.opt.direction in ['A2B', 'AtoB']
        self.real_A_RGB = _input['A_RGB' if A2B else 'B_RGB'].to(self.device)
        self.real_B_RGB = _input['B_RGB' if A2B else 'A_RGB'].to(self.device)
        self.real_A_Gray = _input['A_Gray' if A2B else 'B_Gray'].to(self.device)
        self.real_B_Gray = _input['B_Gray' if A2B else 'A_Gray'].to(self.device)
        self.image_paths = _input['A_paths' if A2B else 'B_paths']

    def forward(self):

        self.fake_A2B_Gray = self.genA2B(self.real_A_Gray)  # G_A(A)
        self.fake_A2B2A_Gray = self.genB2A(self.fake_A2B_Gray)
        self.fake_B2A_Gray = self.genB2A(self.real_B_Gray)
        self.fake_B2A2B_Gray = self.genA2B(self.fake_B2A_Gray)
        self.fake_A2A_Gray = self.genB2A(self.real_A_Gray)
        self.fake_B2B_Gray = self.genA2B(self.real_B_Gray)

    def optimize_parameters(self):
        # get fake and reconstruct image
        self.forward()

        """
            Step #1 Shape Translation
        """

        # G
        self.set_requires_grad([self.disA, self.disB], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        # D
        self.set_requires_grad([self.disA, self.disB], True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        """
            Step #2 Colorization
        """

        self.optimizer_GC.zero_grad()
        self.backward_GC()
        # loss = (self.loss_content_G_A + self.loss_content_G_B) * self.content_weight + \
        #        (self.loss_style_G_A + self.loss_style_G_B) * self.style_weight + \
        #        (self.loss_color_rec_G_A + self.loss_color_rec_G_B) * self.rec_weight
        # loss.backward()
        self.optimizer_GC.step()

    def backward_G(self):

        fake_A_Gray_logit = self.disA(self.fake_B2A_Gray)
        fake_B_Gray_logit = self.disB(self.fake_A2B_Gray)

        self.loss_adv_G_A = self.MSE_loss(fake_A_Gray_logit,
                                          torch.ones_like(fake_A_Gray_logit).to(self.device))
        self.loss_adv_G_B = self.MSE_loss(fake_B_Gray_logit,
                                          torch.ones_like(fake_B_Gray_logit).to(self.device))

        self.loss_rec_G_A = self.L1_loss(self.fake_A2B2A_Gray, self.real_A_Gray)
        self.loss_rec_G_B = self.L1_loss(self.fake_B2A2B_Gray, self.real_B_Gray)

        self.loss_idt_G_A = self.L1_loss(self.fake_A2A_Gray, self.real_A_Gray)
        self.loss_idt_G_B = self.L1_loss(self.fake_B2B_Gray, self.real_B_Gray)

        loss_G = (self.loss_adv_G_A + self.loss_adv_G_B) * self.adv_weight + \
                 (self.loss_rec_G_A + self.loss_rec_G_B) * self.cycle_weight + \
                 (self.loss_idt_G_A + self.loss_idt_G_B) * self.identity_weight
        loss_G.backward()

    def backward_D(self):
        real_A_Gray_logit = self.disA(self.real_A_Gray)
        real_B_Gray_logit = self.disB(self.real_B_Gray)
        fake_A_Gray_logit = self.disA(self.fake_B2A_Gray.detach())
        fake_B_Gray_logit = self.disB(self.fake_A2B_Gray.detach())

        self.loss_adv_D_A = self.MSE_loss(real_A_Gray_logit, torch.ones_like(real_A_Gray_logit).to(self.device)) + \
                            self.MSE_loss(fake_A_Gray_logit, torch.zeros_like(fake_A_Gray_logit).to(self.device))

        self.loss_adv_D_B = self.MSE_loss(real_B_Gray_logit, torch.ones_like(real_B_Gray_logit).to(self.device)) + \
                            self.MSE_loss(fake_B_Gray_logit, torch.zeros_like(fake_B_Gray_logit).to(self.device))

        loss_D = (self.loss_adv_D_A + self.loss_adv_D_B) * 0.5
        loss_D.backward()

    def backward_GC(self):
        style_feats_A, content_feats_A, g_t_feats_A, t_A = self.genColorA2B(to3channel(self.fake_A2B_Gray.detach()), self.real_A_RGB)

        style_feats_B, content_feats_B, g_t_feats_B, t_B = self.genColorB2A(to3channel(self.fake_B2A_Gray.detach()), self.real_B_RGB)

        loss_content_A = self.calc_content_loss(g_t_feats_A[-1], t_A)
        loss_content_B = self.calc_content_loss(g_t_feats_B[-1], t_B)
        loss_style_A = 0
        loss_style_B = 0
        for i in range(4):
            loss_style_A += self.calc_style_loss(g_t_feats_A[i], style_feats_A[i])
            loss_style_B += self.calc_style_loss(g_t_feats_B[i], style_feats_B[i])



        loss = (loss_content_A + loss_content_B) * self.content_weight + \
               (loss_style_A + loss_style_B) * self.style_weight
        loss.backward()


    def calc_content_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        return self.MSE_loss(input, target)

    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.MSE_loss(input_mean, target_mean) + \
               self.MSE_loss(input_std, target_std)

    def calc_rec_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        return self.L1_loss(input, target)
