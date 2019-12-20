import itertools

from models import networks
from models.networks import define_G, define_D, define_Net
from utils.util import to3channel, calc_mean_std, toGray
from .base_model import BaseModel
import torch.nn as nn
import torch


class CycleGANColorizationModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(dataset_mode='colorization')
        if is_train:
            parser.add_argument('--adv_weight', type=float, default=1.0, help='weight for adversarial loss')
            parser.add_argument('--dis_weight', type=float, default=0.5, help='weight for discriminator loss')
            parser.add_argument('--cyc_weight', type=float, default=10.0, help='weight for cycle loss')
            parser.add_argument('--idt_weight', type=float, default=5.0, help='weight for identity loss')
            parser.add_argument('--con_weight', type=float, default=0.2, help='weight for content loss')
            parser.add_argument('--sty_weight', type=float, default=0.2, help='weight for style loss')
            parser.add_argument('--rec_weight', type=float, default=1.0, help='weight for style loss')

            parser.add_argument('--img_size', type=int, default=256, help='size of image')
            # parser.add_argument('--netG', type=str, default='resnet_9blocks',
            #                     help='specify generator architecture')
            # parser.add_argument('--netD', type=str, default='basic',
            #                     help='specify discriminator architecture')
            # parser.add_argument('--n_layers_D', type=int, default=5, help='only used if netD == n_layers')
            parser.add_argument('--netColor', type=str, default='adainstyletransfer',
                                help='specify pretrained color model [adainstyletransfer]')

            parser.add_argument('--preserve_color', action='store_true',
                                help='if specified, preserve color of the content image')
            parser.add_argument('--alpha', type=float, default=1.0,
                                help='the weight that controls the degree of stylization. Should be between 0 and 1')
            parser.add_argument('--no_use_pretrained_vgg', action='store_true',
                                help='the weight that controls the degree of stylization. Should be between 0 and 1')
            parser.add_argument('--pretrained_vgg_path', type=str, default='models/vgg_normalised.pth',
                                help='vgg_model path')


        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # opt.input_nc = 1
        # opt.output_nc = 1

        self.opt = opt
        self.loss_names = [
            'adv_G_A',
            'adv_G_B',
            'rec_G_A',
            'rec_G_B',
            'idt_G_A',
            'idt_G_B',
            'con_A',
            # 'content_B',
            'sty_A',
            # 'style_B',
            'rec_A',
            # 'rec_B',
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
            # 'g_t_A',
        ]
        visual_names_B = [
            'real_B_RGB',
            'real_B_Gray',
            # 'fake_B2A_RGB',
            'fake_B2A_Gray',
            # 'fake_B2A2B_RGB',
            'fake_B2A2B_Gray',
            # 'fake_B2B_RGB',
            'fake_B2B_Gray',
            # 'g_t_B',
        ]
        self.visual_names = visual_names_A + visual_names_B
        # self.model_names = ['genA2B', 'genB2A', 'genColorA2B', 'genColorB2A']
        self.model_names = ['genA2B', 'genB2A', 'genColorA2B']
        if self.isTrain:
            # self.model_names.extend(['disGA', 'disGB', 'disLA', 'disLB'])
            self.model_names.extend(['disA', 'disB', ])

        # define networks
        self.genA2B = define_G(1, 1, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.genB2A = define_G(1, 1, opt.ngf, opt.netG, opt.norm,
                                          not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        self.genColorA2B = define_Net(opt.netColor, gpu_ids=self.gpu_ids, pretrained_encoder=opt.pretrained_vgg_path)
        # self.genColorB2A = define_Net(opt.netColor, gpu_ids=self.gpu_ids, pretrained_encoder=opt.pretrained_vgg_path)

        # self.genColorA2B = define_Net(netType='adainstyletransfer', gpu_ids=self.gpu_ids,
        #                                        pretrained_encoder=None if opt.no_use_pretrained_vgg else 'models/vgg_normalised.pth',
        #                                        pretrained_decoder=None if not opt.use_pretrained_decoder else 'models/decoder.pth')
        #
        # self.genColorB2A = define_Net(netType='adainstyletransfer', gpu_ids=self.gpu_ids,
        #                                        pretrained_encoder=None if opt.no_use_pretrained_vgg else 'models/vgg_normalised.pth',
        #                                        pretrained_decoder=None if not opt.use_pretrained_decoder else 'models/decoder.pth')


        # self.disGA = define_D(opt.input_nc, opt.ndf, opt.netD, norm='spectral_norm', n_layers=7, gpu_ids=self.gpu_ids)
        # self.disGB = define_D(opt.input_nc, opt.ndf, opt.netD, norm='spectral_norm', n_layers=7, gpu_ids=self.gpu_ids)
        # self.disLA = define_D(opt.input_nc, opt.ndf, opt.netD, norm='spectral_norm', n_layers=5, gpu_ids=self.gpu_ids)
        # self.disLB = define_D(opt.input_nc, opt.ndf, opt.netD, norm='spectral_norm', n_layers=5, gpu_ids=self.gpu_ids)

        if self.isTrain:
            self.disA = define_D(1, opt.ndf, opt.netD,
                     opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.disB = define_D(1, opt.ndf, opt.netD,
                                 opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

            self.optimizer_G = torch.optim.Adam(
                itertools.chain(self.genA2B.parameters(), self.genB2A.parameters()),
                lr=opt.lr,
                betas=(opt.beta1, 0.999))

            self.optimizer_D = torch.optim.Adam(
                itertools.chain(self.disA.parameters(), self.disB.parameters()),
                lr=opt.lr,
                betas=(opt.beta1, 0.999))

            self.optimizer_GC = torch.optim.Adam(
                itertools.chain(self.genColorA2B.module.decoder.parameters()),
                lr=opt.lr,
                betas=(opt.beta1, 0.999))

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_GC)

            self.L1_loss = nn.L1Loss().to(self.device)
            self.MSE_loss = nn.MSELoss().to(self.device)

        self.adv_weight = opt.adv_weight
        self.dis_weight = opt.dis_weight
        self.cyc_weight = opt.cyc_weight
        self.idt_weight = opt.idt_weight
        self.sty_weight = opt.sty_weight
        self.con_weight = opt.con_weight
        self.rec_weight = opt.rec_weight

        self.preserve_color = opt.preserve_color
        self.alpha = opt.alpha
        self.no_use_pretrained_vgg = opt.no_use_pretrained_vgg

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
        self.optimizer_GC.step()

    def backward_G(self):

        fake_A_Gray_logit = self.disA(self.fake_A2B_Gray)
        fake_B_Gray_logit = self.disB(self.fake_B2A_Gray)

        self.loss_adv_G_A = self.MSE_loss(fake_A_Gray_logit,
                                          torch.ones_like(fake_A_Gray_logit).to(self.device))
        self.loss_adv_G_B = self.MSE_loss(fake_B_Gray_logit,
                                          torch.ones_like(fake_B_Gray_logit).to(self.device))

        self.loss_rec_G_A = self.L1_loss(self.fake_A2B2A_Gray, self.real_A_Gray)
        self.loss_rec_G_B = self.L1_loss(self.fake_B2A2B_Gray, self.real_B_Gray)

        self.loss_idt_G_A = self.L1_loss(self.fake_A2A_Gray, self.real_A_Gray)
        self.loss_idt_G_B = self.L1_loss(self.fake_B2B_Gray, self.real_B_Gray)

        loss_G = (self.loss_adv_G_A + self.loss_adv_G_B) * self.adv_weight + \
                 (self.loss_rec_G_A + self.loss_rec_G_B) * self.cyc_weight + \
                 (self.loss_idt_G_A + self.loss_idt_G_B) * self.idt_weight
        loss_G.backward()

    def backward_D(self):
        real_A_Gray_logit = self.disB(self.real_A_Gray)
        real_B_Gray_logit = self.disA(self.real_B_Gray)
        fake_A_Gray_logit = self.disB(self.fake_B2A_Gray.detach())
        fake_B_Gray_logit = self.disA(self.fake_A2B_Gray.detach())

        self.loss_adv_D_B = self.MSE_loss(real_A_Gray_logit, torch.ones_like(real_A_Gray_logit).to(self.device)) + \
                            self.MSE_loss(fake_A_Gray_logit, torch.zeros_like(fake_A_Gray_logit).to(self.device))

        self.loss_adv_D_A = self.MSE_loss(real_B_Gray_logit, torch.ones_like(real_B_Gray_logit).to(self.device)) + \
                            self.MSE_loss(fake_B_Gray_logit, torch.zeros_like(fake_B_Gray_logit).to(self.device))

        loss_D = (self.loss_adv_D_A + self.loss_adv_D_B) * self.dis_weight
        loss_D.backward()

    def backward_GC(self):
        style_feats_A, content_feats_A, g_t_feats_A, g_t_A, t_A = \
            self.genColorA2B(to3channel(self.fake_A2B_Gray.detach()), self.real_A_RGB, self.alpha)

        # style_feats_B, content_feats_B, g_t_feats_B, g_t_B, t_B = \
        #     self.genColorB2A(to3channel(self.fake_B2A_Gray.detach()), self.real_B_RGB, self.alpha)

        loss_con_A = self.calc_content_loss(g_t_feats_A[-1], t_A)
        # loss_content_B = self.calc_content_loss(g_t_feats_B[-1], t_B)
        loss_sty_A = 0
        # loss_style_B = 0
        for i in range(4):
            loss_sty_A += self.calc_style_loss(g_t_feats_A[i], style_feats_A[i])
            # loss_style_B += self.calc_style_loss(g_t_feats_B[i], style_feats_B[i])

        loss_rec_A = self.calc_rec_loss(toGray(g_t_A), self.fake_A2B_Gray.detach())
        # loss_rec_B = self.calc_rec_loss(toGray(g_t_B), self.fake_B2A_Gray.detach())

        self.fake_A2B_RGB = g_t_A
        # self.fake_B2A_RGB = g_t_B

        # loss = (loss_content_A + loss_content_B) * self.con_weight + \
        #        (loss_style_A + loss_style_B) * self.sty_weight + \
        #        (loss_rec_A + loss_rec_B) * self.rec_weight

        loss = (loss_con_A) * self.con_weight + \
               (loss_sty_A) * self.sty_weight + \
               (loss_rec_A) * self.rec_weight


        self.loss_con_A = loss_con_A
        # self.loss_content_B = loss_content_B
        self.loss_sty_A = loss_sty_A
        # self.loss_style_B = loss_style_B
        self.loss_rec_A = loss_rec_A
        # self.loss_rec_B = loss_rec_B

        loss.backward()


    def calc_content_loss(self, input, target):
        assert (input.size() == target.size())
        if not self.no_use_pretrained_vgg:
            assert (target.requires_grad is False)
        return self.MSE_loss(input, target)

    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        if not self.no_use_pretrained_vgg:
            assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.MSE_loss(input_mean, target_mean) + \
               self.MSE_loss(input_std, target_std)

    def calc_rec_loss(self, input, target):
        assert (input.size() == target.size())
        if not self.no_use_pretrained_vgg:
            assert (target.requires_grad is False)
        return self.L1_loss(input, target)
