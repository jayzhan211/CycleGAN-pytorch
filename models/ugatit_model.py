import itertools
from .base_model import BaseModel
from .networks import define_G, define_D, RhoClipper
import torch.nn as nn
import torch
import numpy as np
from utils.util import denorm, numpy2tensor, tensor2numpy, RGB2BGR, cam
from models.networks import ResnetGeneratorUGATIT, DiscriminatorUGATIT

class UGATITModel(BaseModel):
    """
    UGATIT implementation based on https://github.com/znxlwm/UGATIT-pytorch
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True)
        parser.set_defaults(use_cam=True)
        if is_train:
            parser.add_argument('--adv_weight', type=float, default=1.0, help='weight for adversarial loss')
            parser.add_argument('--cycle_weight', type=float, default=10.0, help='weight for cycle loss')
            parser.add_argument('--identity_weight', type=float, default=10.0, help='weight for identity loss')
            parser.add_argument('--cam_weight', type=float, default=1000.0, help='weight for class activate map loss')
            parser.add_argument('--img_size', type=int, default=256, help='size of image')

        return parser

    def __init__(self, opt):
        super(UGATITModel, self).__init__(opt)
        self.loss_names = [
            'G_A',
            'G_B',
            'D_A',
            'D_B',
            'rec_G_A',
            'rec_G_B',
            'idt_G_A',
            'idt_G_B',
            'cam_G_A',
            'cam_G_B',
        ]
        # visual_names_A = ['real_A', 'fake_A2B', 'fake_A2A', 'fake_A2B2A', 'fake_A2B_heatmap', 'fake_A2A_heatmap', 'fake_A2B2A_heatmap']
        # visual_names_B = ['real_B', 'fake_B2A', 'fake_B2B', 'fake_B2A2B', 'fake_B2A_heatmap', 'fake_B2B_heatmap', 'fake_B2A2B_heatmap']

        visual_names_A = ['real_A', 'fake_A2B', 'fake_A2A', 'fake_A2B2A']
        visual_names_B = ['real_B', 'fake_B2A', 'fake_B2B', 'fake_B2A2B']
        self.visual_names = visual_names_A + visual_names_B

        self.model_names = ['genA2B', 'genB2A']
        if self.isTrain:
            # self.model_names.extend(['disGA', 'disGB', 'disLA', 'disLB'])
            self.model_names.extend(['disA', 'disB'])

        """ Define Generator, Discriminator """
        self.genA2B = ResnetGeneratorUGATIT(opt.input_nc, opt.output_nc, opt.ngf, opt.n_res, opt.img_size,
                                            opt.light).to(self.device)
        self.genB2A = ResnetGeneratorUGATIT(opt.output_nc, opt.input_nc, opt.ngf, opt.n_res, opt.img_size,
                                            opt.light).to(self.device)

        self.disA = DiscriminatorUGATIT(opt.output_nc, opt.ndf, opt.n_dis).to(self.device)
        self.disB = DiscriminatorUGATIT(opt.output_nc, opt.ndf, opt.n_dis).to(self.device)

        """ Define Loss """
        self.L1_loss = nn.L1Loss().to(self.device)
        self.MSE_loss = nn.MSELoss().to(self.device)
        self.BCE_loss = nn.BCEWithLogitsLoss().to(self.device)

        """ Trainer """
        self.optimizer_G = torch.optim.Adam(
            itertools.chain(self.genA2B.parameters(), self.genB2A.parameters()),
            lr=opt.lr,
            betas=(opt.beta1, 0.999),
            weight_decay=opt.weight_decay)

        self.optimizer_D = torch.optim.Adam(
            itertools.chain(self.disGA.parameters(), self.disGB.parameters(),
                            self.disLA.parameters(), self.disLB.parameters()),
            lr=opt.lr,
            betas=(opt.beta1, 0.999),
            weight_decay=opt.weight_decay)

        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_D)
        self.RhoClipper = RhoClipper(0, 1)

        """ Weight """
        self.adv_weight = opt.adv_weight
        self.cycle_weight = opt.cycle_weight
        self.identity_weight = opt.identity_weight
        self.cam_weight = opt.cam_weight

        self.img_size = opt.img_size

    def set_input(self, _input):
        A2B = self.opt.direction in ['AtoB']
        self.real_A = _input['A' if A2B else 'B'].to(self.device)
        self.real_B = _input['B' if A2B else 'A'].to(self.device)
        self.image_paths = _input['A_paths' if A2B else 'B_paths']

    def forward(self):

        # Update D
        self.optimizer_D.zero_grad()

        fake_A2B, _, _ = self.genA2B(self.real_A)
        fake_B2A, _, _ = self.genB2A(self.real_B)

        real_GA_logit, real_GA_cam_logit, _ = self.disGA(self.real_A)
        real_LA_logit, real_LA_cam_logit, _ = self.disLA(self.real_A)
        real_GB_logit, real_GB_cam_logit, _ = self.disGB(self.real_B)
        real_LB_logit, real_LB_cam_logit, _ = self.disLB(self.real_B)

        fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(fake_B2A)
        fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(fake_B2A)
        fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B)
        fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(fake_A2B)

        # Loss of D
        D_ad_loss_GA = self.MSE_loss(real_GA_logit, torch.ones_like(real_GA_logit).to(self.device)) + \
                       self.MSE_loss(fake_GA_logit, torch.zeros_like(fake_GA_logit).to(self.device))
        D_ad_cam_loss_GA = self.MSE_loss(real_GA_cam_logit, torch.ones_like(real_GA_cam_logit).to(self.device)) + \
                           self.MSE_loss(fake_GA_cam_logit, torch.zeros_like(fake_GA_cam_logit).to(self.device))
        D_ad_loss_LA = self.MSE_loss(real_LA_logit, torch.ones_like(real_LA_logit).to(self.device)) + \
                       self.MSE_loss(fake_LA_logit, torch.zeros_like(fake_LA_logit).to(self.device))
        D_ad_cam_loss_LA = self.MSE_loss(real_LA_cam_logit, torch.ones_like(real_LA_cam_logit).to(self.device)) + \
                           self.MSE_loss(fake_LA_cam_logit, torch.zeros_like(fake_LA_cam_logit).to(self.device))
        D_ad_loss_GB = self.MSE_loss(real_GB_logit, torch.ones_like(real_GB_logit).to(self.device)) + \
                       self.MSE_loss(fake_GB_logit, torch.zeros_like(fake_GB_logit).to(self.device))
        D_ad_cam_loss_GB = self.MSE_loss(real_GB_cam_logit, torch.ones_like(real_GB_cam_logit).to(self.device)) + \
                           self.MSE_loss(fake_GB_cam_logit, torch.zeros_like(fake_GB_cam_logit).to(self.device))
        D_ad_loss_LB = self.MSE_loss(real_LB_logit, torch.ones_like(real_LB_logit).to(self.device)) + \
                       self.MSE_loss(fake_LB_logit, torch.zeros_like(fake_LB_logit).to(self.device))
        D_ad_cam_loss_LB = self.MSE_loss(real_LB_cam_logit, torch.ones_like(real_LB_cam_logit).to(self.device)) + \
                           self.MSE_loss(fake_LB_cam_logit, torch.zeros_like(fake_LB_cam_logit).to(self.device))

        self.loss_D_A = self.adv_weight * (D_ad_loss_GA + D_ad_cam_loss_GA + D_ad_loss_LA + D_ad_cam_loss_LA)
        self.loss_D_B = self.adv_weight * (D_ad_loss_GB + D_ad_cam_loss_GB + D_ad_loss_LB + D_ad_cam_loss_LB)

        loss_D = self.loss_D_A + self.loss_D_B
        loss_D.backward()
        self.optimizer_D.step()

        # Update G

        self.optimizer_G.zero_grad()
        fake_A2B, fake_A2B_cam_logit, fake_A2B_heatmap = self.genA2B(self.real_A)
        fake_B2A, fake_B2A_cam_logit, fake_B2A_heatmap = self.genB2A(self.real_B)

        fake_A2B2A, _, fake_A2B2A_heatmap = self.genB2A(fake_A2B)
        fake_B2A2B, _, fake_B2A2B_heatmap = self.genA2B(fake_B2A)

        fake_A2A, fake_A2A_cam_logit, fake_A2A_heatmap = self.genB2A(self.real_A)
        fake_B2B, fake_B2B_cam_logit, fake_B2B_heatmap = self.genA2B(self.real_B)

        fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(fake_B2A)
        fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(fake_B2A)
        fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B)
        fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(fake_A2B)

        G_ad_loss_GA = self.MSE_loss(fake_GA_logit, torch.ones_like(fake_GA_logit).to(self.device))
        G_ad_cam_loss_GA = self.MSE_loss(fake_GA_cam_logit, torch.ones_like(fake_GA_cam_logit).to(self.device))
        G_ad_loss_LA = self.MSE_loss(fake_LA_logit, torch.ones_like(fake_LA_logit).to(self.device))
        G_ad_cam_loss_LA = self.MSE_loss(fake_LA_cam_logit, torch.ones_like(fake_LA_cam_logit).to(self.device))
        G_ad_loss_GB = self.MSE_loss(fake_GB_logit, torch.ones_like(fake_GB_logit).to(self.device))
        G_ad_cam_loss_GB = self.MSE_loss(fake_GB_cam_logit, torch.ones_like(fake_GB_cam_logit).to(self.device))
        G_ad_loss_LB = self.MSE_loss(fake_LB_logit, torch.ones_like(fake_LB_logit).to(self.device))
        G_ad_cam_loss_LB = self.MSE_loss(fake_LB_cam_logit, torch.ones_like(fake_LB_cam_logit).to(self.device))

        self.loss_rec_G_A = self.L1_loss(fake_A2B2A, self.real_A)
        self.loss_rec_G_B = self.L1_loss(fake_B2A2B, self.real_B)

        self.loss_idt_G_A = self.L1_loss(fake_A2A, self.real_A)
        self.loss_idt_G_B = self.L1_loss(fake_B2B, self.real_B)

        self.loss_cam_G_A = self.BCE_loss(fake_B2A_cam_logit,
                                          torch.ones_like(fake_B2A_cam_logit).to(self.device)) + \
                            self.BCE_loss(fake_A2A_cam_logit,
                                          torch.zeros_like(fake_A2A_cam_logit).to(self.device))

        self.loss_cam_G_B = self.BCE_loss(fake_A2B_cam_logit,
                                          torch.ones_like(fake_A2B_cam_logit).to(self.device)) + \
                            self.BCE_loss(fake_B2B_cam_logit,
                                          torch.zeros_like(fake_B2B_cam_logit).to(self.device))

        self.loss_G_A = self.adv_weight * (G_ad_loss_GA + G_ad_cam_loss_GA + G_ad_loss_LA + G_ad_cam_loss_LA) + \
                        self.cycle_weight * self.loss_rec_G_A + \
                        self.identity_weight * self.loss_idt_G_A + \
                        self.cam_weight * self.loss_cam_G_A

        self.loss_G_B = self.adv_weight * (G_ad_loss_GB + G_ad_cam_loss_GB + G_ad_loss_LB + G_ad_cam_loss_LB) + \
                        self.cycle_weight * self.loss_rec_G_B + \
                        self.identity_weight * self.loss_idt_G_B + \
                        self.cam_weight * self.loss_cam_G_B

        loss_G = self.loss_G_A + self.loss_G_B
        loss_G.backward()
        self.optimizer_G.step()

        self.genA2B.apply(self.RhoClipper)
        self.genB2A.apply(self.RhoClipper)

        # self.realA
        self.fake_A2A_heatmap = fake_A2A_heatmap
        self.fake_A2B_heatmap = fake_A2B_heatmap
        self.fake_A2B2A_heatmap = fake_A2B2A_heatmap
        self.fake_A2A = fake_A2A
        self.fake_A2B = fake_A2B
        self.fake_A2B2A = fake_A2B2A

        # self.realB
        self.fake_B2A_heatmap = fake_B2A_heatmap
        self.fake_B2B_heatmap = fake_B2B_heatmap
        self.fake_B2A2B_heatmap = fake_B2A2B_heatmap
        self.fake_B2A = fake_B2A
        self.fake_B2B = fake_B2B
        self.fake_B2A2B = fake_B2A2B

    def optimize_parameters(self):
        self.forward()
