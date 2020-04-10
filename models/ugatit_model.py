import itertools
from .base_model import BaseModel
from .networks import define_G, define_D, RhoClipper
import torch.nn as nn
import torch
import numpy as np
from utils.util import denorm, tensor2numpy, RGB2BGR, cam, tensor2im
from models.networks import ResnetGeneratorUGATIT, DiscriminatorUGATIT


class UGATITModel(BaseModel):
    """
    UGATIT implementation based on https://github.com/znxlwm/UGATIT-pytorch
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True)
        parser.set_defaults(use_cam=True)
        parser.set_defaults(display_nrows=7)
        parser.add_argument('--n_res', type=int, default=4, help='The number of resblock')
        parser.add_argument('--n_dis', type=int, default=6, help='The number of discriminator layer')
        parser.add_argument('--img_size', type=int, default=256, help='size of image')
        parser.add_argument('--light', action='store_true', help='use light mode')
        if is_train:
            parser.add_argument('--adv_weight', type=float, default=1.0, help='weight for adversarial loss')
            parser.add_argument('--cycle_weight', type=float, default=10.0, help='weight for cycle loss')
            parser.add_argument('--identity_weight', type=float, default=10.0, help='weight for identity loss')
            parser.add_argument('--cam_weight', type=float, default=1000.0, help='weight for class activate map loss')
            parser.add_argument('--weight_decay', type=float, default=0.0001, help='the weight decay in adam optimizer')
        return parser

    def __init__(self, opt):
        super(UGATITModel, self).__init__(opt)

        self.loss_names = [
            'G', 'D',
            # 'G_A',
            # 'G_B',
            # 'D_A',
            # 'D_B',
            # 'rec_G_A',
            # 'rec_G_B',
            # 'idt_G_A',
            # 'idt_G_B',
            # 'cam_G_A',
            # 'cam_G_B',
        ]
        visual_names_A = ['real_A', 'fake_A2B', 'fake_A2A', 'fake_A2B2A', 'fake_A2B_heatmap', 'fake_A2A_heatmap',
                          'fake_A2B2A_heatmap']
        visual_names_B = ['real_B', 'fake_B2A', 'fake_B2B', 'fake_B2A2B', 'fake_B2A_heatmap', 'fake_B2B_heatmap',
                          'fake_B2A2B_heatmap']

        # visual_names_A = ['real_A', 'fake_A2B', 'fake_A2A', 'fake_A2B2A']
        # visual_names_B = ['real_B', 'fake_B2A', 'fake_B2B', 'fake_B2A2B']
        self.visual_names = visual_names_A + visual_names_B

        self.model_names = ['genA2B', 'genB2A']
        if self.isTrain:
            self.model_names.extend(['disGA', 'disGB', 'disLA', 'disLB'])
            # self.model_names.extend(['disA', 'disB'])

        """ Define Generator, Discriminator """
        self.genA2B = ResnetGeneratorUGATIT(input_nc=opt.input_nc, output_nc=opt.output_nc, ngf=opt.ngf,
                                            n_blocks=opt.n_res, img_size=opt.img_size, light=opt.light).to(self.device)
        self.genB2A = ResnetGeneratorUGATIT(input_nc=opt.output_nc, output_nc=opt.input_nc, ngf=opt.ngf,
                                            n_blocks=opt.n_res, img_size=opt.img_size, light=opt.light).to(self.device)

        if self.isTrain:
            # self.disA = DiscriminatorUGATIT(opt.output_nc, opt.ndf, opt.n_dis).to(self.device)
            # self.disB = DiscriminatorUGATIT(opt.input_nc, opt.ndf, opt.n_dis).to(self.device)
            self.disGA = DiscriminatorUGATIT(input_nc=opt.output_nc, ndf=opt.ndf, n_layers=7).to(self.device)
            self.disGB = DiscriminatorUGATIT(input_nc=opt.input_nc, ndf=opt.ndf, n_layers=7).to(self.device)
            self.disLA = DiscriminatorUGATIT(input_nc=opt.output_nc, ndf=opt.ndf, n_layers=5).to(self.device)
            self.disLB = DiscriminatorUGATIT(input_nc=opt.input_nc, ndf=opt.ndf, n_layers=5).to(self.device)

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
                # itertools.chain(self.disA.parameters(), self.disB.parameters()),
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

    def set_input(self, input):
        AtoB = self.opt.direction in ['AtoB']
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.real_A_Bcolor = input.get('A_Bcolor' if AtoB else 'B_Acolor', self.real_A if AtoB else self.real_B).to(self.device)
        self.real_B_Acolor = input.get('B_Acolor' if AtoB else 'A_Bcolor', self.real_B if AtoB else self.real_A).to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def test(self):

        with torch.no_grad():
            fake_A2B, _, fake_A2B_heatmap = self.genA2B(self.real_A)
            fake_B2A, _, fake_B2A_heatmap = self.genB2A(self.real_B)

            fake_A2B2A, _, fake_A2B2A_heatmap = self.genB2A(fake_A2B)
            fake_B2A2B, _, fake_B2A2B_heatmap = self.genA2B(fake_B2A)

            fake_A2A, _, fake_A2A_heatmap = self.genB2A(self.real_A)
            fake_B2B, _, fake_B2B_heatmap = self.genA2B(self.real_B)

            self.fake_A2A_heatmap = tensor2im(fake_A2A_heatmap, use_cam=True, image_size=self.img_size)
            self.fake_A2B_heatmap = tensor2im(fake_A2B_heatmap, use_cam=True, image_size=self.img_size)
            self.fake_A2B2A_heatmap = tensor2im(fake_A2B2A_heatmap, use_cam=True, image_size=self.img_size)
            self.fake_A2A = tensor2im(fake_A2A)
            self.fake_A2B = tensor2im(fake_A2B)
            self.fake_A2B2A = tensor2im(fake_A2B2A)

            self.fake_B2B_heatmap = tensor2im(fake_B2B_heatmap, use_cam=True, image_size=self.img_size)
            self.fake_B2A_heatmap = tensor2im(fake_B2A_heatmap, use_cam=True, image_size=self.img_size)
            self.fake_B2A2B_heatmap = tensor2im(fake_B2A2B_heatmap, use_cam=True, image_size=self.img_size)
            self.fake_B2B = tensor2im(fake_B2B)
            self.fake_B2A = tensor2im(fake_B2A)
            self.fake_B2A2B = tensor2im(fake_B2A2B)

    def train(self, display):
        # Update D
        self.optimizer_D.zero_grad()

        fake_A2B, _, _ = self.genA2B(self.real_A)
        fake_B2A, _, _ = self.genB2A(self.real_B)

        # real_GA_logit, real_GA_cam_logit, _ = self.disGA(self.real_A)
        # real_LA_logit, real_LA_cam_logit, _ = self.disLA(self.real_A)
        # real_GB_logit, real_GB_cam_logit, _ = self.disGB(self.real_B)
        # real_LB_logit, real_LB_cam_logit, _ = self.disLB(self.real_B)

        real_GA_logit, real_GA_cam_logit, _ = self.disGA(self.real_A_Bcolor)
        real_LA_logit, real_LA_cam_logit, _ = self.disLA(self.real_A_Bcolor)
        real_GB_logit, real_GB_cam_logit, _ = self.disGB(self.real_B_Acolor)
        real_LB_logit, real_LB_cam_logit, _ = self.disLB(self.real_B_Acolor)

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

        loss_D_A = self.adv_weight * (D_ad_loss_GA + D_ad_cam_loss_GA + D_ad_loss_LA + D_ad_cam_loss_LA)
        loss_D_B = self.adv_weight * (D_ad_loss_GB + D_ad_cam_loss_GB + D_ad_loss_LB + D_ad_cam_loss_LB)

        self.loss_D = loss_D_A + loss_D_B
        self.loss_D.backward()
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

        loss_rec_G_A = self.L1_loss(fake_A2B2A, self.real_A)
        loss_rec_G_B = self.L1_loss(fake_B2A2B, self.real_B)

        loss_idt_G_A = self.L1_loss(fake_A2A, self.real_A)
        loss_idt_G_B = self.L1_loss(fake_B2B, self.real_B)

        loss_cam_G_A = self.BCE_loss(fake_B2A_cam_logit,
                                     torch.ones_like(fake_B2A_cam_logit).to(self.device)) + \
                       self.BCE_loss(fake_A2A_cam_logit,
                                     torch.zeros_like(fake_A2A_cam_logit).to(self.device))

        loss_cam_G_B = self.BCE_loss(fake_A2B_cam_logit,
                                     torch.ones_like(fake_A2B_cam_logit).to(self.device)) + \
                       self.BCE_loss(fake_B2B_cam_logit,
                                     torch.zeros_like(fake_B2B_cam_logit).to(self.device))

        loss_G_A = self.adv_weight * (G_ad_loss_GA + G_ad_cam_loss_GA + G_ad_loss_LA + G_ad_cam_loss_LA) + \
                   self.cycle_weight * loss_rec_G_A + \
                   self.identity_weight * loss_idt_G_A + \
                   self.cam_weight * loss_cam_G_A

        loss_G_B = self.adv_weight * (G_ad_loss_GB + G_ad_cam_loss_GB + G_ad_loss_LB + G_ad_cam_loss_LB) + \
                   self.cycle_weight * loss_rec_G_B + \
                   self.identity_weight * loss_idt_G_B + \
                   self.cam_weight * loss_cam_G_B

        self.loss_G = loss_G_A + loss_G_B
        self.loss_G.backward()
        self.optimizer_G.step()

        self.genA2B.apply(self.RhoClipper)
        self.genB2A.apply(self.RhoClipper)

        # self.fake_A2A_heatmap = fake_A2A_heatmap
        # self.fake_A2B_heatmap = fake_A2B_heatmap
        # self.fake_A2B2A_heatmap = fake_A2B2A_heatmap
        # self.fake_A2A = fake_A2A
        # self.fake_A2B = fake_A2B
        # self.fake_A2B2A = fake_A2B2A

        # transform to numpy [256, 256, 3]

        if display:

            self.fake_A2A_heatmap = tensor2im(fake_A2A_heatmap, use_cam=True, image_size=self.img_size)
            self.fake_A2B_heatmap = tensor2im(fake_A2B_heatmap, use_cam=True, image_size=self.img_size)
            self.fake_A2B2A_heatmap = tensor2im(fake_A2B2A_heatmap, use_cam=True, image_size=self.img_size)
            self.fake_A2A = tensor2im(fake_A2A)
            self.fake_A2B = tensor2im(fake_A2B)
            self.fake_A2B2A = tensor2im(fake_A2B2A)

            self.fake_B2B_heatmap = tensor2im(fake_B2B_heatmap, use_cam=True, image_size=self.img_size)
            self.fake_B2A_heatmap = tensor2im(fake_B2A_heatmap, use_cam=True, image_size=self.img_size)
            self.fake_B2A2B_heatmap = tensor2im(fake_B2A2B_heatmap, use_cam=True, image_size=self.img_size)
            self.fake_B2B = tensor2im(fake_B2B)
            self.fake_B2A = tensor2im(fake_B2A)
            self.fake_B2A2B = tensor2im(fake_B2A2B)

        # self.realB
        # self.fake_B2A_heatmap = fake_B2A_heatmap
        # self.fake_B2B_heatmap = fake_B2B_heatmap
        # self.fake_B2A2B_heatmap = fake_B2A2B_heatmap
        # self.fake_B2A = fake_B2A
        # self.fake_B2B = fake_B2B
        # self.fake_B2A2B = fake_B2A2B

        # print(self.real_A.size())  # 1, 3, 256 ,256
        # print(self.fake_A2B.size())
        # print(self.fake_A2B_heatmap.size())  # 1, 1, 64, 64
        # print(self.fake_A2B2A.size())
        # print(self.fake_A2B2A_heatmap.size())

