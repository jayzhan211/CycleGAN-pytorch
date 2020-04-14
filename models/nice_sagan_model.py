import itertools
from .base_model import BaseModel
from .networks import define_G, define_D, RhoClipper
import torch.nn as nn
import torch
import numpy as np
from utils.util import denorm, tensor2numpy, RGB2BGR, cam, tensor2im
from models.networks import NICEV2ResnetGenerator, NICEDiscriminator, NICEResnetGenerator, NICESADiscriminator


class NICESAGANModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(display_nrows=5)
        # parser.add_argument('--light', action='store_true', help='use light mode')
        parser.add_argument('--n_dis', type=int, default=7, help='The number of discriminator layer')
        parser.add_argument('--n_res', type=int, default=6, help='The number of residual layer')
        parser.add_argument('--img_size', type=int, default=256, help='size of image')
        parser.set_defaults(light=True)
        parser.set_defaults(netG='nice_gen_v2')
        parser.set_defaults(netD='nice_dis_sa')
        if is_train:
            parser.add_argument('--adv_weight', type=float, default=1.0, help='weight for adversarial loss')
            parser.add_argument('--cycle_weight', type=float, default=10.0, help='weight for cycle loss')
            parser.add_argument('--identity_weight', type=float, default=10.0, help='weight for identity loss')
            parser.add_argument('--weight_decay', type=float, default=0.0001, help='the weight decay in adam optimizer')
        return parser

    def __init__(self, opt):
        super(NICESAGANModel, self).__init__(opt)
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

        visual_names_A = ['real_A', 'fake_A2B', 'fake_A2A', 'fake_A2B2A', 'real_A_heatmap']
        visual_names_B = ['real_B', 'fake_B2A', 'fake_B2B', 'fake_B2A2B', 'real_B_heatmap']
        self.visual_names = visual_names_A + visual_names_B

        self.model_names = ['genA2B', 'genB2A']
        if self.isTrain:
            # self.model_names.extend(['disGA', 'disGB', 'disLA', 'disLB'])
            self.model_names.extend(['disA', 'disB'])

        """ Define Generator, Discriminator """

        if opt.netG == 'nice_gen_v2':
            self.genA2B = NICEV2ResnetGenerator(opt.input_nc, opt.output_nc, opt.ngf, opt.n_res, opt.img_size).to(self.device)
            self.genB2A = NICEV2ResnetGenerator(opt.output_nc, opt.input_nc, opt.ngf, opt.n_res, opt.img_size).to(self.device)
        else:
            raise NotImplementedError('netG : {} is not Implemented.'.format(opt.netG))

        if self.isTrain:
            if opt.netD == 'nice_dis_sa':
                self.disA = NICESADiscriminator(opt.output_nc, opt.ndf, opt.n_dis).to(self.device)
                self.disB = NICESADiscriminator(opt.input_nc, opt.ndf, opt.n_dis).to(self.device)
            else:
                raise NotImplementedError('netD : {} is not Implemented.'.format(opt.netD))

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
                # itertools.chain(self.disGA.parameters(), self.disGB.parameters(),
                #                 self.disLA.parameters(), self.disLB.parameters()),
                itertools.chain(self.disA.parameters(), self.disB.parameters()),
                lr=opt.lr,
                betas=(opt.beta1, 0.999),
                weight_decay=opt.weight_decay)

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            """ Weight """
            self.adv_weight = opt.adv_weight
            self.cycle_weight = opt.cycle_weight
            self.identity_weight = opt.identity_weight

        self.image_size = opt.img_size

    def set_input(self, input):
        AtoB = self.opt.direction in ['AtoB']
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def test(self):
        with torch.no_grad():
            _, _, A_heatmap, real_A_z = self.disA(self.real_A)
            _, _, B_heatmap, real_B_z = self.disB(self.real_B)

            fake_A2B = self.genA2B(real_A_z)
            fake_B2A = self.genB2A(real_B_z)

            _, _, _, fake_A_z = self.disA(fake_B2A)
            _, _, _, fake_B_z = self.disB(fake_A2B)

            fake_B2A2B = self.genA2B(fake_A_z)
            fake_A2B2A = self.genB2A(fake_B_z)

            fake_A2A = self.genB2A(real_A_z)
            fake_B2B = self.genA2B(real_B_z)

            self.real_A_heatmap = tensor2im(A_heatmap, use_cam=True, image_size=self.image_size)
            self.fake_A2A = tensor2im(fake_A2A)
            self.fake_A2B = tensor2im(fake_A2B)
            self.fake_A2B2A = tensor2im(fake_A2B2A)

            self.real_B_heatmap = tensor2im(B_heatmap, use_cam=True, image_size=self.image_size)
            self.fake_B2B = tensor2im(fake_B2B)
            self.fake_B2A = tensor2im(fake_B2A)
            self.fake_B2A2B = tensor2im(fake_B2A2B)

    def train(self, display):

        # Update D
        self.optimizer_D.zero_grad()

        real_LA_logit, real_GA_logit,  _, real_A_z = self.disA(self.real_A)
        real_LB_logit, real_GB_logit,  _, real_B_z = self.disB(self.real_B)

        fake_A2B = self.genA2B(real_A_z).detach()
        fake_B2A = self.genB2A(real_B_z).detach()

        fake_LA_logit, fake_GA_logit,  _, _ = self.disA(fake_B2A)
        fake_LB_logit, fake_GB_logit,  _, _ = self.disB(fake_A2B)

        D_ad_loss_GA = self.MSE_loss(real_GA_logit, torch.ones_like(real_GA_logit).to(self.device)) + self.MSE_loss(
            fake_GA_logit, torch.zeros_like(fake_GA_logit).to(self.device))
        D_ad_loss_LA = self.MSE_loss(real_LA_logit, torch.ones_like(real_LA_logit).to(self.device)) + self.MSE_loss(
            fake_LA_logit, torch.zeros_like(fake_LA_logit).to(self.device))
        D_ad_loss_GB = self.MSE_loss(real_GB_logit, torch.ones_like(real_GB_logit).to(self.device)) + self.MSE_loss(
            fake_GB_logit, torch.zeros_like(fake_GB_logit).to(self.device))
        D_ad_loss_LB = self.MSE_loss(real_LB_logit, torch.ones_like(real_LB_logit).to(self.device)) + self.MSE_loss(
            fake_LB_logit, torch.zeros_like(fake_LB_logit).to(self.device))

        # D_ad_cam_loss_A = self.MSE_loss(real_A_cam_logit,
        #                                 torch.ones_like(real_A_cam_logit).to(self.device)) + self.MSE_loss(
        #     fake_A_cam_logit, torch.zeros_like(fake_A_cam_logit).to(self.device))
        # D_ad_cam_loss_B = self.MSE_loss(real_B_cam_logit,
        #                                 torch.ones_like(real_B_cam_logit).to(self.device)) + self.MSE_loss(
        #     fake_B_cam_logit, torch.zeros_like(fake_B_cam_logit).to(self.device))

        # loss_D_A = self.adv_weight * (D_ad_loss_GA + D_ad_cam_loss_A + D_ad_loss_LA)
        # loss_D_B = self.adv_weight * (D_ad_loss_GB + D_ad_cam_loss_B + D_ad_loss_LB)

        loss_D_A = self.adv_weight * (D_ad_loss_GA + D_ad_loss_LA)
        loss_D_B = self.adv_weight * (D_ad_loss_GB + D_ad_loss_LB)

        self.loss_D = loss_D_A + loss_D_B
        self.loss_D.backward()
        self.optimizer_D.step()

        # Update G

        self.optimizer_G.zero_grad()

        _, _, A_heatmap, real_A_z = self.disA(self.real_A)
        _, _, B_heatmap, real_B_z = self.disB(self.real_B)

        fake_A2B = self.genA2B(real_A_z)
        fake_B2A = self.genB2A(real_B_z)

        fake_LA_logit, fake_GA_logit, _, fake_A_z = self.disA(fake_B2A)
        fake_LB_logit, fake_GB_logit, _, fake_B_z = self.disB(fake_A2B)

        fake_B2A2B = self.genA2B(fake_A_z)
        fake_A2B2A = self.genB2A(fake_B_z)

        G_ad_loss_GA = self.MSE_loss(fake_GA_logit, torch.ones_like(fake_GA_logit).to(self.device))
        G_ad_loss_LA = self.MSE_loss(fake_LA_logit, torch.ones_like(fake_LA_logit).to(self.device))
        G_ad_loss_GB = self.MSE_loss(fake_GB_logit, torch.ones_like(fake_GB_logit).to(self.device))
        G_ad_loss_LB = self.MSE_loss(fake_LB_logit, torch.ones_like(fake_LB_logit).to(self.device))

        # G_ad_cam_loss_A = self.MSE_loss(fake_A_cam_logit, torch.ones_like(fake_A_cam_logit).to(self.device))
        # G_ad_cam_loss_B = self.MSE_loss(fake_B_cam_logit, torch.ones_like(fake_B_cam_logit).to(self.device))

        G_cycle_loss_A = self.L1_loss(fake_A2B2A, self.real_A)
        G_cycle_loss_B = self.L1_loss(fake_B2A2B, self.real_B)

        fake_A2A = self.genB2A(real_A_z)
        fake_B2B = self.genA2B(real_B_z)

        G_recon_loss_A = self.L1_loss(fake_A2A, self.real_A)
        G_recon_loss_B = self.L1_loss(fake_B2B, self.real_B)

        G_loss_A = self.adv_weight * (
                G_ad_loss_GA + G_ad_loss_LA) + self.cycle_weight * G_cycle_loss_A + self.identity_weight * G_recon_loss_A
        G_loss_B = self.adv_weight * (
                G_ad_loss_GB + G_ad_loss_LB) + self.cycle_weight * G_cycle_loss_B + self.identity_weight * G_recon_loss_B


        # G_loss_A = self.adv_weight * (
        #         G_ad_loss_GA + G_ad_cam_loss_A + G_ad_loss_LA) + self.cycle_weight * G_cycle_loss_A + self.identity_weight * G_recon_loss_A
        # G_loss_B = self.adv_weight * (
        #         G_ad_loss_GB + G_ad_cam_loss_B + G_ad_loss_LB) + self.cycle_weight * G_cycle_loss_B + self.identity_weight * G_recon_loss_B

        self.loss_G = G_loss_A + G_loss_B
        self.loss_G.backward()
        self.optimizer_G.step()

        if display:
            self.real_A_heatmap = tensor2im(A_heatmap, use_cam=True, image_size=self.image_size)
            self.fake_A2A = tensor2im(fake_A2A)
            self.fake_A2B = tensor2im(fake_A2B)
            self.fake_A2B2A = tensor2im(fake_A2B2A)

            self.real_B_heatmap = tensor2im(B_heatmap, use_cam=True, image_size=self.image_size)
            self.fake_B2B = tensor2im(fake_B2B)
            self.fake_B2A = tensor2im(fake_B2A)
            self.fake_B2A2B = tensor2im(fake_B2A2B)
