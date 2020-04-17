import itertools
from .base_model import BaseModel
from .networks import define_G, define_D, RhoClipper
import torch.nn as nn
import torch
import numpy as np
from utils.util import denorm, tensor2numpy, RGB2BGR, cam, tensor2im
from models.networks import EfficientDetGenerator, EfficientDetDiscrminator


class EFFICIENTDETGANModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(display_nrows=4)
        # parser.add_argument('--light', action='store_true', help='use light mode')
        # parser.add_argument('--n_dis', type=int, default=7, help='The number of discriminator layer')
        # parser.add_argument('--n_res', type=int, default=6, help='The number of residual layer')
        parser.add_argument('--img_size', type=int, default=256, help='size of image')
        if is_train:
            parser.add_argument('--adv_weight', type=float, default=1.0, help='weight for adversarial loss')
            parser.add_argument('--cycle_weight', type=float, default=10.0, help='weight for cycle loss')
            parser.add_argument('--identity_weight', type=float, default=10.0, help='weight for identity loss')
            parser.add_argument('--weight_decay', type=float, default=0.0001, help='the weight decay in adam optimizer')
        return parser

    def __init__(self, opt):
        super(EFFICIENTDETGANModel, self).__init__(opt)
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

        self.disA = EfficientDetDiscrminator().to(self.device)
        self.disB = EfficientDetDiscrminator().to(self.device)

        self.genA2B = EfficientDetGenerator(opt.output_nc,
                                            bifpn_in_channels=self.disA.backbone.get_list_features()[-5:],
                                            bifpn_out_channels=224,
                                            num_stacks=6,
                                            num_outs=5).to(self.device)

        self.genB2A = EfficientDetGenerator(opt.output_nc,
                                            bifpn_in_channels=self.disB.backbone.get_list_features()[-5:],
                                            bifpn_out_channels=224,
                                            num_stacks=6,
                                            num_outs=5).to(self.device)

        if self.isTrain:
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
            featsA, _ = self.disA(self.real_A)
            featsB, _ = self.disB(self.real_B)

            fake_A2B = self.genA2B(featsA)
            fake_B2A = self.genB2A(featsB)

            fake_featsA, fake_A_logit = self.disA(fake_B2A)
            fake_featsB, fake_B_logit = self.disB(fake_A2B)

            fake_B2A2B = self.genA2B(fake_featsA)
            fake_A2B2A = self.genB2A(fake_featsB)

            fake_A2A = self.genB2A(featsA)
            fake_B2B = self.genA2B(featsB)


            self.fake_A2A = tensor2im(fake_A2A)
            self.fake_A2B = tensor2im(fake_A2B)
            self.fake_A2B2A = tensor2im(fake_A2B2A)

            self.fake_B2B = tensor2im(fake_B2B)
            self.fake_B2A = tensor2im(fake_B2A)
            self.fake_B2A2B = tensor2im(fake_B2A2B)

    def train(self, display):

        # Update D
        self.optimizer_D.zero_grad()

        featsA, realA_logit = self.disA(self.real_A)
        featsB, realB_logit = self.disB(self.real_B)

        fake_A2B = self.genA2B(featsA).detach()
        fake_B2A = self.genB2A(featsB).detach()

        _, fake_A_logit = self.disA(fake_B2A)
        _, fake_B_logit = self.disB(fake_A2B)

        D_adv_loss_A = self.MSE_loss(realA_logit, torch.ones_like(realA_logit).to(self.device)) + self.MSE_loss(
            fake_A_logit, torch.zeros_like(fake_A_logit).to(self.device))
        D_adv_loss_B = self.MSE_loss(realB_logit, torch.ones_like(realB_logit).to(self.device)) + self.MSE_loss(
            fake_B_logit, torch.zeros_like(fake_B_logit).to(self.device))

        self.loss_D = self.adv_weight * (D_adv_loss_A + D_adv_loss_B)
        self.loss_D.backward()
        self.optimizer_D.step()

        # Update G

        self.optimizer_G.zero_grad()

        featsA, _ = self.disA(self.real_A)
        featsB, _ = self.disB(self.real_B)

        fake_A2B = self.genA2B(featsA)
        fake_B2A = self.genB2A(featsB)

        fake_featsA, fake_A_logit = self.disA(fake_B2A)
        fake_featsB, fake_B_logit = self.disB(fake_A2B)

        
        fake_B2A2B = self.genA2B(fake_featsA)
        fake_A2B2A = self.genB2A(fake_featsB)

        G_adv_loss_A = self.MSE_loss(fake_A_logit, torch.ones_like(fake_A_logit).to(self.device))
        G_adv_loss_B = self.MSE_loss(fake_B_logit, torch.ones_like(fake_B_logit).to(self.device))

        G_cycle_loss_A = self.L1_loss(fake_A2B2A, self.real_A)
        G_cycle_loss_B = self.L1_loss(fake_B2A2B, self.real_B)

        fake_A2A = self.genB2A(featsA)
        fake_B2B = self.genA2B(featsB)

        G_recon_loss_A = self.L1_loss(fake_A2A, self.real_A)
        G_recon_loss_B = self.L1_loss(fake_B2B, self.real_B)

        G_loss_A = self.adv_weight * G_adv_loss_A + self.cycle_weight * G_cycle_loss_A + self.identity_weight * G_recon_loss_A
        G_loss_B = self.adv_weight * G_adv_loss_B + self.cycle_weight * G_cycle_loss_B + self.identity_weight * G_recon_loss_B

        self.loss_G = G_loss_A + G_loss_B
        self.loss_G.backward()
        self.optimizer_G.step()

        if display:

            self.fake_A2A = tensor2im(fake_A2A)
            self.fake_A2B = tensor2im(fake_A2B)
            self.fake_A2B2A = tensor2im(fake_A2B2A)

            self.fake_B2B = tensor2im(fake_B2B)
            self.fake_B2A = tensor2im(fake_B2A)
            self.fake_B2A2B = tensor2im(fake_B2A2B)
