import torch
import itertools
from utils.image_pool import ImagePool
from .base_model import BaseModel
from .networks import *


class CycleGANColorizationModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True)
        if is_train:
            parser.add_argument('--adv_weight', type=float, default=10.0, help='weight for adversarial loss')
            parser.add_argument('--cycle_weight', type=float, default=1.0, help='weight for cycle loss')
            parser.add_argument('--identity_weight', type=float, default=100.0, help='weight for identity loss')
            parser.add_argument('--n_res', type=int, default=4, help='number of resblock')
            parser.add_argument('--n_dis', type=int, default=6, help='number of discriminator layer')
            parser.add_argument('--img_size', type=int, default=256, help='size of image')
            # parser.add_argument('--img_ch', type=int, default=3, help='channels of image')
            parser.add_argument('--netG', type=str, default='resnet_ugatit_6blocks',
                                help='specify generator architecture in ugatit [ resnet_ugatit_6blocks ]')
            parser.add_argument('--netD', type=str, default='UGATIT',
                                help='specify discriminator architecture in ugatit [UGATIT]')

        return parser

    def __init__(self, opt):
        super(CycleGANColorizationModel, self).__init__()
        self.loss_names = [
            'G_A', 'G_B',
            'D_A', 'D_B',
            'rec_G_A', 'rec_G_B',
            'idt_G_A', 'idt_G_B'
        ]
        visual_names_A = ['real_A', 'fake_A2B', 'fake_A2B2A', 'fake_A2A']
        visual_names_B = ['real_B', 'fake_B2A', 'fake_B2A2B', 'fake_B2B']
        self.visual_names = visual_names_A + visual_names_B
        self.model_names = ['genA2B', 'genB2A']
        if self.isTrain:
            self.model_names.extend(['disGA', 'disGB', 'disLA', 'disLB'])

        # define networks
        self.genA2B = define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, gpu_ids=self.gpu_ids)
        self.genB2A = define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, gpu_ids=self.gpu_ids)
        # self.disGA = define_D(opt.input_nc, opt.ndf, opt.netD, norm='spectral_norm', n_layers=7, gpu_ids=self.gpu_ids)
        # self.disGB = define_D(opt.input_nc, opt.ndf, opt.netD, norm='spectral_norm', n_layers=7, gpu_ids=self.gpu_ids)
        # self.disLA = define_D(opt.input_nc, opt.ndf, opt.netD, norm='spectral_norm', n_layers=5, gpu_ids=self.gpu_ids)
        # self.disLB = define_D(opt.input_nc, opt.ndf, opt.netD, norm='spectral_norm', n_layers=5, gpu_ids=self.gpu_ids)
        self.disA = define_D(opt.input_nc, opt.output_nc, opt.ngf, opt.netD, gpu_ids=self.gpu_ids)
        self.disB = define_D(opt.input_nc, opt.output_nc, opt.ngf, opt.netD, gpu_ids=self.gpu_ids)

        # define loss
        self.L1_loss = nn.L1Loss().to(self.device)
        self.MSE_loss = nn.MSELoss().to(self.device)
        self.BCE_loss = nn.BCEWithLogitsLoss().to(self.device)

        # define optimizer
        self.optimizer_G = torch.optim.Adam(
            itertools.chain(self.genA2B.parameters(), self.genB2A.parameters()),
            lr=opt.lr,
            betas=(opt.beta1, 0.999),
            weight_decay=opt.weight_decay)

        # self.optimizer_D = torch.optim.Adam(
        #     itertools.chain(self.disGA.parameters(), self.disGB.parameters(),
        #                     self.disLA.parameters(), self.disLB.parameters()),
        #     lr=opt.lr,
        #     betas=(opt.beta1, 0.999),
        #     weight_decay=opt.weight_decay)

        self.optimizer_D = torch.optim.Adam(
            itertools.chain(self.disA.parameters(), self.disB.parameters()),
            lr=opt.lr,
            betas=(opt.beta1, 0.999),
            weight_decay=opt.weight_decay)

        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_D)

        # weight
        self.adv_weight = opt.adv_weight
        self.cycle_weight = opt.cycle_weight
        self.identity_weight = opt.identity_weight

        self.real_A_RGB = None
        self.real_B_RGB = None
        self.real_A_Gray = None
        self.real_B_Gray = None
        self.fake_A2B = None
        self.fake_B2A = None
        self.D_loss_A = None
        self.D_loss_B = None

    def set_input(self, _input):
        A2B = self.opt.direction in ['A2B', 'AtoB']
        self.real_A_RGB = _input['A_RGB' if A2B else 'B'].to(self.device)
        self.real_B_RGB = _input['B_RGB' if A2B else 'A'].to(self.device)
        self.real_A_Gray = _input['A_Gray']
        self.real_B_Gray = _input['B_Gray']
        self.image_paths = _input['A_paths' if A2B else 'B_paths']

    def forward(self):
        # Update D
        self.optimizer_D.zero_grad()
        self.fake_A2B_RGB, self.fake_A2B_Gray = self.genA2B(self.real_A_RGB, self.real_A_Gray)
        self.fake_B2A_RGB, self.fake_B2A_Gray = self.genB2A(self.real_B_RGB, self.real_B_Gray)

        # real_GA_logit = self.disGA(self.real_A_RGB)
        # real_LA_logit = self.disLA(self.real_A_RGB)
        # real_GB_logit = self.disGB(self.real_B_RGB)
        # real_LB_logit = self.disLB(self.real_B_RGB)
        #
        # fake_GA_logit = self.disGA(self.fake_B2A_RGB)
        # fake_LA_logit = self.disLA(self.fake_B2A_RGB)
        # fake_GB_logit = self.disGB(self.fake_A2B_RGB)
        # fake_LB_logit = self.disLB(self.fake_A2B_RGB)

        real_A_logit = self.disA(self.real_A_RGB)
        real_B_logit = self.disB(self.real_B_RGB)
        fake_A_logit = self.disA(self.fake_B2A_RGB)
        fake_B_logit = self.disB(self.fake_A2B_RGB)

        D_ad_loss_A = self.MSE_loss(
            real_A_logit, torch.ones_like(real_A_logit).to(self.device)) + \
                      self.MSE_loss(
                          fake_A_logit, torch.zeros_like(fake_A_logit).to(self.device))

        D_ad_loss_B = self.MSE_loss(
            real_B_logit, torch.ones_like(real_B_logit).to(self.device)) + \
                      self.MSE_loss(
                          fake_B_logit, torch.zeros_like(fake_B_logit).to(self.device))

        # D_ad_loss_GA = self.MSE_loss(
        #     real_GA_logit, torch.ones_like(real_GA_logit).to(self.device)) + \
        #                self.MSE_loss(
        #                    fake_GA_logit, torch.zeros_like(fake_GA_logit).to(self.device))
        #
        # D_ad_loss_LA = self.MSE_loss(
        #     real_LA_logit, torch.ones_like(real_LA_logit).to(self.device)) + \
        #                self.MSE_loss(
        #                    fake_LA_logit, torch.zeros_like(fake_LA_logit).to(self.device))
        #
        # D_ad_loss_GB = self.MSE_loss(
        #     real_GB_logit, torch.ones_like(real_GB_logit).to(self.device)) + \
        #                self.MSE_loss(
        #                    fake_GB_logit, torch.zeros_like(fake_GB_logit).to(self.device))
        #
        # D_ad_loss_LB = self.MSE_loss(
        #     real_LB_logit, torch.ones_like(real_LB_logit).to(self.device)) + \
        #                self.MSE_loss(
        #                    fake_LB_logit, torch.zeros_like(fake_LB_logit).to(self.device))

        # self.D_loss_A = self.adv_weight * (D_ad_loss_GA + D_ad_loss_LA)
        # self.D_loss_B = self.adv_weight * (D_ad_loss_GB + D_ad_loss_LB)

        self.D_loss_A = self.adv_weight * (D_ad_loss_A)
        self.D_loss_B = self.adv_weight * (D_ad_loss_B)

        D_loss = self.D_loss_A + self.D_loss_B
        D_loss.backward()
        self.optimizer_D.step()

        # Update G
        self.optimizer_G.zero_grad()

        self.fake_A2B_RGB, self.fake_A2B_Gray = self.genA2B(self.real_A_RGB, self.real_A_Gray)
        self.fake_B2A_RGB, self.fake_B2A_Gray = self.genB2A(self.real_B_RGB, self.real_B_Gray)

        self.fake_A2B2A_RGB, self.fake_A2B2A_Gray = self.genB2A(self.fake_A2B_RGB, self.fake_A2B_Gray)
        self.fake_B2A2B_RGB, self.fake_B2A2B_Gray = self.genA2B(self.fake_B2A_RGB, self.fake_B2A_Gray)

        self.fake_A2A_RGB, self.fake_A2A_Gray = self.genB2A(self.real_A_RGB, self.real_A_Gray)
        self.fake_B2B_RGB, self.fake_B2B_Gray = self.genA2B(self.real_B_RGB, self.real_B_Gray)

        fake_A_logit = self.disA(self.fake_B2A_RGB)
        fake_B_logit = self.disB(self.fake_A2B_RGB)

        # fake_GA_logit = self.disGA(self.fake_B2A_RGB)
        # fake_LA_logit = self.disLA(self.fake_B2A_RGB)
        # fake_GB_logit = self.disGB(self.fake_A2B_RGB)
        # fake_LB_logit = self.disLB(self.fake_A2B_RGB)

        G_ad_loss_A = self.MSE_loss(
            fake_A_logit, torch.ones_like(fake_A_logit).to(self.device))

        G_ad_loss_B = self.MSE_loss(
            fake_B_logit, torch.ones_like(fake_B_logit).to(self.device))

        # G_ad_loss_GA = self.MSE_loss(
        #     fake_GA_logit, torch.ones_like(fake_GA_logit).to(self.device))
        # G_ad_loss_LA = self.MSE_loss(
        #     fake_LA_logit, torch.ones_like(fake_LA_logit).to(self.device))
        # G_ad_loss_GB = self.MSE_loss(
        #     fake_GB_logit, torch.ones_like(fake_GB_logit).to(self.device))
        # G_ad_loss_LB = self.MSE_loss(
        #     fake_LB_logit, torch.ones_like(fake_LB_logit).to(self.device))

        self.loss_rec_G_A = self.L1_loss(self.fake_A2B2A_RGB, self.real_A_RGB)
        self.loss_rec_G_B = self.L1_loss(self.fake_B2A2B_RGB, self.real_B_RGB)

        self.loss_idt_G_A = self.L1_loss(self.fake_A2A_RGB, self.real_A_RGB)
        self.loss_idt_G_B = self.L1_loss(self.fake_B2B_RGB, self.real_B_RGB)

        # self.loss_G_A = self.adv_weight * (G_ad_loss_GA + G_ad_loss_LA) + \
        #                 self.cycle_weight * self.loss_rec_G_A + self.identity_weight * self.loss_idt_G_A
        #
        # self.loss_G_B = self.adv_weight * (G_ad_loss_GB + G_ad_loss_LB) + \
        #                 self.cycle_weight * self.loss_rec_G_B + self.identity_weight * self.loss_idt_G_B

        self.loss_G_A = self.adv_weight * G_ad_loss_A + \
                        self.cycle_weight * self.loss_rec_G_A + self.identity_weight * self.loss_idt_G_A

        self.loss_G_B = self.adv_weight * G_ad_loss_B + \
                        self.cycle_weight * self.loss_rec_G_B + self.identity_weight * self.loss_idt_G_B

        loss_G = self.loss_G_A + self.loss_G_B
        loss_G.backward()
        self.optimizer_G.step()

    def optimize_parameters(self):
        self.forward()
