import itertools

from .base_model import BaseModel
from .networks import *


class UGATITModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True)
        if is_train:
            parser.add_argument('--adv_weight', type=float, default=1.0, help='weight for adversarial loss')
            parser.add_argument('--cycle_weight', type=float, default=10.0, help='weight for cycle loss')
            parser.add_argument('--identity_weight', type=float, default=0.5, help='weight for identity loss')
            parser.add_argument('--cam_weight', type=float, default=10.0, help='weight for class activate map loss')

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = [
            'G_ad_GA',  # generator adversarial global A->B
            'G_ad_cam_GA',  # cam loss
            'G_ad_LA',  # local
            'G_ad_cam_LA',
            'G_ad_GB',  # B->A
            'G_ad_cam_GB',
            'G_ad_LB',
            'G_ad_cam_LB',
            'G_rec_A',  # reconstruct loss
            'G_rec_B',
            'G_idt_A',  # identity loss
            'G_idt_B',
            'G_cam_A',  # cam loss
            'G_cam_B',

            'D_ad_GA',  # discriminator
            'D_ad_cam_GA',
            'D_ad_LA',
            'D_ad_cam_LA',
            'D_ad_GB',
            'D_ad_cam_GB',
            'D_ad_LB',
            'D_ad_cam_LB',
        ]

        self.visual_names = [
            'real_A', 'fake_B', 'rec_A', 'idt_B',
            'real_B', 'fake_A', 'rec_B', 'idt_A',
        ]

        self.model_names = ['G_A', 'G_B']
        if self.isTrain:
            self.model_names.extend(['D_A', 'D_B'])

        # define networks
        self.genA2B = define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, self.gpu_ids)
        self.genB2A = define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, self.gpu_ids)
        self.disGA = define_D(opt.input_nc, opt.ndf, opt.netD, n_layers=7, gpu_ids=self.gpu_ids)
        self.disGB = define_D(opt.input_nc, opt.ndf, opt.netD, n_layers=7, gpu_ids=self.gpu_ids)
        self.disLA = define_D(opt.input_nc, opt.ndf, opt.netD, n_layers=5, gpu_ids=self.gpu_ids)
        self.disLB = define_D(opt.input_nc, opt.ndf, opt.netD, n_layers=5, gpu_ids=self.gpu_ids)

        # define loss
        self.L1_loss = nn.L1Loss().to(self.device)
        self.MSE_loss = nn.MSELoss().to(self.device)
        self.BCE_loss = nn.BCEWithLogitsLoss().to(self.device)

        # define optimizer
        self.optimizer_G = torch.optim.Adam(itertools.chain(self.genA2B.parameters(), self.genB2A.parameters()),
                                            lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
        self.optimizer_D = torch.optim.Adam(
            itertools.chain(self.disGA.parameters(), self.disGB.parameters(), self.disLA.parameters(),
                            self.disLB.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999),
            weight_decay=opt.weight_decay)

        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_D)
        self.RhoClipper = RhoClipper(0, 1)

        # weight
        self.adv_weight = opt.adv_weight
        self.cycle_weight = opt.cycle_weight

        self.real_A = None
        self.real_B = None
        self.fake_A2B = None
        self.fake_B2A = None
        self.real_GA_cam_logit = None
        self.real_GA_logit = None
        self.real_LA_cam_logit = None
        self.real_LA_logit = None
        self.real_GB_cam_logit = None
        self.real_GB_logit = None
        self.real_LB_cam_logit = None
        self.real_LB_logit = None

        self.fake_GA_cam_logit = None
        self.fake_GA_logit = None
        self.fake_LA_cam_logit = None
        self.fake_LA_logit = None
        self.fake_GB_cam_logit = None
        self.fake_GB_logit = None
        self.fake_LB_cam_logit = None
        self.fake_LB_logit = None
        self.loss_D_GA = None
        self.loss_D_LA = None
        self.loss_cam_D_GA = None
        self.loss_cam_D_LA = None
        self.loss_D_GB = None
        self.loss_D_LB = None
        self.loss_cam_D_GB = None
        self.loss_cam_D_LB = None
        self.loss_D_A = None
        self.loss_D_B = None
        self.loss_D = None

    def set_input(self, _input):
        A2B = self.opt.direction in ['A2B', 'AtoB']
        self.real_A = _input['A' if A2B else 'B'].to(self.device)
        self.real_B = _input['B' if A2B else 'A'].to(self.device)
        self.image_paths = _input['A_paths' if A2B else 'B_paths']

    def forward(self):

        # Update D
        self.optimizer_D.zero_grad()

        self.fake_A2B, _, _ = self.genA2B(self.real_A)
        self.fake_B2A, _, _ = self.genB2A(self.real_B)

        self.real_GA_logit, self.real_GA_cam_logit, _ = self.disGA(self.real_A)
        self.real_LA_logit, self.real_LA_cam_logit, _ = self.disLA(self.real_A)
        self.real_GB_logit, self.real_GB_cam_logit, _ = self.disGB(self.real_B)
        self.real_LB_logit, self.real_LB_cam_logit, _ = self.disLB(self.real_B)

        self.fake_GA_logit, self.fake_GA_cam_logit, _ = self.disGA(self.fake_B2A)
        self.fake_LA_logit, self.fake_LA_cam_logit, _ = self.disLA(self.fake_B2A)
        self.fake_GB_logit, self.fake_GB_cam_logit, _ = self.disGB(self.fake_A2B)
        self.fake_LB_logit, self.fake_LB_cam_logit, _ = self.disLB(self.fake_A2B)

        # Loss of D
        self.loss_D_GA = self.MSE_loss(self.real_GA_logit, torch.ones_like(self.real_GA_logit)).to(self.device) + \
                         self.MSE_loss(self.fake_GA_logit, torch.zeros_like(self.fake_GA_logit)).to(self.device)
        self.loss_cam_D_GA = self.MSE_loss(self.real_GA_cam_logit, torch.ones_like(self.real_GA_cam_logit)).to(
            self.device) + \
                             self.MSE_loss(self.fake_GA_cam_logit, torch.zeros_like(self.fake_GA_cam_logit)).to(
                                 self.device)
        self.loss_D_LA = self.MSE_loss(self.real_LA_logit, torch.ones_like(self.real_LA_logit)).to(self.device) + \
                         self.MSE_loss(self.fake_LA_logit, torch.zeros_like(self.fake_LA_logit)).to(self.device)
        self.loss_cam_D_LA = self.MSE_loss(self.real_LA_cam_logit, torch.ones_like(self.real_LA_cam_logit)).to(
            self.device) + \
                             self.MSE_loss(self.fake_LA_cam_logit, torch.zeros_like(self.fake_LA_cam_logit)).to(
                                 self.device)
        self.loss_D_GB = self.MSE_loss(self.real_GB_logit, torch.ones_like(self.real_GB_logit)).to(self.device) + \
                         self.MSE_loss(self.fake_GB_logit, torch.zeros_like(self.fake_GB_logit)).to(self.device)
        self.loss_cam_D_GB = self.MSE_loss(self.real_GB_cam_logit, torch.ones_like(self.real_GB_cam_logit)).to(
            self.device) + \
                             self.MSE_loss(self.fake_GB_cam_logit, torch.zeros_like(self.fake_GB_cam_logit)).to(
                                 self.device)
        self.loss_D_LB = self.MSE_loss(self.real_LB_logit, torch.ones_like(self.real_LB_logit)).to(self.device) + \
                         self.MSE_loss(self.fake_LB_logit, torch.zeros_like(self.fake_LB_logit)).to(self.device)
        self.loss_cam_D_LB = self.MSE_loss(self.real_LB_cam_logit, torch.ones_like(self.real_LB_cam_logit)).to(
            self.device) + \
                             self.MSE_loss(self.fake_LB_cam_logit, torch.zeros_like(self.fake_LB_cam_logit)).to(
                                 self.device)
        self.loss_D_A = self.adv_weight * (self.loss_D_GA + self.loss_D_LA + self.loss_cam_D_GA + self.loss_D_LA)
        self.loss_D_B = self.adv_weight * (self.loss_D_GB + self.loss_D_LB + self.loss_cam_D_GB + self.loss_D_LB)

        self.loss_D = self.loss_D_A + self.loss_D_B
        self.loss_D.backward()
        self.optimizer_D.step()

        # Update G
