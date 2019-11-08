import itertools
from .base_model import BaseModel
from .networks import *
from utils.util import str2bool


class UGATITModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True)
        if is_train:
            parser.add_argument('--adv_weight', type=float, default=1.0, help='weight for adversarial loss')
            parser.add_argument('--cycle_weight', type=float, default=1.0, help='weight for cycle loss')
            parser.add_argument('--identity_weight', type=float, default=1000.0, help='weight for identity loss')
            parser.add_argument('--cam_weight', type=float, default=1.0, help='weight for class activate map loss')
            parser.add_argument('--n_res', type=int, default=4, help='number of resblock')
            parser.add_argument('--n_dis', type=int, default=6, help='number of discriminator layer')
            parser.add_argument('--img_size', type=int, default=256, help='size of image')
            # parser.add_argument('--img_ch', type=int, default=3, help='channels of image')
            parser.add_argument('--netG', type=str, default='resnet_ugatit_6blocks',
                                help='specify generator architecture in ugatit [ resnet_ugatit_6blocks ]')
            parser.add_argument('--netD', type=str, default='UGATIT',
                                help='specify discriminator architecture in ugatit [UGATIT]')
            parser.add_argument('--light', type=str2bool, default=False,
                                help='use light model for UGATIT')

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
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
        visual_names_A = ['real_A', 'fake_A2B', 'fake_A2B2A', 'fake_A2A']
        visual_names_B = ['real_B', 'fake_B2A', 'fake_B2A2B', 'fake_B2B']
        self.visual_names = visual_names_A + visual_names_B

        self.model_names = ['genA2B', 'genB2A']
        if self.isTrain:
            self.model_names.extend(['disGA', 'disGB', 'disLA', 'disLB'])

        # define networks
        self.genA2B = define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, gpu_ids=self.gpu_ids, light=opt.light)
        self.genB2A = define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, gpu_ids=self.gpu_ids, light=opt.light)
        self.disGA = define_D(opt.input_nc, opt.ndf, opt.netD, norm='spectral_norm', n_layers=7, gpu_ids=self.gpu_ids)
        self.disGB = define_D(opt.input_nc, opt.ndf, opt.netD, norm='spectral_norm', n_layers=7, gpu_ids=self.gpu_ids)
        self.disLA = define_D(opt.input_nc, opt.ndf, opt.netD, norm='spectral_norm', n_layers=5, gpu_ids=self.gpu_ids)
        self.disLB = define_D(opt.input_nc, opt.ndf, opt.netD, norm='spectral_norm', n_layers=5, gpu_ids=self.gpu_ids)

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

        self.optimizer_D = torch.optim.Adam(
            itertools.chain(self.disGA.parameters(), self.disGB.parameters(),
                            self.disLA.parameters(), self.disLB.parameters()),
            lr=opt.lr,
            betas=(opt.beta1, 0.999),
            weight_decay=opt.weight_decay)

        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_D)
        self.RhoClipper = RhoClipper(0, 1)

        # weight
        self.adv_weight = opt.adv_weight
        self.cycle_weight = opt.cycle_weight
        self.identity_weight = opt.identity_weight
        self.cam_weight = opt.cam_weight

        self.real_A = None
        self.real_B = None
        self.fake_A2B = None
        self.fake_B2A = None
        self.fake_A2B2A = None
        self.fake_B2A2B = None
        self.fake_A2A = None
        self.fake_B2B = None

        self.loss_D_A = None
        self.loss_D_B = None
        self.loss_G_A = None
        self.loss_G_B = None
        self.loss_rec_G_A = None
        self.loss_rec_G_B = None
        self.loss_idt_G_A = None
        self.loss_idt_G_B = None
        self.loss_cam_G_A = None
        self.loss_cam_G_B = None

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

        real_GA_logit, real_GA_cam_logit, _ = self.disGA(self.real_A)
        real_LA_logit, real_LA_cam_logit, _ = self.disLA(self.real_A)
        real_GB_logit, real_GB_cam_logit, _ = self.disGB(self.real_B)
        real_LB_logit, real_LB_cam_logit, _ = self.disLB(self.real_B)

        fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(self.fake_B2A)
        fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(self.fake_B2A)
        fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(self.fake_A2B)
        fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(self.fake_A2B)

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
        self.fake_A2B, fake_A2B_cam_logit, _ = self.genA2B(self.real_A)
        self.fake_B2A, fake_B2A_cam_logit, _ = self.genB2A(self.real_B)

        self.fake_A2B2A, _, _ = self.genB2A(self.fake_A2B)
        self.fake_B2A2B, _, _ = self.genA2B(self.fake_B2A)

        self.fake_A2A, fake_A2A_cam_logit, _ = self.genB2A(self.real_A)
        self.fake_B2B, fake_B2B_cam_logit, _ = self.genA2B(self.real_B)

        fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(self.fake_B2A)
        fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(self.fake_B2A)
        fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(self.fake_A2B)
        fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(self.fake_A2B)

        G_ad_loss_GA = self.MSE_loss(fake_GA_logit, torch.ones_like(fake_GA_logit).to(self.device))
        G_ad_cam_loss_GA = self.MSE_loss(fake_GA_cam_logit, torch.ones_like(fake_GA_cam_logit).to(self.device))
        G_ad_loss_LA = self.MSE_loss(fake_LA_logit, torch.ones_like(fake_LA_logit).to(self.device))
        G_ad_cam_loss_LA = self.MSE_loss(fake_LA_cam_logit, torch.ones_like(fake_LA_cam_logit).to(self.device))
        G_ad_loss_GB = self.MSE_loss(fake_GB_logit, torch.ones_like(fake_GB_logit).to(self.device))
        G_ad_cam_loss_GB = self.MSE_loss(fake_GB_cam_logit, torch.ones_like(fake_GB_cam_logit).to(self.device))
        G_ad_loss_LB = self.MSE_loss(fake_LB_logit, torch.ones_like(fake_LB_logit).to(self.device))
        G_ad_cam_loss_LB = self.MSE_loss(fake_LB_cam_logit, torch.ones_like(fake_LB_cam_logit).to(self.device))

        self.loss_rec_G_A = self.L1_loss(self.fake_A2B2A, self.real_A)
        self.loss_rec_G_B = self.L1_loss(self.fake_B2A2B, self.real_B)

        self.loss_idt_G_A = self.L1_loss(self.fake_A2A, self.real_A)
        self.loss_idt_G_B = self.L1_loss(self.fake_B2B, self.real_B)

        self.loss_cam_G_A = self.BCE_loss(fake_B2A_cam_logit,
                                          torch.ones_like(fake_B2A_cam_logit).to(self.device)) + \
                            self.BCE_loss(fake_A2A_cam_logit,
                                          torch.zeros_like(fake_A2A_cam_logit).to(self.device))

        self.loss_cam_G_B = self.BCE_loss(fake_A2B_cam_logit,
                                          torch.ones_like(fake_A2B_cam_logit).to(self.device)) + \
                            self.BCE_loss(fake_B2B_cam_logit,
                                          torch.zeros_like(fake_B2B_cam_logit).to(self.device))

        self.loss_G_A = self.adv_weight * (G_ad_loss_GA + G_ad_cam_loss_GA + G_ad_loss_LA + G_ad_cam_loss_LA) + \
                        self.cycle_weight * self.loss_rec_G_A + self.identity_weight * self.loss_idt_G_A + \
                        self.cam_weight * self.loss_cam_G_A

        self.loss_G_B = self.adv_weight * (G_ad_loss_GB + G_ad_cam_loss_GB + G_ad_loss_LB + G_ad_cam_loss_LB) + \
                        self.cycle_weight * self.loss_rec_G_B + self.identity_weight * self.loss_idt_G_B + \
                        self.cam_weight * self.loss_cam_G_B

        loss_G = self.loss_G_A + self.loss_G_B
        loss_G.backward()
        self.optimizer_G.step()

        self.genA2B.apply(self.RhoClipper)
        self.genB2A.apply(self.RhoClipper)

    def optimize_parameters(self):
        self.forward()
