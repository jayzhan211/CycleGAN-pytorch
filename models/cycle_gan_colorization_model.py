import itertools

from models import networks
from models.networks import define_G, define_D
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
            # parser.add_argument('--n_res', type=int, default=4, help='number of resblock')
            # parser.add_argument('--n_dis', type=int, default=6, help='number of discriminator layer')
            parser.add_argument('--img_size', type=int, default=256, help='size of image')
            parser.add_argument('--netG', type=str, default='resnet_6blocks_colorization',
                                help='specify generator architecture in ugatit [ resnet_ugatit_6blocks ]')
            parser.add_argument('--netD', type=str, default='discriminator_colorization',
                                help='specify discriminator architecture in ugatit [UGATIT]')
            parser.add_argument('--input_nc', type=int, default=1,
                                help='# of input image channels: 3 for RGB and 1 for grayscale')
            parser.add_argument('--output_nc', type=int, default=1,
                                help='# of output image channels: 3 for RGB and 1 for grayscale')

        return parser

    def __init__(self, opt):
        super(CycleGANColorizationModel, self).__init__(opt)
        self.loss_names = [
            'G_A', 'G_B',
            'D_A', 'D_B',
            'rec_G_A', 'rec_G_B',
            'idt_G_A', 'idt_G_B'
        ]
        visual_names_A = [
            'real_A_RGB',
            'real_A_Gray',
            'fake_A2B_RGB',
            'fake_A2B_Gray',
            'fake_A2B2A_RGB',
            'fake_A2B2A_Gray',
            'fake_A2A_RGB',
            'fake_A2A_Gray',
        ]
        visual_names_B = [
            'real_B_RGB',
            'real_B_Gray',
            'fake_B2A_RGB',
            'fake_B2A_Gray',
            'fake_B2A2B_RGB',
            'fake_B2A2B_Gray',
            'fake_B2B_RGB',
            'fake_B2B_Gray',
        ]
        self.visual_names = visual_names_A + visual_names_B
        self.model_names = ['genA2B', 'genB2A']
        if self.isTrain:
            # self.model_names.extend(['disGA', 'disGB', 'disLA', 'disLB'])
            self.model_names.extend(['disA', 'disB'])

        # define networks
        self.genA2B = define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, gpu_ids=self.gpu_ids)
        self.genB2A = define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, gpu_ids=self.gpu_ids)
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

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            self.L1_loss = nn.L1Loss().to(self.device)
            self.MSE_loss = nn.MSELoss().to(self.device)

        self.adv_weight = opt.adv_weight
        self.cycle_weight = opt.cycle_weight
        self.identity_weight = opt.identity_weight

        self.real_A_Gray = None
        self.real_B_Gray = None

        self.fake_A2B_Gray = None
        self.fake_A2B2A_Gray = None
        self.fake_B2A_Gray = None
        self.fake_B2A2B_Gray = None
        self.fake_A2A_Gray = None
        self.fake_B2B_Gray = None

        self.loss_adv_G_A = None
        self.loss_adv_G_B = None
        self.loss_rec_G_A = None
        self.loss_rec_G_B = None
        self.loss_idt_G_A = None
        self.loss_idt_G_B = None

    def set_input(self, _input):
        A2B = self.opt.direction in ['A2B', 'AtoB']
        self.real_A_RGB = _input['A_RGB' if A2B else 'B'].to(self.device)
        self.real_B_RGB = _input['B_RGB' if A2B else 'A'].to(self.device)
        self.real_A_Gray = _input['A_Gray']
        self.real_B_Gray = _input['B_Gray']
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

        loss_G = (self.loss_adv_G_A + self.loss_adv_G_B) * self.adv_weight +\
                 (self.loss_rec_G_A + self.loss_rec_G_B) * self.cycle_weight +\
                 (self.loss_idt_G_A + self.loss_idt_G_B) * self.identity_weight
        loss_G.backward()

    def backward_D(self):
        real_A_Gray_logit = self.disA(self.real_A_Gray)
        real_B_Gray_logit = self.disB(self.real_B_Gray)
        fake_A_Gray_logit = self.disA(self.fake_B2A_Gray.fatch())
        fake_B_Gray_logit = self.disB(self.fake_A2B_Gray.fatch())

        self.loss_adv_D_A = self.MSE_loss(real_A_Gray_logit, torch.ones_like(real_A_Gray_logit).to(self.device)) + \
                            self.MSE_loss(fake_A_Gray_logit, torch.zeros_like(fake_A_Gray_logit).to(self.device))

        self.loss_adv_D_B = self.MSE_loss(real_B_Gray_logit, torch.ones_like(real_B_Gray_logit).to(self.device)) + \
                            self.MSE_loss(fake_B_Gray_logit, torch.zeros_like(fake_B_Gray_logit).to(self.device))

        loss_D = (self.loss_adv_D_A + self.loss_adv_D_B) * 0.5
        loss_D.backward()










