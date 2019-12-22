import torch
import itertools

from torch import nn

from .base_model import BaseModel
from . import networks


class DoubleCycleGANColorizationModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True)
        parser.set_defaults(dataset_mode='colorization')
        if is_train:
            parser.add_argument('--adv_weight', type=float, default=1.0, help='weight for adversarial loss')
            parser.add_argument('--dis_weight', type=float, default=0.25, help='weight for discriminator loss')
            parser.add_argument('--cyc_weight', type=float, default=10.0, help='weight for cycle loss')
            parser.add_argument('--idt_weight', type=float, default=5.0, help='weight for identity loss')
            parser.add_argument('--adv_color_weight', type=float, default=10.0, help='weight for adversarial loss')
            parser.add_argument('--dis_color_weight', type=float, default=1.0, help='weight for discriminator loss')
            parser.add_argument('--cyc_color_weight', type=float, default=100.0, help='weight for cycle loss')
            # parser.add_argument('--idt_color_weight', type=float, default=50.0, help='weight for identity loss')

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.loss_names = [
            'adv_G_A', 'adv_G_B',
            'rec_G_A', 'rec_G_B',
            'idt_G_A', 'idt_G_B',
            'D_A', 'D_B',
            'adv_G_A_C', 'adv_G_B_C',
            'rec_G_A_C', 'rec_G_B_C',
            # 'idt_G_A', 'idt_G_B',
            'D_A_C', 'D_B_C',
        ]
        visual_names_A = [
            'real_A_RGB',
            'real_A_Gray',
            # 'fake_A2B_RGB',
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
        visual_names_C = [
            'fake_A2B_RGB',
            'fake_A_Gray',
            'fake_A2B2A_Gray',
            'fake_A_RGB',
        ]

        self.visual_names = visual_names_A + visual_names_B + visual_names_C

        self.model_names = [
            'genA2B_S', 'genB2A_S',
            'genA2B_C', 'genB2A_C',
        ]
        if self.isTrain:
            self.model_names.extend(['disA_S', 'disB_S', 'disA_C', 'disB_C'])

        # define shape model
        self.genA2B_S = networks.define_G(1, 1, opt.ngf, opt.netG, opt.norm,
                                          not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        self.genB2A_S = networks.define_G(1, 1, opt.ngf, opt.netG, opt.norm,
                                          not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        # define color model
        self.genA2B_C = networks.define_G(1, 3, opt.ngf, opt.netG, opt.norm,
                                          not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        self.genB2A_C = networks.define_G(3, 1, opt.ngf, opt.netG, opt.norm,
                                          not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define discriminators
            self.disA_S = networks.define_D(1, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.disB_S = networks.define_D(1, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

            self.disA_C = networks.define_D(3, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.disB_C = networks.define_D(1, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions

            self.L1_loss = nn.L1Loss().to(self.device)
            self.MSE_loss = nn.MSELoss().to(self.device)

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G_S = torch.optim.Adam(
                itertools.chain(self.genA2B_S.parameters(), self.genB2A_S.parameters()),
                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_S = torch.optim.Adam(itertools.chain(self.disA_S.parameters(), self.disB_S.parameters()),
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizer_G_C = torch.optim.Adam(
                itertools.chain(self.genA2B_C.parameters(), self.genB2A_C.parameters()),
                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_C = torch.optim.Adam(itertools.chain(self.disA_C.parameters(), self.disB_C.parameters()),
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers.append(self.optimizer_G_S)
            self.optimizers.append(self.optimizer_D_S)
            self.optimizers.append(self.optimizer_G_C)
            self.optimizers.append(self.optimizer_D_C)

            self.adv_weight = opt.adv_weight
            self.dis_weight = opt.dis_weight
            self.cyc_weight = opt.cyc_weight
            self.idt_weight = opt.idt_weight
            self.adv_color_weight = opt.adv_color_weight
            self.dis_color_weight = opt.dis_color_weight
            self.cyc_color_weight = opt.cyc_color_weight

    def set_input(self, input):
        """
        Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        A2B = self.opt.direction in ['AtoB', 'A2B']
        self.real_A_RGB = input['A_RGB' if A2B else 'B_RGB'].to(self.device)
        self.real_B_RGB = input['B_RGB' if A2B else 'A_RGB'].to(self.device)
        self.real_A_Gray = input['A_Gray' if A2B else 'B_Gray'].to(self.device)
        self.real_B_Gray = input['B_Gray' if A2B else 'A_Gray'].to(self.device)
        self.image_paths = input['A_paths' if A2B else 'B_paths']

    def forward(self):

        self.forward_S()
        self.forward_C()

    def optimize_parameters(self):

        self.forward_S()

        # S
        self.set_requires_grad([self.disA_S, self.disA_S], False)
        self.optimizer_G_S.zero_grad()
        self.backward_G_S()
        self.optimizer_G_S.step()
        # D_A, D_B
        self.set_requires_grad([self.disA_S, self.disA_S], True)
        self.optimizer_D_S.zero_grad()
        self.backward_D_S()
        self.optimizer_D_S.step()

        self.forward_C()

        # C
        self.set_requires_grad([self.disA_C, self.disA_C], False)
        self.optimizer_G_C.zero_grad()
        self.backward_G_C()
        self.optimizer_G_C.step()
        # D_A, D_B
        self.set_requires_grad([self.disA_C, self.disA_C], True)
        self.optimizer_D_C.zero_grad()
        self.backward_D_C()
        self.optimizer_D_C.step()

    def forward_S(self):
        self.fake_A2B_Gray = self.genA2B_S(self.real_A_Gray)
        self.fake_B2A_Gray = self.genB2A_S(self.real_B_Gray)

        self.fake_A2B2A_Gray = self.genB2A_S(self.fake_A2B_Gray)
        self.fake_B2A2B_Gray = self.genA2B_S(self.fake_B2A_Gray)

        self.fake_A2A_Gray = self.genB2A_S(self.real_A_Gray)
        self.fake_B2B_Gray = self.genA2B_S(self.real_B_Gray)

    def backward_G_S(self):
        fake_A_logit = self.disA_S(self.fake_A2B_Gray)
        fake_B_logit = self.disB_S(self.fake_B2A_Gray)

        self.loss_adv_G_A = self.MSE_loss(fake_A_logit,
                                          torch.ones_like(fake_A_logit).to(self.device))
        self.loss_adv_G_B = self.MSE_loss(fake_B_logit,
                                          torch.ones_like(fake_B_logit).to(self.device))

        self.loss_rec_G_A = self.L1_loss(self.fake_A2B2A_Gray, self.real_A_Gray)
        self.loss_rec_G_B = self.L1_loss(self.fake_B2A2B_Gray, self.real_B_Gray)

        self.loss_idt_G_A = self.L1_loss(self.fake_A2A_Gray, self.real_A_Gray)
        self.loss_idt_G_B = self.L1_loss(self.fake_B2B_Gray, self.real_B_Gray)

        loss_G = (self.loss_adv_G_A + self.loss_adv_G_B) * self.adv_weight + \
                 (self.loss_rec_G_A + self.loss_rec_G_B) * self.cyc_weight + \
                 (self.loss_idt_G_A + self.loss_idt_G_B) * self.idt_weight

        loss_G.backward()

    def backward_D_S(self):
        real_A_logit = self.disB_S(self.real_A_Gray)
        fake_A_logit = self.disB_S(self.fake_B2A_Gray.detach())
        real_B_logit = self.disA_S(self.real_B_Gray)
        fake_B_logit = self.disA_S(self.fake_A2B_Gray.detach())

        self.loss_D_A = self.MSE_loss(real_B_logit, torch.ones_like(real_B_logit).to(self.device)) + \
                        self.MSE_loss(fake_B_logit, torch.zeros_like(fake_B_logit).to(self.device))

        self.loss_D_B = self.MSE_loss(real_A_logit, torch.ones_like(real_A_logit).to(self.device)) + \
                        self.MSE_loss(fake_A_logit, torch.zeros_like(fake_A_logit).to(self.device))

        loss_D = (self.loss_D_A + self.loss_D_B) * self.dis_weight
        loss_D.backward()

    def forward_C(self):
        # C
        self.fake_A2B_RGB = self.genA2B_C(self.fake_A2B_Gray.detach())
        self.fake_A_Gray = self.genB2A_C(self.real_A_RGB)

        self.fake_A2B2A_Gray = self.genB2A_C(self.fake_A2B_RGB)
        self.fake_A_RGB = self.genA2B_C(self.fake_A_Gray)

    def backward_G_C(self):
        fake_A_logit = self.disA_C(self.fake_A2B_RGB)
        fake_B_logit = self.disB_C(self.fake_A_Gray)

        self.loss_adv_G_A_C = self.MSE_loss(fake_A_logit,
                                            torch.ones_like(fake_A_logit).to(self.device))
        self.loss_adv_G_B_C = self.MSE_loss(fake_B_logit,
                                            torch.ones_like(fake_B_logit).to(self.device))

        self.loss_rec_G_A_C = self.L1_loss(self.fake_A2B2A_Gray, self.fake_A2B_Gray.detach())
        self.loss_rec_G_B_C = self.L1_loss(self.fake_A_RGB, self.real_A_RGB)

        # self.loss_idt_G_A = self.L1_loss(self.fake_A2A_Gray, self.real_A_Gray)
        # self.loss_idt_G_B = self.L1_loss(self.fake_B2B_Gray, self.real_B_Gray)

        loss_G_C = (self.loss_adv_G_A_C + self.loss_adv_G_B_C) * self.adv_color_weight + \
                   (self.loss_rec_G_A_C + self.loss_rec_G_B_C) * self.cyc_color_weight
        # (self.loss_idt_G_A + self.loss_idt_G_B) * self.idt_weight

        loss_G_C.backward()

    def backward_D_C(self):
        real_A_logit = self.disB_C(self.fake_A2B_Gray.detach())
        fake_A_logit = self.disB_C(self.fake_A_Gray.detach())
        real_B_logit = self.disA_C(self.real_A_RGB)
        fake_B_logit = self.disA_C(self.fake_A2B_RGB.detach())

        self.loss_D_A_C = self.MSE_loss(real_B_logit, torch.ones_like(real_B_logit).to(self.device)) + \
                          self.MSE_loss(fake_B_logit, torch.zeros_like(fake_B_logit).to(self.device))

        self.loss_D_B_C = self.MSE_loss(real_A_logit, torch.ones_like(real_A_logit).to(self.device)) + \
                          self.MSE_loss(fake_A_logit, torch.zeros_like(fake_A_logit).to(self.device))

        loss_D_C = (self.loss_D_A_C + self.loss_D_B_C) * self.dis_color_weight
        loss_D_C.backward()
