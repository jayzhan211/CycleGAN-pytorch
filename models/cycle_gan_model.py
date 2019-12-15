import torch
import itertools

from torch import nn

from .base_model import BaseModel
from . import networks


class CycleGANModel(BaseModel):
    """
        This class implements the CycleGAN model, for learning image-to-image translation without paired data.
        The model training requires '--dataset_mode unaligned' dataset.
        By default, it uses a '--netG resnet_9blocks' ResNet generator,
        a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
        and a least-square GANs objective ('--gan_mode lsgan').
        CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """
        Add new dataset-specific options, and rewrite default values for existing options.
        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.
        Returns:
            the modified parser.
        For CycleGAN-pytorch, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN-pytorch paper.
        :param parser:
        """
        parser.set_defaults(no_dropout=True)
        if is_train:
            parser.add_argument('--adv_weight', type=float, default=1.0, help='weight for adversarial loss')
            parser.add_argument('--dis_weight', type=float, default=0.5, help='weight for discriminator loss')
            parser.add_argument('--cyc_weight', type=float, default=10.0, help='weight for cycle loss')
            parser.add_argument('--idt_weight', type=float, default=5.0, help='weight for identity loss')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN-pytorch class.
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)

        self.loss_names = [
            'adv_G_A', 'adv_G_B',
            'rec_G_A', 'rec_G_B',
            'idt_G_A', 'idt_G_B',
            'D_A', 'D_B',
        ]
        visual_names_A = [
            'real_A',
            'fake_A2B',
            'fake_A2B2A',
            'fake_A2A',
        ]
        visual_names_B = [
            'real_B',
            'fake_B2A',
            'fake_B2A2B',
            'fake_B2B',
        ]
        self.visual_names = visual_names_A + visual_names_B

        self.model_names = ['genA2B', 'genB2A']
        if self.isTrain:
            self.model_names.extend(['disA', 'disB'])

        # define network

        self.genA2B = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.genB2A = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        # self.genA2B = networks.define_G(opt.input_nc, opt.output_nc, ngf=opt.ngf, netG=opt.netG, gpu_ids=self.gpu_ids,
        #                                 norm=opt.norm, use_dropout=not opt.no_dropout,
        #                                 init_type=opt.init_type, init_gain=opt.init_gain)
        # self.genB2A = networks.define_G(opt.output_nc, opt.input_nc, ngf=opt.ngf, netG=opt.netG, gpu_ids=self.gpu_ids,
        #                                 norm=opt.norm, use_dropout=not opt.no_dropout,
        #                                 init_type=opt.init_type, init_gain=opt.init_gain)

        if self.isTrain:  # define discriminators
            self.disA = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.disB = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:

            # define loss functions
            # self.criterionGAN = networks.GANLoss(gan_mode=opt.gan_mode).to(self.device)  # define GAN loss.

            self.L1_loss = nn.L1Loss().to(self.device)
            self.MSE_loss = nn.MSELoss().to(self.device)

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.genA2B.parameters(), self.genB2A.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.disA.parameters(), self.disB.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

        self.adv_weight = opt.adv_weight
        self.dis_weight = opt.dis_weight
        self.cyc_weight = opt.cyc_weight
        self.idt_weight = opt.idt_weight

    def set_input(self, input):
        """
        Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == ['AtoB', 'A2B']
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.fake_A2B = self.genA2B(self.real_A)
        self.fake_B2A = self.genB2A(self.real_B)
        self.fake_A2B2A = self.genB2A(self.fake_A2B)
        self.fake_B2A2B = self.genA2B(self.fake_B2A)
        self.fake_A2A = self.genB2A(self.real_A)
        self.fake_B2B = self.genA2B(self.real_B)

    def backward_G(self):
        """
        Calculate the loss for generators G_A and G_B
        """
        fake_A_logit = self.disA(self.fake_A2B)
        fake_B_logit = self.disB(self.fake_B2A)

        self.loss_adv_G_A = self.MSE_loss(fake_A_logit,
                                          torch.ones_like(fake_A_logit).to(self.device))
        self.loss_adv_G_B = self.MSE_loss(fake_B_logit,
                                          torch.ones_like(fake_B_logit).to(self.device))

        self.loss_rec_G_A = self.L1_loss(self.fake_A2B2A, self.real_A)
        self.loss_rec_G_B = self.L1_loss(self.fake_B2A2B, self.real_B)

        self.loss_idt_G_A = self.L1_loss(self.fake_A2A, self.real_A)
        self.loss_idt_G_B = self.L1_loss(self.fake_B2B, self.real_B)

        loss_G = (self.loss_adv_G_A + self.loss_adv_G_B) * self.adv_weight + \
                 (self.loss_rec_G_A + self.loss_rec_G_B) * self.cyc_weight + \
                 (self.loss_idt_G_A + self.loss_idt_G_B) * self.idt_weight
        
        loss_G.backward()

    def backward_D(self):
        real_A_logit = self.disB(self.real_A)
        fake_A_logit = self.disB(self.fake_B2A.detach())
        real_B_logit = self.disA(self.real_B)
        fake_B_logit = self.disA(self.fake_A2B.detach())

        self.loss_D_A = self.MSE_loss(real_B_logit, torch.ones_like(real_B_logit).to(self.device)) + \
                        self.MSE_loss(fake_B_logit, torch.zeros_like(fake_B_logit).to(self.device))

        self.loss_D_B = self.MSE_loss(real_A_logit, torch.ones_like(real_A_logit).to(self.device)) + \
                            self.MSE_loss(fake_A_logit, torch.zeros_like(fake_A_logit).to(self.device))

        loss_D = (self.loss_D_A + self.loss_D_B) * self.dis_weight
        loss_D.backward()

    def optimize_parameters(self):

        self.forward()

        self.set_requires_grad([self.disA, self.disA], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # D_A, D_B
        self.set_requires_grad([self.disA, self.disA], True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
