import itertools
import torch.nn as nn
import torch
import torchvision.transforms as transforms
from models.networks import define_G, define_D, define_Net
from .base_model import BaseModel



class CycleGANVggModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(dataset_mode='unaligned')
        if is_train:
            parser.add_argument('--adv_weight', type=float, default=1.0, help='weight for adversarial loss')
            parser.add_argument('--cycle_weight', type=float, default=10.0, help='weight for cycle loss')
            parser.add_argument('--identity_weight', type=float, default=10.0, help='weight for identity loss')
            parser.add_argument('--content_weight', type=float, default=0.0, help='weight for content loss')
            # parser.add_argument('--style_weight', type=float, default=0.2, help='weight for style loss')
            # parser.add_argument('--rec_weight', type=float, default=1.0, help='weight for style loss')

            # parser.add_argument('--img_size', type=int, default=256, help='size of image')
            parser.add_argument('--netG', type=str, default='resnet_9blocks',
                                help='specify generator architecture')
            parser.add_argument('--netD', type=str, default='n_layers',
                                help='specify discriminator architecture')
            parser.add_argument('--n_layers_D', type=int, default=5, help='only used if netD == n_layers')
            parser.add_argument('--netVgg', type=str, default='vgg19',
                                help='specify vgg pretrained model [vgg19]')

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.loss_names = [
            'adv_G_A', 'adv_G_B',
            'rec_G_A', 'rec_G_B',
            'idt_G_A', 'idt_G_B',
            'con_G_A', 'con_G_B',

            'adv_D_A', 'adv_D_B',
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

        # define networks
        self.genA2B = define_G(opt.input_nc, opt.output_nc, ngf=opt.ngf, netG=opt.netG, gpu_ids=self.gpu_ids,
                               norm=opt.norm, use_dropout=not opt.no_dropout,
                               init_type=opt.init_type, init_gain=opt.init_gain)

        self.genB2A = define_G(opt.output_nc, opt.input_nc, ngf=opt.ngf, netG=opt.netG, gpu_ids=self.gpu_ids,
                               norm=opt.norm, use_dropout=not opt.no_dropout,
                               init_type=opt.init_type, init_gain=opt.init_gain)

        self.netVgg = define_Net(netType=opt.netVgg, gpu_ids=self.gpu_ids)

        if self.isTrain:
            self.disA = define_D(opt.output_nc, opt.ndf, opt.netD, n_layers=opt.n_layers_D,
                                 norm_type='spectral_norm', padding_type='reflect',
                                 init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids)
            self.disB = define_D(opt.input_nc, opt.ndf, opt.netD, n_layers=opt.n_layers_D,
                                 norm_type='spectral_norm', padding_type='reflect',
                                 init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids)

            self.optimizer_G = torch.optim.Adam(
                itertools.chain(self.genA2B.parameters(), self.genB2A.parameters()),
                lr=opt.lr,
                betas=(opt.beta1, 0.999),
                weight_decay=opt.weight_decay,
            )

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
        self.content_weight = opt.content_weight
        # self.style_weight = opt.style_weight
        # self.rec_weight = opt.rec_weight

    def set_input(self, _input):
        A2B = self.opt.direction in ['A2B', 'AtoB']
        self.real_A = _input['A' if A2B else 'B'].to(self.device)
        self.real_B = _input['B' if A2B else 'A'].to(self.device)
        self.image_paths = _input['A_paths' if A2B else 'B_paths']

    def forward(self):
        """
        Run forward pass; called by both functions <optimize_parameters> and <test>.
        """
        self.fake_A2B = self.genA2B(self.real_A)
        self.fake_B2A = self.genA2B(self.real_B)
        self.fake_A2B2A = self.genB2A(self.fake_A2B)
        self.fake_B2A2B = self.genA2B(self.fake_B2A)
        self.fake_A2A = self.genB2A(self.real_A)
        self.fake_B2B = self.genA2B(self.real_B)


    def optimize_parameters(self):
        """
        Calculate losses, gradients, and update network weights
        called in every training iteration
        """
        self.forward()

        # D
        self.set_requires_grad([self.disA, self.disB], True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        # G
        self.set_requires_grad([self.disA, self.disB], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()




    def backward_G(self):
        # forward

        # fake_A2B = self.genA2B(self.real_A)
        # fake_B2A = self.genA2B(self.real_B)
        #
        # fake_A2B2A = self.genB2A(fake_A2B)
        # fake_B2A2B = self.genA2B(fake_B2A)
        #
        # fake_A2A = self.genB2A(self.real_A)
        # fake_B2B = self.genA2B(self.real_B)

        fake_GA_logit = self.disA(self.fake_B2A)
        fake_GB_logit = self.disB(self.fake_A2B)

        fake_A_feats = self.netVgg(self.real_A)
        fake_A2B_feats = self.netVgg(self.fake_A2B.detach())
        fake_B_feats = self.netVgg(self.real_B)
        fake_B2A_feats = self.netVgg(self.fake_B2A.detach())

        # loss

        loss_adv_G_A = self.MSE_loss(fake_GA_logit, torch.ones_like(fake_GA_logit).to(self.device))
        loss_adv_G_B = self.MSE_loss(fake_GB_logit, torch.ones_like(fake_GB_logit).to(self.device))

        loss_rec_G_A = self.L1_loss(self.fake_A2B2A, self.real_A)
        loss_rec_G_B = self.L1_loss(self.fake_B2A2B, self.real_B)

        loss_idt_G_A = self.L1_loss(self.fake_A2A, self.real_A)
        loss_idt_G_B = self.L1_loss(self.fake_B2B, self.real_B)

        loss_con_G_A = 0.0
        loss_con_G_B = 0.0

        for i in range(4):
            loss_con_G_A += self.calc_content_loss(fake_A_feats[i], fake_A2B_feats[i])
            loss_con_G_B += self.calc_content_loss(fake_B_feats[i], fake_B2A_feats[i])

        loss_G = (loss_adv_G_A + loss_adv_G_B) * self.adv_weight + \
                 (loss_rec_G_A + loss_rec_G_B) * self.cycle_weight + \
                 (loss_idt_G_A + loss_idt_G_B) * self.identity_weight + \
                 (loss_con_G_A + loss_con_G_B) * self.content_weight

        # self.fake_A2B = fake_A2B
        # self.fake_B2A = fake_B2A
        # self.fake_A2B2A = fake_A2B2A
        # self.fake_B2A2B = fake_B2A2B
        # self.fake_A2A = fake_A2A
        # self.fake_B2B = fake_B2B

        self.loss_adv_G_A = loss_adv_G_A
        self.loss_adv_G_B = loss_adv_G_B
        self.loss_rec_G_A = loss_rec_G_A
        self.loss_rec_G_B = loss_rec_G_B
        self.loss_idt_G_A = loss_idt_G_A
        self.loss_idt_G_B = loss_idt_G_B
        self.loss_con_G_A = loss_con_G_A
        self.loss_con_G_B = loss_con_G_B

        loss_G.backward()

    def calc_content_loss(self, input, target):
        assert (input.size() == target.size())
        return self.MSE_loss(input, target)

    def backward_D(self):

        # fake_A2B = self.genA2B(self.real_A)
        # fake_B2A = self.genA2B(self.real_B)

        real_GA_logit = self.disA(self.real_A)
        real_GB_logit = self.disB(self.real_B)

        fake_GA_logit = self.disA(self.fake_B2A.detach())
        fake_GB_logit = self.disB(self.fake_A2B.detach())

        loss_adv_D_A = self.MSE_loss(fake_GA_logit, torch.zeros_like(fake_GA_logit).to(self.device)) + \
            self.MSE_loss(real_GA_logit, torch.ones_like(real_GA_logit).to(self.device))
        loss_adv_D_B = self.MSE_loss(fake_GB_logit, torch.zeros_like(fake_GB_logit).to(self.device)) + \
                       self.MSE_loss(real_GB_logit, torch.ones_like(real_GB_logit).to(self.device))

        loss_D = (loss_adv_D_A + loss_adv_D_B) * self.adv_weight

        self.loss_adv_D_A = loss_adv_D_A
        self.loss_adv_D_B = loss_adv_D_B

        loss_D.backward()




