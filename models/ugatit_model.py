import itertools

from .base_model import BaseModel
from .networks import *


class UGATITModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True)
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
                                            lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=self.weight_decay)
        # TODO fix the lr_policy for naive weight decay

