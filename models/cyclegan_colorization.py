import torch
import itertools
from utils.image_pool import ImagePool
from .base_model import BaseModel
from .networks import *


class CyceGANColorizationModedl(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True)
        # if is_train:
        return parser

    def __init__(self, opt):
        super(CyceGANColorizationModedl, self).__init__(opt)
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
        self.disGA = define_D(opt.input_nc, opt.ndf, opt.netD, norm='spectral_norm', n_layers=7, gpu_ids=self.gpu_ids)
        self.disGB = define_D(opt.input_nc, opt.ndf, opt.netD, norm='spectral_norm', n_layers=7, gpu_ids=self.gpu_ids)
        self.disLA = define_D(opt.input_nc, opt.ndf, opt.netD, norm='spectral_norm', n_layers=5, gpu_ids=self.gpu_ids)
        self.disLB = define_D(opt.input_nc, opt.ndf, opt.netD, norm='spectral_norm', n_layers=5, gpu_ids=self.gpu_ids)
