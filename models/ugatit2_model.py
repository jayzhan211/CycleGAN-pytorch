import itertools

from torch import nn

from utils.util import tensor2im
from .base_model import BaseModel
from .networks_ugatit import Generator, Encoder
import torch


class UGATIT2Model(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(display_nrows=4)

        parser.add_argument('--img_size', type=int, default=256, help='size of image')
        parser.add_argument('--init-f', type=int, default=32, help='init filters number.')
        parser.add_argument('--max-f', type=int, default=256, help='max filters number.')
        parser.add_argument('--num-layers', type=int, default=6, help='init size=4, result resolution is 4 * (2 ** '
                                                                      'num_layers).')
        parser.add_argument('--latent-size', type=int, default=256, help='latent space (W space) dimension.')
        parser.add_argument('--mapping-layers', type=int, default=3, help='input channel')
        parser.add_argument('--num_channels', type=int, default=3, help='input channel')

        if is_train:
            parser.add_argument('--adv_weight', type=float, default=1.0, help='weight for adversarial loss')
            parser.add_argument('--cycle_weight', type=float, default=10.0, help='weight for cycle loss')
            parser.add_argument('--identity_weight', type=float, default=10.0, help='weight for identity loss')
            parser.add_argument('--weight_decay', type=float, default=1e-4, help='the weight decay in adam optimizer')

        return parser

    def __init__(self, opt):
        super(UGATIT2Model, self).__init__(opt)
        init_f = opt.init_f
        max_f = opt.max_f
        latent_size = opt.latent_size
        num_layers = opt.num_layers
        num_channels = opt.num_channels
        lr = opt.lr
        beta1 = opt.beta1
        beta2 = opt.beta2
        weight_decay = opt.weight_decay

        ###################################
        # Model
        ###################################
        if self.isTrain:
            self.disA = Encoder(latent_size=latent_size, num_layers=num_layers).to(self.device)
            self.disB = Encoder(latent_size=latent_size, num_layers=num_layers).to(self.device)
        self.genA2B = Generator(num_layers=num_layers, latent_size=latent_size).to(self.device)
        self.genB2A = Generator(num_layers=num_layers, latent_size=latent_size).to(self.device)

        ###################################
        # Optimizer
        ###################################
        if self.isTrain:
            self.optimizer_G = torch.optim.Adam(
                itertools.chain(self.genA2B.parameters(), self.genB2A.parameters()),
                lr=lr,
                betas=(beta1, beta2),
                weight_decay=weight_decay)

            self.optimizer_D = torch.optim.Adam(
                itertools.chain(self.disA.parameters(), self.disB.parameters()),
                lr=lr,
                betas=(beta1, beta2),
                weight_decay=weight_decay)

        ###################################
        # Losses
        ###################################
        if self.isTrain:
            self.L1_loss = nn.L1Loss().to(self.device)
            self.MSE_loss = nn.MSELoss().to(self.device)
            self.BCE_loss = nn.BCEWithLogitsLoss().to(self.device)

            self.adv_weight = opt.adv_weight
            self.cycle_weight = opt.cycle_weight
            self.identity_weight = opt.identity_weight

        ###################################
        #
        ###################################
        self.num_layers = num_layers
        self.latent_size = latent_size
        self.direction = opt.direction

        self.loss_names = ['G', 'D']
        visual_names_A = ['realA', 'fakeA2B', 'fakeA2A', 'fakeA2B2A']
        visual_names_B = ['realB', 'fakeB2A', 'fakeB2B', 'fakeB2A2B']
        self.visual_names = visual_names_A + visual_names_B

        self.model_names = ['genA2B', 'genB2A']
        if self.isTrain:
            self.model_names.extend(['disA', 'disB'])

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        AtoB = self.direction in ['AtoB']
        self.realA = input['A' if AtoB else 'B'].to(self.device)
        self.realB = input['B' if AtoB else 'A'].to(self.device)

    def test(self):
        with torch.no_grad():
            realA_z, stylesA, realA_logit = self.disA(self.realA)
            realB_z, stylesB, realB_logit = self.disB(self.realB)

            fakeA2B = self.genA2B(realA_z, stylesA)
            fakeB2A = self.genB2A(realB_z, stylesB)

            fakeB2A_z, styles_fakeB2A, fakeB2A_logit = self.disA(fakeB2A)
            fakeA2B_z, styles_fakeA2B, fakeA2B_logit = self.disA(fakeA2B)

            fakeA2B2A = self.genB2A(fakeA2B_z, styles_fakeA2B)
            fakeB2A2B = self.genA2B(fakeB2A_z, styles_fakeB2A)

            fakeA2A = self.genB2A(realA_z, stylesA)
            fakeB2B = self.genA2B(realB_z, stylesB)

        self.fakeA2B = tensor2im(fakeA2B)
        self.fakeB2A = tensor2im(fakeB2A)
        self.fakeB2A2B = tensor2im(fakeB2A2B)
        self.fakeA2B2A = tensor2im(fakeA2B2A)
        self.fakeB2B = tensor2im(fakeB2B)
        self.fakeA2A = tensor2im(fakeA2A)

    def train(self, display):
        ###################################
        # Encoder and Discriminator
        ###################################
        self.optimizer_D.zero_grad()

        realA_z, stylesA, realA_logit = self.disA(self.realA)
        realB_z, stylesB, realB_logit = self.disB(self.realB)

        fakeA2B = self.genA2B(realA_z, stylesA).detach()
        fakeB2A = self.genB2A(realB_z, stylesB).detach()

        _, _, fakeB2A_logit = self.disA(fakeB2A)
        _, _, fakeA2B_logit = self.disA(fakeA2B)

        D_adv_loss_A = self.MSE_loss(realA_logit, torch.ones_like(realA_logit).to(self.device)) + self.MSE_loss(
            fakeB2A_logit, torch.zeros_like(fakeB2A_logit).to(self.device))

        D_adv_loss_B = self.MSE_loss(realB_logit, torch.ones_like(realA_logit).to(self.device)) + self.MSE_loss(
            fakeA2B_logit, torch.zeros_like(fakeA2B_logit).to(self.device))

        loss_D_A = self.adv_weight * D_adv_loss_A
        loss_D_B = self.adv_weight * D_adv_loss_B

        self.loss_D = loss_D_A + loss_D_B
        self.loss_D.backward()
        self.optimizer_D.step()

        ###################################
        # Generator
        ###################################

        self.optimizer_G.zero_grad()

        realA_z, stylesA, realA_logit = self.disA(self.realA)
        realB_z, stylesB, realB_logit = self.disB(self.realB)

        fakeA2B = self.genA2B(realA_z, stylesA)
        fakeB2A = self.genB2A(realB_z, stylesB)

        fakeB2A_z, styles_fakeB2A, fakeB2A_logit = self.disA(fakeB2A)
        fakeA2B_z, styles_fakeA2B, fakeA2B_logit = self.disA(fakeA2B)

        fakeA2B2A = self.genB2A(fakeA2B_z, styles_fakeA2B)
        fakeB2A2B = self.genA2B(fakeB2A_z, styles_fakeB2A)

        fakeA2A = self.genB2A(realA_z, stylesA)
        fakeB2B = self.genA2B(realB_z, stylesB)

        G_adv_loss_A = self.MSE_loss(fakeB2A_logit, torch.ones_like(fakeB2A_logit).to(self.device))
        G_adv_loss_B = self.MSE_loss(fakeA2B_logit, torch.ones_like(fakeA2B_logit).to(self.device))

        G_cyc_loss_A = self.L1_loss(self.realA, fakeA2B2A)
        G_cyc_loss_B = self.L1_loss(self.realB, fakeB2A2B)

        G_idt_loss_A = self.L1_loss(self.realA, fakeA2A)
        G_idt_loss_B = self.L1_loss(self.realB, fakeB2B)

        loss_G_A = self.adv_weight * G_adv_loss_A + self.cycle_weight * G_cyc_loss_A + self.identity_weight * G_idt_loss_A
        loss_G_B = self.adv_weight * G_adv_loss_B + self.cycle_weight * G_cyc_loss_B + self.identity_weight * G_idt_loss_B

        self.loss_G = loss_G_A + loss_G_B
        self.loss_G.backward()
        self.optimizer_G.step()

        if display:
            self.fakeA2B = tensor2im(fakeA2B)
            self.fakeB2A = tensor2im(fakeB2A)
            self.fakeB2A2B = tensor2im(fakeB2A2B)
            self.fakeA2B2A = tensor2im(fakeA2B2A)
            self.fakeB2B = tensor2im(fakeB2B)
            self.fakeA2A = tensor2im(fakeA2A)

