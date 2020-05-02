import itertools

from torch import nn

from utils.util import tensor2im
from .base_model import BaseModel
from .networks_alae import Encoder, Generator, Discriminator
import torch


class ALAEModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(display_nrows=4)

        parser.add_argument('--img_size', type=int, default=256, help='size of image')
        parser.add_argument('--init-f', type=int, default=32, help='init filters number.')
        parser.add_argument('--max-f', type=int, default=256, help='max filters number.')
        parser.add_argument('--num-layers', type=int, default=7, help='init size=4, result resolution is 4 * (2 ** '
                                                                      'num_layers).')
        parser.add_argument('--latent-size', type=int, default=128, help='latent space (W space) dimension.')
        parser.add_argument('--mapping-layers', type=int, default=3, help='input channel')
        parser.add_argument('--num_channels', type=int, default=3, help='input channel')

        if is_train:
            parser.add_argument('--adv_weight', type=float, default=1.0, help='weight for adversarial loss')
            parser.add_argument('--cycle_weight', type=float, default=10.0, help='weight for cycle loss')
            parser.add_argument('--identity_weight', type=float, default=10.0, help='weight for identity loss')
            parser.add_argument('--weight_decay', type=float, default=0, help='the weight decay in adam optimizer')

        return parser

    def __init__(self, opt):
        super(ALAEModel, self).__init__(opt)
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

        # encode image A to W_A
        self.encoderA = Encoder(init_f, max_f, latent_size, num_layers, num_channels).to(self.device)
        # encode image B to W_B
        self.encoderB = Encoder(init_f, max_f, latent_size, num_layers, num_channels).to(self.device)
        # generate fakeA2B with W_A
        self.decoderA2B = Generator(init_f, max_f, num_layers, latent_size, num_channels).to(self.device)
        # generate fakeB2A with W_B
        self.decoderB2A = Generator(init_f, max_f, num_layers, latent_size, num_channels).to(self.device)
        # discriminate realA vs fakeB2A
        self.discriminatorA = Discriminator(mapping_layers=3, latent_size=latent_size, mapping_fmaps=latent_size).to(self.device)
        # discriminate realB vs fakeA2B
        self.discriminatorB = Discriminator(mapping_layers=3, latent_size=latent_size, mapping_fmaps=latent_size).to(self.device)

        ###################################
        # Optimizer
        ###################################

        self.encoder_optimizer = torch.optim.Adam(
            itertools.chain(self.encoderA.parameters(), self.discriminatorA.parameters(), self.encoderB.parameters(),
                            self.discriminatorB.parameters()),
            lr=lr,
            betas=(beta1, beta2),
            weight_decay=weight_decay,
        )

        self.decoder_optimizer = torch.optim.Adam(
            itertools.chain(self.decoderA2B.parameters(), self.decoderB2A.parameters()),
            lr=lr,
            betas=(beta1, beta2),
            weight_decay=weight_decay,
        )

        ###################################
        # Losses
        ###################################
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
        self.loss_names = ['G', 'D', 'ae']

        visual_names_A = ['realA', 'fakeA2B', 'fakeA2A', 'fakeA2B2A']
        visual_names_B = ['realB', 'fakeB2A', 'fakeB2B', 'fakeB2A2B']
        self.visual_names = visual_names_A + visual_names_B

        self.model_names = ['encoderA','encoderB','decoderA2B','decoderB2A','discriminatorA','discriminatorB']

        self.optimizers.append(self.encoder_optimizer)
        self.optimizers.append( self.decoder_optimizer)

    def set_input(self, input):
        AtoB = self.direction in ['AtoB']
        self.realA = input['A' if AtoB else 'B'].to(self.device)
        self.realB = input['B' if AtoB else 'A'].to(self.device)

    def generate(self, stylesA, stylesB, noise=True, return_style=False):
        sA = stylesA.view(stylesA.size(0), 1, stylesA.size(1))
        stylesA = sA.repeat(1, self.num_layers * 2, 1)
        sB = stylesB.view(stylesB.size(0), 1, stylesB.size(1))
        stylesB = sB.repeat(1, self.num_layers * 2, 1)

        fakeA2B = self.decoderA2B(stylesA, noise)
        fakeB2A = self.decoderB2A(stylesB, noise)
        if return_style:
            return {
                'fakeA2B': fakeA2B,
                'fakeB2A': fakeB2A,
                'stylesA': stylesA,
                'stylesB': stylesB,
            }
        else:
            return {
                'fakeA2B': fakeA2B,
                'fakeB2A': fakeB2A,
            }

    def encodeA(self, x):
        w_A = self.encoderA(x)
        w_A_cls = self.discriminatorA(w_A)
        return w_A, w_A_cls

    def encodeB(self, x):
        w_B = self.encoderB(x)
        w_B_cls = self.discriminatorB(w_B)
        return w_B, w_B_cls

    def test(self):
        with torch.no_grad():
            stylesA = self.encoderA(self.realA)
            stylesB = self.encoderB(self.realB)
            res = self.generate(stylesA, stylesB, noise=True)
            fakeA2B = res['fakeA2B']
            fakeB2A = res['fakeB2A']

            fakeB2A_styles = self.encoderA(fakeB2A)
            fakeA2B_styles = self.encoderB(fakeA2B)

            res = self.generate(fakeB2A_styles, fakeA2B_styles, noise=True)
            fakeB2A2B = res['fakeA2B']
            fakeA2B2A = res['fakeB2A']

            res = self.generate(stylesB, stylesA, noise=True)
            fakeB2B = res['fakeA2B']
            fakeA2A = res['fakeB2A']

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
        self.encoder_optimizer.zero_grad()

        with torch.no_grad():
            stylesA = self.encoderA(self.realA)
            stylesB = self.encoderB(self.realB)
            res = self.generate(stylesA, stylesB, noise=True)
            fakeA2B = res['fakeA2B']
            fakeB2A = res['fakeB2A']

        # self.set_requires_grad([self.encoderA, self.encoderB], requires_grad=True)

        _, realA_logit = self.encodeA(self.realA)
        _, fakeB2A_logit = self.encodeA(fakeB2A.detach())

        _, realB_logit = self.encodeB(self.realB)
        _, fakeA2B_logit = self.encodeB(fakeA2B.detach())

        D_adv_loss_A = self.MSE_loss(realA_logit, torch.ones_like(realA_logit).to(self.device)) + self.MSE_loss(
            fakeB2A_logit, torch.zeros_like(fakeB2A_logit).to(self.device))

        D_adv_loss_B = self.MSE_loss(realB_logit, torch.ones_like(realA_logit).to(self.device)) + self.MSE_loss(
            fakeA2B_logit, torch.zeros_like(fakeA2B_logit).to(self.device))

        loss_D_A = D_adv_loss_A
        loss_D_B = D_adv_loss_B

        self.loss_D = loss_D_A + loss_D_B
        self.loss_D.backward()
        self.encoder_optimizer.step()

        ###################################
        # Generator
        ###################################

        self.decoder_optimizer.zero_grad()
        with torch.no_grad():
            stylesA = self.encoderA(self.realA)
            stylesB = self.encoderB(self.realB)

        res = self.generate(stylesA, stylesB, noise=True)
        fakeA2B = res['fakeA2B']
        fakeB2A = res['fakeB2A']

        fakeB2A_styles, fakeB2A_logit = self.encodeA(fakeB2A.detach())
        fakeA2B_styles, fakeA2B_logit = self.encodeB(fakeA2B.detach())

        # cycle
        res = self.generate(fakeB2A_styles.detach(), fakeA2B_styles.detach(), noise=True)
        fakeB2A2B = res['fakeA2B']
        fakeA2B2A = res['fakeB2A']

        # identity
        res = self.generate(stylesB, stylesA, noise=True)
        fakeB2B = res['fakeA2B']
        fakeA2A = res['fakeB2A']

        with torch.no_grad():
            fakeA2B2A_styles = self.encoderA(fakeA2B2A)
            fakeB2A2B_styles = self.encoderB(fakeB2A2B)

            fakeA2A_styles = self.encoderA(fakeA2A)
            fakeB2B_styles = self.encoderB(fakeB2B)

        G_adv_loss_A = self.MSE_loss(fakeB2A_logit, torch.ones_like(fakeB2A_logit).to(self.device))
        G_adv_loss_B = self.MSE_loss(fakeA2B_logit, torch.ones_like(fakeA2B_logit).to(self.device))

        G_cyc_loss_A = self.L1_loss(stylesA, fakeA2B2A_styles)
        G_cyc_loss_B = self.L1_loss(stylesB, fakeB2A2B_styles)

        G_idt_loss_A = self.L1_loss(stylesA, fakeA2A_styles)
        G_idt_loss_B = self.L1_loss(stylesB, fakeB2B_styles)

        loss_G_A = self.adv_weight * G_adv_loss_A + self.cycle_weight * G_cyc_loss_A + self.identity_weight * G_idt_loss_A
        loss_G_B = self.adv_weight * G_adv_loss_B + self.cycle_weight * G_cyc_loss_B + self.identity_weight * G_idt_loss_B

        self.loss_G = loss_G_A + loss_G_B
        self.loss_G.backward()
        self.decoder_optimizer.step()

        ###################################
        # AutoEncoder
        ###################################

        self.decoder_optimizer.zero_grad()
        self.encoder_optimizer.zero_grad()

        res = self.generate(stylesA, stylesB, noise=True)
        fakeA2B = res['fakeA2B']
        fakeB2A = res['fakeB2A']

        fakeB2A_styles = self.encoderA(fakeB2A)
        fakeA2B_styles = self.encoderB(fakeA2B)

        loss_ae_A = self.MSE_loss(stylesA, fakeB2A_styles)
        loss_ae_B = self.MSE_loss(stylesB, fakeA2B_styles)

        self.loss_ae = loss_ae_A + loss_ae_B
        self.loss_ae.backward()
        self.decoder_optimizer.step()
        self.encoder_optimizer.step()

        if display:
            self.fakeA2B = tensor2im(fakeA2B)
            self.fakeB2A = tensor2im(fakeB2A)
            self.fakeB2A2B = tensor2im(fakeB2A2B)
            self.fakeA2B2A = tensor2im(fakeA2B2A)
            self.fakeB2B = tensor2im(fakeB2B)
            self.fakeA2A = tensor2im(fakeA2A)

