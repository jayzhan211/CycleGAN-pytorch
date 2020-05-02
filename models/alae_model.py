import itertools

from torch import nn

from .base_model import BaseModel
from .networks_alae import Encoder, Generator, Discriminator
import torch


class ALAEModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--init-f', type=int, default=32, help='init filters number.')
        parser.add_argument('--max-f', type=int, default=256, help='max filters number.')
        parser.add_argument('--num-layers', type=int, default=6, help='init size=4, result resolution is 4 * (2 ** '
                                                                      'num_layers).')
        parser.add_argument('--latent-size', type=int, default=128, help='latent space (W space) dimension.')
        parser.add_argument('--mapping-layers', type=int, default=3, help='input channel')
        parser.add_argument('--num_channels', type=int, default=3, help='input channel')

        if is_train:
            parser.add_argument('--adv_weight', type=float, default=1.0, help='weight for adversarial loss')
            parser.add_argument('--cycle_weight', type=float, default=10.0, help='weight for cycle loss')
            parser.add_argument('--identity_weight', type=float, default=10.0, help='weight for identity loss')

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
        self.encoderA = Encoder(init_f, max_f, latent_size, num_layers, num_channels)
        # encode image B to W_B
        self.encoderB = Encoder(init_f, max_f, latent_size, num_layers, num_channels)
        # generate fakeA2B with W_A
        self.decoderA2B = Generator(init_f, max_f, num_layers, latent_size, num_channels)
        # generate fakeB2A with W_B
        self.decoderB2A = Generator(init_f, max_f, num_layers, latent_size, num_channels)
        # discriminate realA vs fakeB2A
        self.discriminatorA = Discriminator(mapping_layers=3, latent_size=latent_size, mapping_fmaps=latent_size)
        # discriminate realB vs fakeA2B
        self.discriminatorB = Discriminator(mapping_layers=3, latent_size=latent_size, mapping_fmaps=latent_size)

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


        self.num_layers = num_layers
        self.latent_size = latent_size
        self.direction = opt.direction
        self.loss_names = ['G', 'D']

    def set_input(self, input):
        AtoB = self.direction in ['AtoB']
        self.realA = input['A' if AtoB else 'B'].to(self.device)
        self.realB = input['B' if AtoB else 'A'].to(self.device)

    def generate(self, stylesA, stylesB, noise=True, return_style=False):
        sA = stylesA.view(stylesA.size(0), 1, stylesA.size(1))
        stylesA = sA.repeat(1, self.num_layers * 2, 1)

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








