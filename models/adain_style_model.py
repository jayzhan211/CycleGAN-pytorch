import torch

from models.networks import define_Net
from utils.util import calc_mean_std, coral
from .base_model import BaseModel
from . import networks

class AdaInStyleModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(dataset_mode='unaligned')
        if is_train:
            parser.add_argument('--content_weight', type=float, default=5.0, help='weight for content loss')
            parser.add_argument('--style_weight', type=float, default=1.0, help='weight for style loss')
            parser.add_argument('--preserve_color', action='store_true',
                                help='if specified, preserve color of the content image')
            parser.add_argument('--alpha', type=float, default=1.0,
                                help='the weight that controls the degree of stylization. Should be between 0 and 1')
            parser.add_argument('--no_use_pretrained_vgg', action='store_true',
                                help='the weight that controls the degree of stylization. Should be between 0 and 1')

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ['content', 'style']
        self.visual_names = [
            'real_A', 'real_B',
            # 'style_feat_1', 'style_feat_2', 'style_feat_3', 'style_feat_4',
            # 'content_feat_1', 'content_feat_2', 'content_feat_3', 'content_feat_4',
            # 'g_t_feat_1', 'g_t_feat_2', 'g_t_feat_3', 'g_t_feat_4',
            'g_t',
        ]
        self.model_names = ['net']

        if not opt.no_use_pretrained_vgg:
            self.net = define_Net(net_type='adainstyletransfer', pretrained_model_path='models/vgg_normalised.pth',
                                  gpu_ids=self.gpu_ids)
        else:
            self.net = define_Net(net_type='adainstyletransfer', pretrained_model_path=None,
                                  gpu_ids=self.gpu_ids)

        self.optimizer = torch.optim.Adam(self.net.module.decoder.parameters(), lr=opt.lr)
        self.optimizers.append(self.optimizer)

        self.l1_loss = torch.nn.L1Loss()
        self.mse_loss = torch.nn.MSELoss()

        self.content_weight = opt.content_weight
        self.style_weight = opt.style_weight

        self.preserve_color = opt.preserve_color
        self.alpha = opt.alpha

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        if self.preserve_color:
            self.real_B = coral(self.real_B, self.real_A)

    def optimize_parameters(self):
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()

    def backward(self):
        style_feats, content_feats, g_t_feats, g_t, t = self.net(self.real_A, self.real_B, self.alpha)
        loss_content = self.calc_content_loss(g_t_feats[-1], t)
        loss_style = 0
        for i in range(4):
            loss_style += self.calc_style_loss(g_t_feats[i], style_feats[i])

        loss = loss_content * self.content_weight + loss_style * self.style_weight
        loss.backward()

        self.loss_content, self.loss_style = loss_content, loss_style
        self.g_t = g_t

    def calc_rec_loss(self, input, target):
        assert (input.size() == target.size())

        return self.l1_loss(input, target)

    def calc_content_loss(self, input, target):
        assert (input.size() == target.size())

        return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())

        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)


