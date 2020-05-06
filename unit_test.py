import os
from collections import OrderedDict

from torch import nn

from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from utils import html
from utils.visualizer import save_images
import torch
import numpy as np
import cv2
from PIL import Image
from models.networks import UnetGenerator, ResnetGeneratorUGATIT, NICE3SResnetGenerator, NICEDiscriminator, \
    NICE3SDiscriminator, NICEResnetGenerator, ILN, DiscriminatorUGATIT, NICEV2ResnetGenerator, NICESADiscriminator, \
    EfficientDetGenerator, EfficientDetDiscrminator
from torchvision.models import mobilenet_v2
# from thop import profile
import torch.nn.functional as F
# from efficientnet_pytorch import EfficientNet
from models.bifpn import BIFPN
from models.efficientnet import EfficientNet
from models.vq_layer import VectorQuantizerEMA
import numpy as np


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


from models.networks_alae import Encoder


def upscale2d(x, factor=2):
    s = x.shape
    x = torch.reshape(x, [-1, s[1], s[2], 1, s[3], 1])
    x = x.repeat(1, 1, 1, factor, 1, factor)
    x = torch.reshape(x, [-1, s[1], s[2] * factor, s[3] * factor])
    return x


if __name__ == '__main__':
    x = torch.randn(1, 4, 2, 2)
    z = torch.randn(1, 4, 1, 1)
    p = []
    p.append(x)
    p.insert(0, z)
    print(p)

    # print((x[:, 0, 0, 0] + x[:, 1, 0, 0] + x[:, 2, 0, 0] + x[:, 3, 0, 0]) / 4)