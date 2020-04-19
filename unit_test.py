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


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def testImage():
    genA2B = ResnetGeneratorUGATIT(input_nc=3, output_nc=3, ngf=64,
                                   n_blocks=4, img_size=256, light=True)
    fake_A2B = genA2B()


if __name__ == '__main__':
    input = torch.randn(10, 3, 2, 2)
    empty = input.new_empty(0)
    print(empty)

