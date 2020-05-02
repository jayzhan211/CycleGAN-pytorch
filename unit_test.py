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


if __name__ == '__main__':
    x = torch.randn(3, 1, 2)
    x[:, 0] += torch.randn(3, 2)
    print(x, x.size())
    print(x[:, :1].size())
