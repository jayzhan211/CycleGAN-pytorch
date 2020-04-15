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
from models.networks import UnetGenerator, ResnetGeneratorUGATIT, NICE3SResnetGenerator, NICEDiscriminator, NICE3SDiscriminator, NICEResnetGenerator, ILN, DiscriminatorUGATIT, NICEV2ResnetGenerator, NICESADiscriminator
from torchvision.models import mobilenet_v2
from thop import profile
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from models.bifpn import BIFPN
from models.efficientnet import EfficientNet
# from .module

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # model = EfficientDet(num_classes=3,
    #                      network='efficientdet-d4',
    #                      W_bifpn=224,
    #                      D_bifpn=6,
    #                      D_class=4
    #                      )
    model = EfficientNet.from_pretrained("efficientnet-b2")
    # print(model)
    # print(x1.size())
    x = torch.randn(1, 3, 256, 256)
    out = model(x)[-5:]
    neck = BIFPN(in_channels=model.get_list_features()[-5:],
                          out_channels=112,
                          stack=4,
                          num_outs=5)
    print(neck)
    out1 = neck(out)
    print(len(out1))
    for idx, feat in enumerate(out):
        print(feat.size())
    for idx, feat in enumerate(out1):
        print(feat.size())
    # print(out)
    # print(x)
    # x = F.pad(x, [1, 1])
    # print(x.size())
    # print(x)
