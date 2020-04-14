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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':

    model = NICEV2ResnetGenerator(3, 3, 64)
    print(count_parameters(model))

    model = NICESADiscriminator(3, 64)                    # 46M
    print(count_parameters(model))


    # model = NICE3SResnetGenerator(3, 3, 64)             # 116M
    # print(count_parameters(model))
    #
    # model = NICE3SDiscriminator(3, 64)                  # 93M
    # print(count_parameters(model))
    #
    # model = NICEResnetGenerator(3, 3, 64, light=True)   # 8M
    # print(count_parameters(model))
    #
    # model = NICEDiscriminator(3, 64)                    # 46M
    # print(count_parameters(model))

    # model = ResnetGeneratorUGATIT(3, 3, light=False)    # 283M
    # print(count_parameters(model))
    #
    # model = ResnetGeneratorUGATIT(3, 3, light=True)     # 15M
    # print(count_parameters(model))
    #
    # model = DiscriminatorUGATIT(3, 64, n_layers=7)     # 53M
    # print(count_parameters(model))

    # model = mobilenet_v2(pretrained=False)
    # print(model)

    # print(img)

    # img_size = 256
    # ngf = 64
    # mult = 4
    #
    # UpBlock2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
    #              nn.ReflectionPad2d(1),
    #              nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=0, bias=False),
    #              ILN(int(ngf * mult / 2)),
    #              nn.ReLU(True))
    #
    # print(count_parameters(UpBlock2))
    #
    # style = [
    #     nn.Conv2d(ngf * mult, ngf * mult, kernel_size=img_size // mult, stride=1, padding=0, groups=img_size // mult,
    #               bias=False),
    #     nn.ReLU(True),]
    # fc = [
    #     nn.Linear(ngf * mult, ngf * mult, bias=False),
    #     nn.ReLU(True),
    #     nn.Linear(ngf * mult, ngf * mult, bias=False),
    #     nn.ReLU(True),
    # ]
    # style = nn.Sequential(*style)
    # fc = nn.Sequential(*fc)
    # # print(count_parameters(style))
    #
    # x = torch.randn(1, 256, 64, 64)
    # x_ = style(x).view(1, -1)
    # z = fc(x_)
    # print(z.size())


    #
    # m1 = nn.Sequential(
    #     nn.Conv2d(256, 256, kernel_size=64, stride=1, padding=0, bias=False),
    #     nn.ReLU(True),
    #     nn.Linear(256, 256, bias=False)
    # )
    # print(count_parameters(m1))
    #
    # m2 = nn.Sequential(
    #     nn.Linear(64 * 64 * 256, 256),
    #     nn.ReLU(True),
    #     nn.Linear(256, 256)
    # )
    # print(count_parameters(m2))