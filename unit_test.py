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
from models.networks import UnetGenerator, ResnetGeneratorUGATIT, NICE3SResnetGenerator, NICEDiscriminator, NICE3SDiscriminator, NICEResnetGenerator, ILN, DiscriminatorUGATIT


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
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
    #
    # model = ResnetGeneratorUGATIT(3, 3, light=False)    # 283M
    # print(count_parameters(model))
    #
    # model = ResnetGeneratorUGATIT(3, 3, light=True)     # 15M
    # print(count_parameters(model))
    #
    # model = DiscriminatorUGATIT(3, 64, n_layers=7)     # 53M
    # print(count_parameters(model))
    img = Image.open('112.jfif').convert('RGB')
    print(img)
    img.save('112.png')
    # print(img)