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
from models.networks import UnetGenerator, ResnetGeneratorUGATIT

if __name__ == '__main__':
    # a = np.random.randn(64, 64, 3)
    # print(a)
    # cam_img = cv2.resize(a, (256, 256)).astype(np.uint8)
    # print(type(cam_img))

    # x = torch.randn(1, 5, 3, 3)
    # print(x.size(0))
    # print(x.shape[0])
    # y = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        # print(y.size())
    a = [5, a[-1]*2, a[-1]*3]
    print(a)
    # m = nn.Sequential(
    #     nn.Conv2d(3, 5, kernel_size=3),
    #     nn.ReLU(True),
    # )
    # print(m)

    # def count_parameters(model):
    #     return sum(p.numel() for p in model.parameters() if p.requires_grad)
    #
    # print(count_parameters(m))
    #
    # print(m)

    # a = [1, 3]
    # b = [4, 6]
    # c = a + [9, 2, 3] + b
    # print(c)
