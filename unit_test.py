import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from data import create_dataset
from options.train_options import TrainOptions
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
# from utils.visualizer import Visualizer
import time
import numpy as np
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib

if __name__ == '__main__':
    im = Image.open('dataset/sketch2Line/trainB/0054.png')
    # im.show()
    m = transforms.ToTensor()

    im = m(im)
    im = im.unsqueeze(0)
    print(im.size())
    assert im.size() == (1, 3, 256, 256)
    im = 0.2989 * im[:, 0, ...] + 0.5870 * im[:, 1, ...] + 0.1140 * im[:, 2, ...]
    print(im)

    m = transforms.ToPILImage()

    im = m(im)
    im.show()


    # im = m(im)
    # # print(im.size())
    #
    # m = transforms.Compose([
    #     transforms.ToPILImage(),
    #     transforms.Grayscale(),
    #     transforms.ToTensor()
    # ])
    #
    # im = m(im)
    # # im.show()
    # print(im.size())

    # x = torch.randn(1, 3, 256, 256)
    #
    # m = transforms.Compose([
    #     transforms.ToPILImage(),
    # ])
    #
    # im = m(x[0]).convert('RGB')
    # im.show()