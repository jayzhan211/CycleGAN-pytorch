import functools
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from data import create_dataset
from models.networks import VGG19

from options.train_options import TrainOptions
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model, networks
# from utils.visualizer import Visualizer
import time
import numpy as np
from torchvision import transforms
from PIL import Image

from utils.util import toGray, calc_mean_std, coral


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.gen = networks.StyleTransfer('models/vgg_normalised.pth')

    def func(self):
        net = getattr(self, 'gen')
        print(net.module)

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.gen = networks.ResnetGenerator(3, 64, 64)

    def func(self):
        net = getattr(self, 'gen')
        print(net.module.cpu().state_dict())

def defineG():
    net = networks.StyleTransfer(model_path='models/vgg_normalised.pth')
    return net

def calc_content_loss(input, target):
    assert (input.size() == target.size())
    m = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    for b in range(input.size()[0]):

        input[b] = m(input[b])
        target[b] = m(target[b])


if __name__ == '__main__':
    x = torch.randn(1, 3, 256, 256)
    m = VGG19()
    z = m(x)
    print()


