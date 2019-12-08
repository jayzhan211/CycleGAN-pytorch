import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from data import create_dataset
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

if __name__ == '__main__':

    x = torch.randn(1, 3, 4, 4)
    y = torch.randn(1, 3, 4, 4)
    m = nn.L1Loss()
    z = m(x, y)
    print(x)
    print(y)
    print(z)
    print(abs(x - y))
