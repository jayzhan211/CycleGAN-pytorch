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
    device = torch.device('cuda:0')
    print(device)
    x = torch.randn(3, 3).to(device)
    z = torch.randn(3, 3).to(device)
    y = torch.eye(3)
    print(x.device)
    print(y.device)
    z = x + y
    print(z)
