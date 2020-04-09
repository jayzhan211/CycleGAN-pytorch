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
    x = torch.randn(1, 4, 2, 2)
    z = x
    z = torch.randn(1, 4)
    print(x.size(), z.size())