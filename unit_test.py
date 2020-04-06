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
    # params = torch.load(os.path.join('./checkpoints/draw2paintV3-256x_unet256_idt10/epoch_000.pth'))
    params = torch.load(os.path.join('./checkpoints/draw2paintV3-256x_unet256/epoch_200.pth'))
    print(params.keys())