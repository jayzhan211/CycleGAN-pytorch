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
    x  = {
        'A':3,
        'B':4
    }

    y = x['A']
    print(y)
    z = x.get('AA', 'BB')
    print(z)
