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

if __name__ == '__main__':
    a = np.random.randn(64, 64, 3)
    print(a)
    cam_img = cv2.resize(a, (256, 256)).astype(np.uint8)
    print(type(cam_img))



