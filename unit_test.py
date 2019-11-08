import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from data import create_dataset
from options.train_options import TrainOptions
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from utils.visualizer import Visualizer
import time
import numpy as np
from torchvision import transforms
from PIL import Image

x = torch.randn(1, 3 * 1 * 1)
y = x.clone()
print(x)
print(y)
