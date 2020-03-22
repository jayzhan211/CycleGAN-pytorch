import os

from torch import nn

from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from utils import html
from utils.visualizer import save_images
import torch

if __name__ == '__main__':
    h, w = 256, 256
    x = torch.randn(1, 256, h//4, w//4)
    print(x.size())
    gap = torch.nn.functional.adaptive_avg_pool2d(x, 1)
    print(gap.size())
    gap_fc = nn.Linear(256, 1, bias=False)
    gap_logit = gap_fc(gap.view(x.shape[0], -1))
    print(gap_logit)



