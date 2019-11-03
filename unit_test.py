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
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms

from PIL import Image

if __name__ == '__main__':
    opt = TrainOptions().parse()
    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    model = create_model(opt)
    model.setup(opt)
    visualizer = Visualizer(opt)
    total_iters = 0
    t_data = 0

    for i, data in enumerate(dataset):
        im_A = data['A'][0]
        im_A = torch.clamp(im_A.permute(1, 2, 0), 0, 1)
        im_A_L = data['A_L'][0][0]
        print(im_A_L.size())
        # im_A_L = torch.clamp(im_A_L.permute(1, 2, 0), 0, 1)
        # print(type(im_A))
        fig = plt.figure(figsize=(8, 8))
        rows = 1
        columns = 2
        fig.add_subplot(1, 2, 1)
        plt.imshow(im_A)
        fig.add_subplot(1, 2, 2)
        plt.imshow(im_A_L, cmap='gray')
        plt.show()
        # im = data
        # im.show()
        # breakpoint()
        break