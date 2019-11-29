import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from data import create_dataset
from options.train_options import TrainOptions
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
# from utils.visualizer import Visualizer
import time
import numpy as np
from torchvision import transforms
from PIL import Image

if __name__ == '__main__':
    a = []
    a += [transforms.ToTensor()]
    print(a)

    # opt = TrainOptions().parse()
    # print(opt)
    # dataset = create_dataset(opt)
    # dataset_size = len(dataset)
    # model = create_model(opt)
    # model.setup(opt)
    # visualizer = Visualizer(opt)
    # total_iters = 0
    # t_data = 0
    # for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    #     print('Epoch_{} is starting'.format(epoch))
    #     # time for entire epoch
    #     epoch_start_time = time.time()
    #     # time of data loading
    #     iter_data_time = time.time()
    #     # the number of training iteration in cur epoch
    #     epoch_iter = 0
    #
    #     for i, data in enumerate(dataset):
    #         print(data)
    #         break