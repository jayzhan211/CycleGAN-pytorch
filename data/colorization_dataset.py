from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from skimage import color
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import os.path
import random

class ColorizationDataset(BaseDataset):
    """
    Load RGB images, return (L, RGB)
    '--model colorization'
    """
    @staticmethod
    def modify_commandline_options(parser):
        return parser

    def __init__(self, opt):
        super(ColorizationDataset, self).__init__()
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

        self.transform_A_RGB = get_transform(self.opt, convert=False)
        self.transform_B_RGB = get_transform(self.opt, convert=False)
        self.transform_A_gray = get_transform(self.opt, convert=False, gray_scale=True)
        self.transform_B_gray = get_transform(self.opt, convert=False, gray_scale=True)

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        if self.opt.serial_batches:
            index_B = index % self.B_size
        else:
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A_RGB = Image.open(A_path).convert('RGB')
        B_RGB = Image.open(B_path).convert('RGB')
        A_gray = Image.open(A_path).convert('L')
        B_gray = Image.open(B_path).convert('L')

        A_RGB = self.transform_A_RGB(A_RGB)
        B_RGB = self.transform_B_RGB(B_RGB)
        A_gray = self.transform_A_gray(A_gray)
        B_gray = self.transform_B_gray(B_gray)

        return {
            'A_RGB': A_RGB,
            'B_RGB': B_RGB,
            'A_gray': A_gray,
            'B_gray': B_gray,
            'A_paths': A_path,
            'B_paths': B_path
        }

    def __len__(self):
        return max(self.A_size, self.B_size)