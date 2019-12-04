from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import os.path
import random


class ColorizationDataset(BaseDataset):
    """
    Load RGB images, return (L, RGB)
    '--model colorization'
    """
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt):
        super(ColorizationDataset, self).__init__(opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

        self.transform_A_RGB = get_transform(self.opt)
        self.transform_B_RGB = get_transform(self.opt)
        self.transform_A_gray = get_transform(self.opt, gray_scale=True)
        self.transform_B_gray = get_transform(self.opt, gray_scale=True)

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        if self.opt.serial_batches:
            index_B = index % self.B_size
        else:
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        A_RGB = self.transform_A_RGB(A_img)
        B_RGB = self.transform_B_RGB(B_img)
        A_gray = self.transform_A_gray(A_img)
        B_gray = self.transform_B_gray(B_img)

        return {
            'A_RGB': A_RGB,
            'B_RGB': B_RGB,
            'A_Gray': A_gray,
            'B_Gray': B_gray,
            'A_paths': A_path,
            'B_paths': B_path
        }

    def __len__(self):
        return max(self.A_size, self.B_size)
