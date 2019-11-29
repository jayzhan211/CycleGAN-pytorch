import random
from PIL import Image
from data.image_folder import make_dataset
from .base_dataset import BaseDataset, get_transform
import os


class UnalignedDataset(BaseDataset):
    def __init__(self, opt):
        super(UnalignedDataset, self).__init__(opt)
        # BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.data_root, opt.phase + 'A')
        self.dir_B = os.path.join(opt.data_root, opt.phase + 'B')

        self.A_paths = make_dataset(self.dir_A, opt.max_dataset_size)
        self.B_paths = make_dataset(self.dir_B, opt.max_dataset_size)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        b2a = self.opt.direction in ['BtoA', 'B2A']
        input_nc = self.opt.output_nc if b2a else self.opt.input_nc
        output_nc = self.opt.input_nc if b2a else self.opt.output_nc
        self.transform_A = get_transform(self.opt, gray_scale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, gray_scale=(output_nc == 1))

    def __getitem__(self, index):
        """
        Return a data point and its metadata information.
        :param index (int)
        :return:
            A (tensor) input image
            B (tensor) output image
            A_path, B_path (str)
        """
        A_path = self.A_paths[index % self.A_size]
        if self.opt.serial_batches:
            index_B = index % self.B_size
        else:
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        A = self.transform_A(A_img)
        B = self.transform_B(B_img)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)
