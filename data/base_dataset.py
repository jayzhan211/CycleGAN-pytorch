from abc import ABC, abstractmethod
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms


class BaseDataset(data.Dataset, ABC):
    def __init__(self, opt):
        """
        abstract base class for datasets

        To create a subclass, 4 functions below must be implement
        <__init__>
        <__len__>
        <__getitem__>

        :param opt:
        """
        self.opt = opt
        self.root = opt.data_root

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, index):
        pass


def __print_size_warning(ow, oh, w, h):
    """

    :param ow:
    :param oh:
    :param w:
    :param h:
    :return:
    """
    if not hasattr(__print_size_warning, 'has_printed'):

        print('The image size will be adjust to multiple of 4')
        __print_size_warning.has_printed = True


def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img

    __print_size_warning(ow, oh, w, h)
    return img.resize((w, h), method)


def get_transform(opt, params=None, gray_scale=False, method=Image.BICUBIC, convert=True):
    transforms_list = []
    if gray_scale:
        transforms_list.append(transforms.Grayscale(1))
    if 'resize' in opt.preprocess:
        size = [opt.load_size, opt.load_size]
        transforms_list.append(transforms.Resize(size, method))
    elif 'scale_width' in opt.preprocess:
        # (h,w) -> (opt.load_size * h/w, opt.load_size)
        transforms_list.append(transforms.Resize(opt.load_size, method))

    if 'crop' in opt.preprocess:
        if params is None:
            transforms_list.append(transforms.RandomCrop(opt.crop_size))
        else:
            transforms_list.append(transforms.Lambda(
                lambda img: __crop(img, params['crop_pos'], opt.crop_size)
            ))

    if opt.preprocess == 'none':
        transforms_list.append(transforms.Lambda(
            lambda img: __make_power_2(img, base=4, method=method)
        ))

    if not opt.no_flip:
        if params is None:
            transforms_list.append(transforms.RandomHorizontalFlip())
        elif params['flip']:
            transforms_list.append(transforms.RandomHorizontalFlip(p=1.0))

    if convert:
        transforms_list += [transforms.ToTensor()]
        if gray_scale:
            transforms_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transforms_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    return transforms.Compose(transforms_list)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if ow > tw or oh > th:
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img
