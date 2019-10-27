from PIL import Image
import os
import numpy as np
import torch


def tensor2numpy(img, img_type=np.uint8):

    if isinstance(img, torch.Tensor):
        img_tensor = img.data
        assert img_tensor.max() <= 1.0 and img_tensor.min() >= -1.0, 'torch.tensor is out of range [-1.0, 1.0]'
        img_np = img_tensor[0].cpu().float().numpy()
        if img_np.shape[0] == 1:
            img_np = np.tile(img_np, (3, 1, 1))
        # [-1.0, 1.0] -> [0.0, 255.0]
        img_np = (np.transpose(img_np, (1, 2, 0)) + 1.0) / 2.0 * 255.0
        img = img_np

    elif not isinstance(img, np.ndarray):
        raise TypeError('img must be torch.Tensor or np.ndarray')

    return img.astype(img_type)


def save_image(img, img_pth, aspect_ratio=1.0):
    """

    :param img: (numpy array)
    :param img_pth:
    :param aspect_ratio:
    :return:
    """
    assert isinstance(img, np.ndarray), 'img should be np.ndarray, but' \
                                        'found {} instead'.format(type(img))
    img_pil = Image.fromarray(img)
    h, w, _ = img.shape
    if aspect_ratio > 1.0:
        img_pil = img_pil.resize(h, w * aspect_ratio, Image.BICUBIC)
    if aspect_ratio < 1.0:
        img_pil = img_pil.resize(h / aspect_ratio, w, Image.BICUBIC)
    img_pil.save(img_pth)


def mkdir(pth):
    """

    :param pth: list or str
    :return:
    """
    if isinstance(pth, list):
        for p in pth:
            if not os.path.exists(p):
                os.mkdir(p)
    elif isinstance(pth, str):
        if not os.path.exists(pth):
            print('pth {}'.format(pth))
            os.mkdir(pth)
    else:
        raise NotImplementedError('path must be list|str')


def str2bool(x):
    return x.lower() == 'true'
