from PIL import Image
import os
import numpy as np
import torch


def tensor2img(img, img_type=np.uint8):
    if isinstance(img, np.ndarray):
        return img.astype(img_type)
    if isinstance(img, torch.Tensor):
        img_np = img.data[0].cpu().float().numpy()
        if img_np.shape[0] == 1:
            img_np = np.tile(img_np, (3, 1, 1))
        img_np = np.transpose(img_np, (1, 2, 0))
        return img_np
    else:
        raise TypeError('img must be torch.Tensor or np.ndarray')


def save_image(img, img_pth, aspect_ratio=1.0):
    """

    :param img: (numpy array)
    :param img_pth:
    :param aspect_ratio:
    :return:
    """
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


