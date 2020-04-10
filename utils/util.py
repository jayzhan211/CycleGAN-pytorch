from PIL import Image
import os
import numpy as np
import torch
import cv2


def cam(x, image_size=256):
    cam_img = np.interp(x, (x.min(), x.max()), (0, 255)).astype(np.uint8)
    cam_img = cv2.resize(cam_img, (image_size, image_size))
    cam_img = cv2.applyColorMap(cam_img, cv2.COLORMAP_JET)
    return cam_img


def denorm(x):
    return x * 0.5 + 0.5


def tensor2numpy(x):
    return x.detach().cpu().numpy().transpose(1, 2, 0)


def RGB2BGR(x):
    return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)


def tensor2im(input_image, imtype=np.uint8, use_cam=False, image_size=256):
    """
    Converts a Tensor array into a numpy image array.
    :param input_image: (tensor) the input image tensor array
    :param imtype: (type) the desired type of the converted numpy array
    :return:
    """

    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))

        if use_cam:
            image_numpy = cam(np.transpose(image_numpy, (1, 2, 0)), image_size)
        else:
            image_numpy = (np.transpose(image_numpy,
                                        (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk
    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def mkdirs(paths):
    """create empty directories if they don't exist
    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist
    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def str2bool(x):
    return x.lower() == 'true'


def toGray(im):
    b, c, h, w = im.size()
    assert c == 3
    im = 0.2989 * im[:, 0, :, :] + 0.5870 * im[:, 1, :, :] + 0.1140 * im[:, 2, :, :]
    return im.unsqueeze(1)


def to3channel(x):
    b, c, h, w = x.size()
    assert c == 1
    x = x.repeat(1, 3, 1, 1)
    return x


def calc_mean_std(feat, eps=1e-5):
    size = feat.size()
    assert (len(size) == 4)
    mean, var = torch.mean(feat, dim=[2, 3], keepdim=True), torch.var(feat, dim=[2, 3], keepdim=True)
    std = torch.sqrt(var + eps)
    return mean, std


def calc_feat_flatten_mean_std(feat):
    assert (len(feat.size()) == 3)
    assert (feat.size()[0] == 3)
    assert feat.dtype == torch.float
    feat_flatten = feat.view(3, -1)
    mean = feat_flatten.mean(dim=-1, keepdim=True)
    std = feat_flatten.std(dim=-1, keepdim=True)
    return feat_flatten, mean, std


def _mat_sqrt(x):
    U, D, V = torch.svd(x)
    return torch.mm(torch.mm(U, D.pow(0.5).diag()), V.t())


def coral(source, target):
    """
    color-preserved in adain
    :param self:
    :return:
    """
    device = source.device
    assert len(source.size()) == len(target.size()) == 3
    assert source.size()[0] == target.size()[0] == 3

    source_f, source_f_mean, source_f_std = calc_feat_flatten_mean_std(source)
    source_f_norm = (source_f - source_f_mean) / source_f_std
    source_f_cov_eye = torch.mm(source_f_norm, source_f_norm.t()) + torch.eye(3).to(device)

    target_f, target_f_mean, target_f_std = calc_feat_flatten_mean_std(target)
    target_f_norm = (target_f - target_f_mean) / target_f_std
    target_f_cov_eye = torch.mm(target_f_norm, target_f_norm.t()) + torch.eye(3).to(device)

    source_f_norm_transfer = torch.mm(
        _mat_sqrt(target_f_cov_eye),
        torch.mm(torch.inverse(_mat_sqrt(source_f_cov_eye)),
                 source_f_norm)
    )

    source_f_transfer = source_f_norm_transfer * target_f_std + target_f_mean
    result = source_f_transfer.view(source.size())
    return result
