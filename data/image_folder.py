import torch.utils.data as data
import os.path

from torchvision.datasets.folder import is_image_file


def make_dataset(_dir, max_data_size=float('inf')):
    images = []
    assert os.path.isdir(_dir), '{} is not a valid' \
                                'directory'.format(_dir)

    for dirpath, _, file_names in os.walk(_dir):
        for file_name in file_names:
            if is_image_file(file_name):
                pth = os.path.join(dirpath, file_name)
                images.append(pth)
    return images[:min(max_data_size, len(images))]

