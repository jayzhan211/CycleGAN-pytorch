from torchvision import transforms
from PIL import Image
from torchvision.datasets.folder import is_image_file
from torchvision.utils import save_image
import os
import argparse


def make_resize_dataset(_dir, _transform, new_dir_pth):
    assert os.path.isdir(_dir), '{} is not a valid' \
                                'directory'.format(_dir)
    if not os.path.exists(new_dir_pth):
        os.makedirs(new_dir_pth)
    num = 0
    for dir_path, _, file_names in os.walk(_dir):
        for file_name in file_names:
            if is_image_file(file_name):
                pth = os.path.join(dir_path, file_name)
                img = Image.open(pth).convert('RGB')
                trans = _transform(img)
                save_image(trans, os.path.join(new_dir_pth, '{:05d}.{}'.format(num, file_name.split('.')[-1])))
                num += 1


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='name of dataset')
    args = parser.parse_args()

    datasetname = args.dataset
    Image.MAX_IMAGE_PIXELS = 987654321
    transform_list = [transforms.Resize((256, 256)), transforms.ToTensor()]
    transform = transforms.Compose(transform_list)
    dataroot = "dataset/{}".format(datasetname)

    for dir in ['trainA', 'trainB', 'testA', 'testB']:
        dirPth = os.path.join(dataroot, dir)
        if os.path.exists(dirPth):
            newDirPth = "dataset/{}-256x/{}".format(datasetname, dir)
            make_resize_dataset(dirPth, transform, newDirPth)

    print("END")

