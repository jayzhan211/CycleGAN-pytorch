from torchvision import transforms
from PIL import Image
from torchvision.datasets.folder import is_image_file
from torchvision.utils import save_image
import os


def make_resize_dataset(_dir, _transform, new_dir_pth):
    assert os.path.isdir(_dir), '{} is not a valid' \
                                'directory'.format(_dir)
    if not os.path.exists(new_dir_pth):
        os.makedirs(new_dir_pth)
    for dir_path, _, file_names in os.walk(_dir):
        for file_name in file_names:
            if is_image_file(file_name):
                pth = os.path.join(dir_path, file_name)
                img = Image.open(pth).convert('RGB')
                trans = _transform(img)
                save_image(trans, os.path.join(new_dir_pth, file_name))


if __name__ == '__main__':
    Image.MAX_IMAGE_PIXELS = 987654321
    transform_list = [transforms.Resize((256, 256)), transforms.ToTensor()]
    transform = transforms.Compose(transform_list)
    dataroot = "dataset/art2photo"
    # dirA = os.path.join(dataroot, 'trainA')
    # dirPth = "dataset/art2photo-256x/trainA"
    # make_resize_dataset(dirA, transform, dirPth)

    # dirB = os.path.join(dataroot, 'trainB')
    # dirPth = r"dataset/art2photo-256x/trainB"
    # make_resize_dataset(dirB, transform, dirPth)

    # dirA = os.path.join(dataroot, 'testA')
    # dirPth = r"dataset/art2photo-256x/testA"
    # make_resize_dataset(dirA, transform, dirPth)

    dirB = os.path.join(dataroot, 'testB')
    dirPth = r"dataset/art2photo-256x/testB"
    make_resize_dataset(dirB, transform, dirPth)

    print("END")
