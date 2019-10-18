import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from data.image_folder import make_dataset
import torchvision.transforms as transforms
from utils.util import mkdir

"""
resize image to (256,256) and save to disk
"""

class CustomerDataset(Dataset):
    def __init__(self, data_root):
        self.data_root = data_root
        self.imgs_path = make_dataset(self.data_root)

    def __getitem__(self, index):
        img_pth = self.imgs_path[index]
        img = Image.open(img_pth).convert('RGB')
        return {
            'img': img,
            'img_path': img_pth,
        }

    def __len__(self):
        return len(self.imgs_path)


dataset = CustomerDataset(data_root='./sketch')
mkdir('./sketch_256')
for i, data in enumerate(dataset):
    img = data['img']
    img_pth = data['img_path']
    if img_pth[-3:] not in ['jpg', 'png']:
        continue
    # print(img.size)
    # print(img_pth)
    img = img.resize((256, 256), Image.BILINEAR)
    # img = img.resize((256, 256), Image.BICUBIC)
    # img = img.resize((256, 256), Image.NEAREST)
    # print(img.size)
    img.save('./sketch_256/{:04d}.{}'.format(i, img_pth[-3:]))
    # data.show()
    # break

