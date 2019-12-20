import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import ImageFile, Image

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 636480000

"""
resize images in dataset random_split to train and test dataset 

add new folder datsetA or datasetB and push only one dataset at once 
"""

if __name__ == '__main__':
    datasetA = 'drawing'
    datasetB = 'painting'
    ratio = 0.9

    # =========================================================================

    for dataset_name, datafolder in [(datasetA, 'datasetA'), (datasetB, 'datasetB')]:

        dataset = datasets.ImageFolder(root='./{}'.format(datafolder), transform=transforms.Compose([
            # transforms.Resize(256),
            transforms.ToTensor()
        ]))
        train_sz = int(ratio * len(dataset))
        test_sz = len(dataset) - train_sz
        train_ds, test_ds = torch.utils.data.random_split(dataset, [train_sz, test_sz])

        save_path_directory_train = 'dataset/new_{}/train'.format(dataset_name)
        save_path_directory_test = 'dataset/new_{}/test'.format(dataset_name)
        if not os.path.exists(save_path_directory_train):
            os.makedirs(save_path_directory_train)
        if not os.path.exists(save_path_directory_test):
            os.makedirs(save_path_directory_test)

        train_loaders = DataLoader(train_ds)
        test_loaders = DataLoader(test_ds)

        for i, (data, label) in enumerate(train_loaders):
            img = transforms.ToPILImage()(data[0]).convert("RGB")
            img.save('{}/{:05d}.jpg'.format(save_path_directory_train, i))
        for i, (data, label) in enumerate(test_loaders):
            img = transforms.ToPILImage()(data[0]).convert("RGB")
            img.save('{}/{:05d}.jpg'.format(save_path_directory_test, i))

