from data.base_dataset import BaseDataset
import importlib
from torch.utils.data import DataLoader


def find_dataset_using_name(dataset_name):
    """
    import data/[dataset_name]_dataset.py

    :param dataset_name:
    :return:
    """
    dataset_filename = 'data.' + dataset_name + '_dataset'
    # print(dataset_filename)
    datasetlib = importlib.import_module(dataset_filename)
    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    # print('target {}'.format(target_dataset_name))
    for name, cls in datasetlib.__dict__.items():
        # print('name {}'.format(name.lower()))
        if name.lower() == target_dataset_name.lower() \
                and issubclass(cls, BaseDataset):
            dataset = cls
    if dataset is None:
        raise NotImplementedError("In {}.py , there should be a subclass of BaseDataset with"
                                  "class name that matches {} in lowercase.".format(dataset_filename, target_dataset_name))

    return dataset


def get_option_setter(dataset_name):
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options


def create_dataset(opt):
    data_loader = CustomDataLoader(opt)
    dataset = data_loader.load_data()
    return dataset


class CustomDataLoader:
    def __init__(self, opt):
        self.opt = opt
        dataset_class = find_dataset_using_name(opt.dataset_mode)
        self.dataset = dataset_class(opt)
        print('dataset [{}] was created.'.format(type(self.dataset).__name__))
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.num_threads)
        )

    def load_data(self):
        return self

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data


