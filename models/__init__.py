import importlib
from models.base_model import BaseModel


def find_model_using_name(model_name):
    model_filename = 'models.{}_model'.format(model_name)
    model_lib = importlib.import_module(model_filename)
    model = None
    target_model_name = model_name.replace('_', '') + 'model'
    for name, cls in model_lib.__dict__.items():
        if name.lower() == target_model_name.lower() \
                and issubclass(cls, BaseModel):
            model = cls

    if model is None:
        print('{} not found'.format(target_model_name))
        exit(0)
    return model


def get_option_setter(model_name):
    model_class = find_model_using_name(model_name)
    return model_class.modify_commandline_options


def create_model(opt):
    model = find_model_using_name(opt.model)
    instance = model(opt)
    print("model [{}] was created".format(type(instance).__name__))
    return instance
