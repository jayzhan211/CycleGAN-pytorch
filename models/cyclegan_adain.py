from .base_model import BaseModel


class CycleGANAdaINModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(dataset_mode='colorization')