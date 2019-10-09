from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def __init__(self):
        self.isTrain = False

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--n_test', type=int, default=float("inf"), help='# of test examples.')
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test', help='train|val|test')
        """
        BatchNorm and Dropout have different behavior between train and test
        """
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=50, help='how many test images to run')
        parser.set_defaults(model='test')
        parser.set_defaults(load_size=parser.get_default('crop_size'))
        self.isTrain = False
        return parser


