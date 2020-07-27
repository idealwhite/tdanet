from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self,  parser):
        parser = BaseOptions.initialize(self, parser)

        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test')
        parser.add_argument('--nsampling', type=int, default=50, help='ramplimg # times for each images')
        parser.add_argument('--ncaptions', type=int, default=10, help='Number of captions for each image')
        parser.add_argument('--save_number', type=int, default=10, help='choice # reasonable results based on the discriminator score')
        parser.add_argument('--no_variance', action='store_true', help='set variation to 0 when generate')

        self.isTrain = False

        return parser
