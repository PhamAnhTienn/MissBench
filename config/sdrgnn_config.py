from config.base_config import BaseConfig


class Config(BaseConfig):
    """Config for GCNet"""
    
    def __init__(self):
        super().__init__()
    
    def setup_model_specific_args(self):
        self.parser.add_argument('--time-attn', action='store_true', default=False, help='whether to use nodal attention in graph model: Equation 4,5,6 in Paper')
        self.parser.add_argument('--windowp', type=int, default=6, help='context window size for constructing edges in graph model for past utterances, -1: fully connect')
        self.parser.add_argument('--windowf', type=int, default=6, help='context window size for constructing edges in graph model for future utterances, -1: fully connect')
        self.parser.add_argument('--hidden', type=int, default=100, help='hidden size in model training')
        self.parser.add_argument('--dropout', type=float, default=0.5, metavar='dropout', help='dropout rate')
        self.parser.add_argument('--loss-recon', action='store_true', default=False, help='whether to use reconstrctuion loss')
        self.parser.add_argument('--reccls-flag', action='store_true', default=False, help='whether to use reconstrctuion features for classification')
        self.parser.add_argument('--lower-bound', action='store_true', default=False, help='whether remove missing modality in the training process')
        self.parser.add_argument('--recon-loss-weight', type=float, default=0.5, help='set reconstruction loss weight')