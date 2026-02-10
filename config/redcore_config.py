from config.base_config import BaseConfig


class Config(BaseConfig):
    """Config for RedCore model"""
    
    def __init__(self):
        super().__init__()
    
    def setup_model_specific_args(self):
        self.parser.add_argument('--embd_size_a', type=int, default=128, help='audio model embedding size')
        self.parser.add_argument('--embd_size_l', type=int, default=128, help='text model embedding size')
        self.parser.add_argument('--embd_size_v', type=int, default=128, help='visual model embedding size')
        self.parser.add_argument('--AE_layers', type=str, default='256,128,64', help='256,128 for 2 layers with 256, 128 nodes respectively')
        self.parser.add_argument('--n_blocks', type=int, default=5, help='number of AE blocks')
        self.parser.add_argument('--cls_layers', type=str, default='128,128', help='256,128 for 2 layers with 256, 128 nodes respectively')
        self.parser.add_argument('--dropout_rate', type=float, default=0.5, help='rate of dropout')
        self.parser.add_argument('--bn', action='store_true', help='if specified, use bn layers in FC')
        self.parser.add_argument('--mse_weight', type=float, default=4.0, help='weight of mse loss')
        self.parser.add_argument('--ii', type=int, default=2, help='iteration interval for beta update')
        self.parser.add_argument('--etaext', type=float, default=1.5, help='growth rate of eta')
        self.parser.add_argument('--beta', type=float, default=0.95, help='starting value of beta')
        self.parser.add_argument('--eta', type=float, default=0.001)
        self.parser.add_argument('--share_weight', action='store_true', help='if specified, share weights in AE blocks')