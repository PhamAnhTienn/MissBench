from config.base_config import BaseConfig


class Config(BaseConfig):
    """Config for UttFusion model"""
    
    def __init__(self):
        super().__init__()
    
    def setup_model_specific_args(self):
        self.parser.add_argument('--embd_size_a', type=int, default=128, help='audio model embedding size')
        self.parser.add_argument('--embd_size_l', type=int, default=128, help='text model embedding size')
        self.parser.add_argument('--embd_size_v', type=int, default=128, help='visual model embedding size')
        self.parser.add_argument('--embd_method_a', type=str, default='maxpool', choices=['last', 'maxpool', 'attention'], help='audio embedding method,last,mean or atten')
        self.parser.add_argument('--embd_method_v', type=str, default='maxpool', choices=['last', 'maxpool', 'attention'], help='visual embedding method,last,mean or atten')
        self.parser.add_argument('--cls_layers', type=str, default='128,128', help='256,128 for 2 layers with 256, 128 nodes respectively')
        self.parser.add_argument('--dropout_rate', type=float, default=0.3, help='rate of dropout')
        self.parser.add_argument('--bn', action='store_true', help='if specified, use bn layers in FC')