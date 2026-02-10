from config.base_config import BaseConfig


class Config(BaseConfig):
    """Config for OGM model"""
    
    def __init__(self):
        super().__init__()
    
    def setup_model_specific_args(self):
        self.parser.add_argument('--method', type=str, default='OGM_GE', choices=['OGM_GE', 'OGM'])
        self.parser.add_argument('--alpha', type=float, default=0.1)
        self.parser.add_argument('--fusion_method', type=str, default='concat', choices=['concat', 'sum'])