from config.base_config import BaseConfig


class Config(BaseConfig):
    """Config for Simple Model"""
    
    def __init__(self):
        super().__init__()
    
    def setup_model_specific_args(self):
        self.parser.add_argument('--imputation', type=str, default='zero', choices=['zero', 'mean', 'random'], help='imputation method for missing data')
