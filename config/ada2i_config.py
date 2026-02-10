from config.base_config import BaseConfig


class Config(BaseConfig):
    """Config for Ada2i model"""
    
    def __init__(self):
        super().__init__()
    
    def setup_model_specific_args(self):
        self.parser.add_argument("--alpha", type=float, default=0.5)
        self.parser.add_argument("--normalize", action="store_true", default=False)
        self.parser.add_argument("--mmcosine", action="store_true", default=False)
        self.parser.add_argument("--encoder_modules", type=str, default="transformer", choices=["transformer", "lstm"])
        self.parser.add_argument("--encoder_nlayers", type=int, default=2)
        self.parser.add_argument("--mmt_nlayers", type=int, default=2)
        self.parser.add_argument("--tensor_rank", type=int, default=8)
        self.parser.add_argument("--beta", type=float, default=0.7)
        self.parser.add_argument("--nheads", type=int, default=2)
        self.parser.add_argument("--hidden_dim", type=int, default=200)
        self.parser.add_argument("--fusion", type=str, default="concat", choices=["sum", "concat", "film", "gated"])
        self.parser.add_argument("--drop_rate", type=float, default=0.5)
        self.parser.add_argument("--no_mmt", action="store_true", default=False)