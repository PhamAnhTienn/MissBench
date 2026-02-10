from config.base_config import BaseConfig


class Config(BaseConfig):
    """Config for MCE model"""
    
    def __init__(self):
        super().__init__()
    
    def setup_model_specific_args(self):
        self.parser.add_argument('--input_dim_a', type=int, default=130, help='acoustic input dim')
        self.parser.add_argument('--input_dim_l', type=int, default=1024, help='lexical input dim')
        self.parser.add_argument('--input_dim_v', type=int, default=384, help='lexical input dim')
        self.parser.add_argument('--embd_size_a', default=128, type=int, help='audio model embedding size')
        self.parser.add_argument('--embd_size_l', default=128, type=int, help='text model embedding size')
        self.parser.add_argument('--embd_size_v', default=128, type=int, help='visual model embedding size')
        self.parser.add_argument('--embd_method_a', default='maxpool', type=str, choices=['last', 'maxpool', 'attention'], \
            help='audio embedding method,last,mean or atten')
        self.parser.add_argument('--embd_method_v', default='maxpool', type=str, choices=['last', 'maxpool', 'attention'], \
            help='visual embedding method,last,mean or atten')
        self.parser.add_argument('--AE_layers', type=str, default='128,64,32', help='256,128 for 2 layers with 256, 128 nodes respectively')
        self.parser.add_argument('--n_blocks', type=int, default=3, help='number of AE blocks')
        self.parser.add_argument('--cls_layers', type=str, default='128,128', help='256,128 for 2 layers with 256, 128 nodes respectively')
        self.parser.add_argument('--dropout_rate', type=float, default=0.3, help='rate of dropout')
        self.parser.add_argument('--bn', action='store_true', help='if specified, use bn layers in FC')
        self.parser.add_argument('--pretrained_path', type=str, help='where to load pretrained encoder network')
        self.parser.add_argument('--ce_weight', type=float, default=1.0, help='weight of ce loss')
        self.parser.add_argument('--mse_weight', type=float, default=1.0, help='weight of mse loss')
        self.parser.add_argument('--cycle_weight', type=float, default=1.0, help='weight of cycle loss')
        self.parser.add_argument('--rec_weight', type=float, default=1.0, help='weight of reconstruction loss')
        self.parser.add_argument('--sep_weight', type=float, default=1.0, help='weight of single modality loss')
        self.parser.add_argument('--sub_weight', type=float, default=1.0, help='weight of sub fuse loss')
        self.parser.add_argument('--share_weight', action='store_true', help='share weight of forward and backward autoencoders')
        self.parser.add_argument('--shap_weight', type=float, default=1.0, help='weight of shapley loss')
        self.parser.add_argument('--use_a', action='store_true', help='use factor a')
        self.parser.add_argument('--use_b', action='store_true', help='use factor b')