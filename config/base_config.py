from abc import ABC, abstractmethod
import argparse
import datetime
import torch


class BaseConfig(ABC):
    """Base config class for all models"""
    
    def __init__(self):
        self.setup_common_args()
        self.setup_model_specific_args()
        self.setup_experimental_args()
        
    def setup_common_args(self):
        """Common arguments for all models"""
        parser = argparse.ArgumentParser()
     
        # Logging Setting
        parser.add_argument("--comet", action="store_true", default=False)
        parser.add_argument("--comet_api", type=str, default="")
        parser.add_argument("--comet_workspace", type=str, default="")
        parser.add_argument("--project_name", type=str, default="")
        parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--log_path', type=str, default='./logs', help='logs are saved here')
        
        # base parameters
        parser.add_argument('--base_encoder', type=str, default='lstm', help='chooses which encoder to use. [lstm | gru | transformer]')
        parser.add_argument('--bi_enc', action='store_true', help='whether to use bidirectional LSTM/GRU')
        parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization [instance | batch | none]')
        parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay when training')
        parser.add_argument('--init_method', type=str, default='kaiming', help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.012, help='scaling factor for normal, xavier and orthogonal.')
        
        # dataset parameters
        parser.add_argument("--dataset", type=str, choices=["iemocap", "mosi", "mosei", "sims"], default="iemocap")
        parser.add_argument("--data_dir", type=str, default="./dataset")
        parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
        parser.add_argument('--dataset_mode', type=str, default='con', choices=['con', 'noncon'], help='load dataset as conversational(con) or non-conversational(noncon)')
        parser.add_argument("--ratio", type=float, default=0.1, help="train/val set ratio (for non-split datasets like iemocap)")
        parser.add_argument('--num_threads', type=int, default=0, help='number of threads for loading data')
        parser.add_argument('--batch_first', action='store_true', help='whether to use batch first in dataloader')
        
        # Training Setting
        parser.add_argument("--seed", default=12)
        parser.add_argument("--epochs", type=int, default=50)
        parser.add_argument('--early_stop', type=int, default=-1, help='early stopping epochs, set to -1 to disable')
        parser.add_argument("--start_mod", type=int, default=0, help="the starting epoch for modulation")
        parser.add_argument("--end_mod", type=int, default=3000, help="the ending epoch for modulation")
        parser.add_argument('--modal', default='AVL', choices=['AVL', 'AV', 'AL', 'VL', 'A', 'V', 'L'], help='modalities used in the model')
        parser.add_argument('--modal_MR', type=float, nargs='+', help='Missing rate modality A/V/L, use 1 value for SMR or 3 values for IMR, e.g [0.5] or [0.5, 0.3, 0.2]', default=[0.0, 0.0, 0.0])

        parser.add_argument('--task_weight', type=float, default=1.0, help='weight for main task (cls/reg) loss')
        parser.add_argument("--optimizer", type=str, default="adam", choices=["sgd", "adam", "adamw", "rmsprop"])
        parser.add_argument("--lr", type=float, default=2e-4, help="initial learning rate for adam")
        parser.add_argument("--l2", type=float, default=0.0001)
        parser.add_argument("--max_grad_norm", type=float, default=1.0)
        parser.add_argument('--lr_policy', type=str, default='plateau', help='learning rate policy')
        parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
        parser.add_argument("--modulation", action="store_true", default=False)
        parser.add_argument('--regression', action='store_true', default=False, help='is regression task if True, else classification task')
        parser.add_argument('--select_metric', type=str, default='f1', choices=['wa', 'ua', 'f1', 'loss'], help='the metric for selecting best model after validation [wa | ua | f1 | loss]')
        
        self.parser = parser
        
    @abstractmethod
    def setup_model_specific_args(self):
        """Setup model-specific arguments"""
        pass
    
    def setup_experimental_args(self):
        """Setup experimental arguments"""
        self.parser.add_argument('--MEI', action='store_true', help='enable MEI evaluation')
        self.parser.add_argument('--MEI_epoch', type=int, default=1, help='run MEI every n epochs')
        self.parser.add_argument('--MLI', action='store_true', help='enable MLI evaluation')
        pass
    
    def get_args(self):
        """Parse arguments and return as a namespace"""
        args, unknown = self.parser.parse_known_args()
        print("Unknown args:", unknown[1:])
            
        # Setup label dicts
        args.dataset_label_dict = {
            "iemocap": {"hap": 0, "sad": 1, "neu": 2, "ang": 3, "exc": 4, "fru": 5},
            "mosi":{"neg": 0, "neu": 1, "pos": 2}
        }
        
        args.dataset_speaker_dict = {
            "iemocap": 2, "mosi": 1, "mosei": 1, "sims": 1
        }
        
        if args.regression:
            args.output_dim = 1
        else:
            args.output_dim = 4
        
        # Process seed
        if args.seed == "time":
            args.seed = int(datetime.now().timestamp())
        else:
            args.seed = int(args.seed)
            
        if args.device=='cuda' and not torch.cuda.is_available():
            print("CUDA is not available, switch device to CPU")
            args.device = "cpu"
            
        return args