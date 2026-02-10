import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import batch_flating
from .encoder import Transfomer_Encoder, LSTM_Encoder
from .AFW import AFW

class Net(nn.Module):
    def __init__(self, args):
        super().__init__()
        tag_size = args.output_dim
        self.args = args
        self.num_modal = len(args.modal)
        self.modal = args.modal
        self.input_dim_a = args.embedding_dim['A']
        self.input_dim_v = args.embedding_dim['V']
        self.input_dim_l = args.embedding_dim['L']
        
        self.embedding_dim = {
            'A': self.input_dim_a,
            'V': self.input_dim_v,
            'L': self.input_dim_l
        }
        
        self.device = args.device
        self.hidden_dim = args.hidden_dim
        self.dropout = args.drop_rate
        
        self.encoder = nn.ModuleDict()

        if not args.no_mmt:
            print("Net -> Using AFW")
            self.mmt = AFW([self.hidden_dim] * self.num_modal, args.tensor_rank, self.modal, args.beta, args.mmt_nlayers, self.dropout)
        else:
            self.mmt = None

        self.fc_out = nn.Linear(self.hidden_dim * self.num_modal, tag_size)

        for m in self.modal:
            self.encoder.add_module(m, self.get_encoder(m))
        
        if self.args.mmcosine:
            print("Net --> Using AMW")
    
    def get_encoder(self, m):
        print(f"Net --> {m} encoder: {self.args.encoder_modules}")
        if self.args.encoder_modules == "transformer":
            return Transfomer_Encoder(
                self.embedding_dim[m], 
                self.hidden_dim,
                self.args.encoder_nlayers,
                dropout=self.dropout )
        
        elif self.args.encoder_modules == "lstm":
            return LSTM_Encoder(
                self.embedding_dim[m], 
                self.hidden_dim,
                self.args.encoder_nlayers,
                dropout=self.dropout )



    def forward(self, data):
        x = {
            m: data[f"{m}_feat"] * data[f'{m}_miss_index'].unsqueeze(2)
            for m in self.modal         
        }
        
        lengths = data["seq_lengths"]

        encoded = {}

        for m in self.modal:
            encoded[m] = self.encoder[m](x[m], lengths)

        amn_loss = torch.tensor(0.0).to(self.device)
        if self.mmt:
            if self.training:
                out, amn_loss = self.mmt(encoded)
            else:
                out = self.mmt(encoded)
        else:
            out = encoded # {A : tensor, V: tensor, L: tensor}
        
        out = batch_flating(out, lengths, self.modal, self.device) # {A : tensor (t, dim), V: tensor, L: tensor}
        _out = out

        if self.args.mmcosine:
            weight_dict = self.get_fusion_weight()
            out = {
                m: torch.matmul(F.normalize(out[m], dim=1), F.normalize(torch.transpose(weight_dict[m]["weight"], 0, 1), dim=1)) + weight_dict[m]["bias"]
                for m in self.modal
            }
            stack = torch.stack(list(out.values()), dim=1)
            out = torch.sum(stack, dim=1)
        else:
            out = torch.cat(list(out.values()), dim=-1)
            out = self.fc_out(out) # (t, output_dim)

        return out, _out, amn_loss
    
    def get_encoder_params(self):
        params_dict = {
            m:[] for m in self.modal
        }

        for m in self.modal:
            for params in self.encoder[m].parameters():
                params_dict[m].append(params)
        return params_dict
    
    def get_fusion_weight(self):
        weight_dict = {
            m: {} for m in self.modal
        }
        
        for j, m in enumerate(self.modal):
            weight_dict[m]["weight"] = self.fc_out.weight[:, j * self.hidden_dim: (j + 1) * self.hidden_dim]
            weight_dict[m]["bias"] = self.fc_out.bias / self.num_modal
    
        return weight_dict