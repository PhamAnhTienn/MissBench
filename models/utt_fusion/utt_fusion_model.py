import torch
import torch.nn as nn
from models.mmin.networks.classifier import FcClassifier
from models.mmin.networks.lstm import LSTMEncoder
from models.mmin.networks.textcnn import TextCNN


class UttFusionModel(nn.Module):
    def __init__(self, args, base_encoder):
        super().__init__()
        self.modality = args.modal
        self.device = args.device
        self.input_dim_a = args.embedding_dim['A']
        self.input_dim_v = args.embedding_dim['V']
        self.input_dim_l = args.embedding_dim['L']
        cls_layers = list(map(lambda x: int(x), args.cls_layers.split(',')))
        cls_input_size = args.embd_size_a * int("A" in self.modality) + \
                         args.embd_size_v * int("V" in self.modality) + \
                         args.embd_size_l * int("L" in self.modality)
        self.netC = FcClassifier(cls_input_size, cls_layers, output_dim=args.output_dim, dropout=args.dropout_rate, use_bn=args.bn)

        # acoustic model
        if 'A' in self.modality:
            self.netA = LSTMEncoder(self.input_dim_a, args.embd_size_a, embd_method=args.embd_method_a)
        # lexical model
        if 'L' in self.modality:
            self.netL = TextCNN(self.input_dim_l, args.embd_size_l)
        # visual model
        if 'V' in self.modality:
            self.netV = LSTMEncoder(self.input_dim_v, args.embd_size_v, embd_method=args.embd_method_v)
    
    def set_input(self, input):
        """
        Unpack input data from the dataloader
        """
        self.acoustic = input['A_feat']
        self.lexical = input['L_feat']
        self.visual = input['V_feat']
          
    def forward(self):
        final_embd = []
        
        if 'A' in self.modality:
            feat_A = self.netA(self.acoustic)
            final_embd.append(feat_A)
        if 'L' in self.modality:
            feat_L = self.netL(self.lexical)
            final_embd.append(feat_L)
        if 'V' in self.modality:
            feat_V = self.netV(self.visual)
            final_embd.append(feat_V)
        
        # get model outputs
        feat = torch.cat(final_embd, dim=-1)
        self.logits, _ = self.netC(feat)

    def get_outputs(self):
        self.forward()
        return [self.logits]
    
    def get_loss(self, outputs):
        """Calculate the loss for back propagation"""
        return {}