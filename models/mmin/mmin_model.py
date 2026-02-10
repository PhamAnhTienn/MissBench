import torch
import os
from collections import OrderedDict
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from models.mmin.networks.classifier import FcClassifier
from models.mmin.networks.lstm import LSTMEncoder
from models.mmin.networks.textcnn import TextCNN
from models.mmin.networks.autoencoder import ResidualAE
from models.utt_fusion.utt_fusion_model import UttFusionModel
from models.loss import MaskedCELoss, MaskedMSELoss


class MMINModel(nn.Module):
    def __init__(self, args, base_encoder):
        super().__init__()
        self.device = args.device
        self.modal = args.modal
        self.input_dim_a = args.embedding_dim['A']
        self.input_dim_v = args.embedding_dim['V']
        self.input_dim_l = args.embedding_dim['L']
        
        cls_layers = list(map(lambda x: int(x), args.cls_layers.split(',')))
        
        self.model_names = ['netA', 'netV', 'netL', 'netC', 'netAE', 'netAE_cycle']
        self.mse_weight = args.mse_weight
        self.cycle_weight = args.cycle_weight
        
        # acoustic model
        self.netA = LSTMEncoder(self.input_dim_a, args.embd_size_a, embd_method=args.embd_method_a)
        # lexical model
        self.netL = TextCNN(self.input_dim_l, args.embd_size_l)
        # visual model
        self.netV = LSTMEncoder(self.input_dim_v, args.embd_size_v, embd_method=args.embd_method_v)

        # AE model
        AE_layers = list(map(lambda x: int(x), args.AE_layers.split(',')))
        AE_input_dim = args.embd_size_a + args.embd_size_v + args.embd_size_l
        self.netAE = ResidualAE(AE_layers, args.n_blocks, AE_input_dim, dropout=0, use_bn=False)
        
        
        if args.share_weight:
            self.netAE_cycle = self.netAE
        else:
            self.netAE_cycle = ResidualAE(AE_layers, args.n_blocks, AE_input_dim, dropout=0, use_bn=False)
            
        cls_input_size = AE_layers[-1] * args.n_blocks
        self.netC = FcClassifier(cls_input_size, cls_layers, output_dim=args.output_dim, dropout=args.dropout_rate, use_bn=args.bn)

        if self.training:
            self.load_pretrained_encoder(args) # load teacher model
            
        self.criterion_mse = MaskedMSELoss(args)
            
    def load_pretrained_encoder(self, args):
        pretrained_path = os.path.join(args.checkpoint_dir, f"utt_fusion_{args.dataset}.pt") # checkpoints/mmin
        print(f'[ Init ] Load pretrained encoder from {pretrained_path}')
        checkpoint = torch.load(pretrained_path, map_location=self.device)
        
        pretrained_config = checkpoint["args"]  # teacher model should be in test mode
        
        self.pretrained_encoder = UttFusionModel(pretrained_config, None)  
        self.pretrained_encoder.load_state_dict(checkpoint["state_dict"])
        self.pretrained_encoder.cuda()
        self.pretrained_encoder.eval()
    
    def post_process(self):
        if self.training:
            print('[ Init ] Load parameters from pretrained encoder network')
            self.netA.load_state_dict(self.pretrained_encoder.netA.state_dict())
            self.netV.load_state_dict(self.pretrained_encoder.netV.state_dict())
            self.netL.load_state_dict(self.pretrained_encoder.netL.state_dict())
            
    def set_input(self, input):
        """
        Unpack input data from the dataloader
        """
        acoustic = input['A_feat']
        lexical = input['L_feat']
        visual = input['V_feat']
            
        self.umask = input['umask']  # [batch]
        self.label = input['target']
        self.missing_index = [input[f'{m}_miss_index'] for m in self.modal]
        self.missing_index = torch.stack(self.missing_index, dim=-1).to(self.device)  # [?, len(modal)]
        # A modality
        self.A_miss_index = self.missing_index[:, 0].unsqueeze(1).unsqueeze(2)
        self.A_miss = acoustic * self.A_miss_index
        self.A_reverse = acoustic * -1 * (self.A_miss_index - 1)
        # V modality
        self.V_miss_index = self.missing_index[:, 1].unsqueeze(1).unsqueeze(2)
        self.V_miss = visual * self.V_miss_index
        self.V_reverse = visual * -1 * (self.V_miss_index - 1)
        # L modality
        self.L_miss_index = self.missing_index[:, 2].unsqueeze(1).unsqueeze(2)
        self.L_miss = lexical * self.L_miss_index
        self.L_reverse = lexical * -1 * (self.L_miss_index - 1)

    def forward(self):
        # get utt level representattion
        feat_A_miss = self.netA(self.A_miss) # (batch, emb_dim)
        feat_L_miss = self.netL(self.L_miss) # (batch, emb_dim)
        feat_V_miss = self.netV(self.V_miss) # (batch, emb_dim)
        # fusion miss
        self.feat_fusion_miss = torch.cat([feat_A_miss, feat_L_miss, feat_V_miss], dim=-1) # (batch, 3 * emb_dim)
        # calc reconstruction of teacher's output
        self.recon_fusion, latent = self.netAE(self.feat_fusion_miss)  # recon_fusion: (batch, 3 * emb_dim), latent: (batch, n_blocks * last_layer_dim)
        self.recon_cycle, latent_cycle = self.netAE_cycle(self.recon_fusion)
        # get fusion outputs for missing modality
        self.logits, _ = self.netC(latent) # (batch, output_dim)
        
        # for training (getting teacher's embedding)
        if self.training:
            with torch.no_grad():
                T_embd_A = self.pretrained_encoder.netA(self.A_reverse)
                T_embd_L = self.pretrained_encoder.netL(self.L_reverse)
                T_embd_V = self.pretrained_encoder.netV(self.V_reverse)
                self.T_embds = torch.cat([T_embd_A, T_embd_L, T_embd_V], dim=-1) # (batch, 3 * emb_dim)
    
    def get_outputs(self):
        self.forward()
        
        if self.training:
            return [self.logits, self.recon_fusion, self.recon_cycle, self.feat_fusion_miss, self.T_embds]

        return [self.logits]
    
    def get_loss(self, outputs):
        """Calculate the loss for back propagation"""
        if self.training:
            loss_mse = self.mse_weight * self.criterion_mse(outputs[4], outputs[1], self.umask)
            loss_cycle = self.cycle_weight * self.criterion_mse(outputs[3].detach(), outputs[2], self.umask)
        
            return {
                'mse_loss': loss_mse,
                'cycle_loss': loss_cycle
            }
        else:
            return {}