import torch
import torch.nn as nn
import numpy as np

class SimpleModel(nn.Module):
    def __init__(self, args, base_encoder):
        super().__init__()
        self.imputation = args.imputation
        self.device = args.device
        self.modal = args.modal
        self.input_dim_a = args.embedding_dim['A']
        self.input_dim_v = args.embedding_dim['V']
        self.input_dim_l = args.embedding_dim['L']

        self.output_layer = nn.Linear(self.input_dim_a + self.input_dim_v + self.input_dim_l, args.output_dim)
    
    def set_input(self, input):
        self.acoustic = input['A_feat']
        self.lexical = input['L_feat']
        self.visual = input['V_feat']
            
        self.label = input['target'].to(self.device)
        self.missing_index = [input[f'{m}_miss_index'] for m in self.modal]
        self.missing_index = torch.stack(self.missing_index, dim=-1).to(self.device)  # [?, len(modal)]
        
        self.A_mask = self.missing_index[:, :, 0:1]
        self.V_mask = self.missing_index[:, :, 1:2]
        self.L_mask = self.missing_index[:, :, 2:3]

    def impute(self, tensor, mask, modality):
        # 1. Zero Imputation
        if self.imputation == 'zero':
            return tensor * mask
        else:
            raise ValueError("Unknown imputation method")
        
    def forward(self):
        A_fused = self.impute(self.acoustic, self.A_mask, 'A')
        V_fused = self.impute(self.visual, self.V_mask, 'V')
        L_fused = self.impute(self.lexical, self.L_mask, 'L')
        
        # Concatenate
        fused = torch.cat([A_fused, L_fused, V_fused], dim=-1)

        self.logits = self.output_layer(fused)
        
    def get_outputs(self):
        self.forward()
        return [self.logits] 
    
    def get_loss(self, outputs):
        """Calculate the loss for back propagation"""
        return {}
    