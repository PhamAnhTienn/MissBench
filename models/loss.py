import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class MaskedCELoss(nn.Module):

    def __init__(self, args, reduction='mean'):
        super(MaskedCELoss, self).__init__()
        self.args = args
        self.loss = nn.NLLLoss(reduction=reduction)

    def forward(self, pred, target, umask):
        """
        pred -> [batch*seq_len, n_classes]
        target -> [batch*seq_len]
        umask -> [batch, seq_len]
        """
        umask = umask.view(-1) # [batch*seq_len]
        target = target.view(-1) # [batch*seq_len]
        pred = F.log_softmax(pred, dim=-1).view(-1, pred.shape[-1]) # [batch*seqlen, n_classes]
        
        if umask.shape[0] == target.shape[0]:
            target = target[umask.bool().squeeze()].long()  # [num_valid]
        if umask.shape[0] == pred.shape[0]:
            pred = pred[umask.bool().squeeze()]      # [num_valid, n_classes]
        
        loss = self.loss(pred.view(-1, pred.shape[-1]), target.view(-1))
        
        return loss
    
class MaskedMSELoss(nn.Module):

    def __init__(self, args, reduction='mean'):
        super(MaskedMSELoss, self).__init__()
        self.args = args
        self.loss = nn.MSELoss(reduction=reduction)

    def forward(self, pred, target, umask):
        """
        pred -> [batch, seq_len, dim]
        target -> [batch, seq_len, dim]
        umask -> [batch, seq_len]
        """
        last_view = pred.shape[-1]
        umask = umask.view(-1) # [batch*seq_len]
        pred = pred.view(-1, last_view) # [batch*seq_len, ?]
        target = target.view(-1, last_view) # [batch*seq_len, ?]
        
        if umask.shape[0] == target.shape[0]:
            target = target[umask.bool().squeeze()]  # [num_valid, ?]
        if umask.shape[0] == pred.shape[0]:
            pred = pred[umask.bool().squeeze()]      # [num_valid, ?]
            
        loss = self.loss(pred, target)
        return loss
    
    
class MaskedMAELoss(nn.Module):

    def __init__(self, args):
        super(MaskedMAELoss, self).__init__()
        self.args = args
        self.loss = nn.L1Loss(reduction='mean')

    def forward(self, pred, target, umask):
        """
        pred -> [batch*seq_len]
        target -> [batch*seq_len]
        umask -> [batch*seq_len]
        """
        last_view = pred.shape[-1]
        umask = umask.view(-1) # [batch*seq_len]
        pred = pred.view(-1, last_view) # [batch*seq_len, ?]
        target = target.view(-1, last_view) # [batch*seq_len, ?]
        
        if umask.shape[0] == target.shape[0]:
            target = target[umask.bool().squeeze()]  # [num_valid, ?]
        if umask.shape[0] == pred.shape[0]:
            pred = pred[umask.bool().squeeze()]      # [num_valid, ?]
            
        loss = self.loss(pred, target)
        return loss