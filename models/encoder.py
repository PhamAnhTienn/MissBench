import torch
import torch.nn as nn

class Transformer_Encoder(nn.Module):
    def __init__(self, input_size, hidden_dim, dropout=0.1, nlayers=8, bidirectional=True, batch_first=True):
        super(Transformer_Encoder, self).__init__()
        
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.batch_first = batch_first
        self.dropout = dropout
        self.output_dim = self.hidden_dim
        
        nhead = 1
        for h in range(5, 15):
            if self.input_size % h == 0:
                nhead = h
                break
            
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.input_size, nhead=nhead, dropout=self.dropout, batch_first=self.batch_first)
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.lin_out = torch.nn.Linear(self.input_size, self.hidden_dim, bias=True)
        
    def forward(self, x):
        x = self.transformer_encoder(x)
        x = self.lin_out(x)
        return x
    
class LSTM_Encoder(nn.Module):
    def __init__(self, input_size, hidden_dim, dropout=0.1, nlayers=2, bidirectional=True, batch_first=True):
        super(LSTM_Encoder, self).__init__()
        
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.batch_first = batch_first
        self.num_layers = nlayers
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.output_dim = self.hidden_dim * 2 if self.bidirectional else self.hidden_dim
        
        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_dim,
                            num_layers=self.num_layers,
                            batch_first=self.batch_first,
                            bidirectional=self.bidirectional,
                            dropout=self.dropout)
        
    def forward(self, x):
        outputs, _ = self.lstm(x)
        return outputs