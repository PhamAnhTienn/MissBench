import torch
import torch.nn as nn
import torch.nn.functional as F

from .net import Net


class Ada2IModel(nn.Module):
    def __init__(self, args, base_encoder):
        super().__init__()
        self.device = args.device
        self.modal = args.modal
        self.input_dim_a = args.embedding_dim['A']
        self.input_dim_v = args.embedding_dim['V']
        self.input_dim_l = args.embedding_dim['L']
        self.regression = args.regression

        self.net = Net(args)
        self.criterion = nn.NLLLoss() if not self.regression else nn.MSELoss()
        self.args = args
        self.mmcosine = args.mmcosine
        self.modulation = args.modulation
        self.alpha = args.alpha

    def set_input(self, input):
        self.data = input
        
        batch = self.data['target'].shape[0]
        lengths = self.data['seq_lengths']
        
        self.labels = []
        
        for i in range(batch):
            label = self.data['target'][i][:lengths[i]]
            self.labels.extend(label)
            
        self.labels = torch.tensor(self.labels).to(self.device).long()

    def forward(self):
        out, m_out, self.amn_loss = self.net(self.data)
    
        weight_dict = self.net.get_fusion_weight()

        if not self.regression:
            with torch.no_grad():
                m_out = {
                    m: torch.matmul(m_out[m], torch.transpose(weight_dict[m]["weight"], 0, 1)) + weight_dict[m]["bias"]
                    for m in self.modal
                }
                self.prob_m = {
                    m: F.log_softmax(m_out[m], dim=1)
                    for m in self.modal
                }
                scores = {
                    m: sum([F.softmax(m_out[m], dim=1)[i][self.labels[i]] for i in range(out.size(0))])
                    for m in self.modal
                }

                min_score = min(scores.values())
                self.ratio = {
                    m: scores[m] / min_score
                    for m in self.modal
                }
            
            self.prob = F.log_softmax(out, dim=1)
        else:
            with torch.no_grad():
                m_out = {
                    m: torch.matmul(m_out[m], torch.transpose(weight_dict[m]["weight"], 0, 1)) + weight_dict[m]["bias"]
                    for m in self.modal
                }
                self.prob_m = {
                    m: m_out[m]
                    for m in self.modal
                }
                scores = {
                    m: sum([self.reg2prob(m_out[m])[i][(self.labels[i]>0).long()] if self.labels[i]!=0 else 0 for i in range(out.size(0))])
                    for m in self.modal
                }

                min_score = min(scores.values())
                self.ratio = {
                    m: scores[m] / min_score
                    for m in self.modal
                }
            self.prob = out

        return self.prob, self.amn_loss, self.prob_m, self.ratio # prob     : Tensor[N, C]    amn_loss : Tensor[]    prob_m   : Dict[str, Tensor[N, C]]    ratio    : Dict[str, float]
    
    def reg2prob(self, x, log=False):
        # Convert regression output to probability distribution
        x = x.squeeze(-1)
        bs = x.shape[0]
        if log:
            reg_output = torch.stack([F.logsigmoid(-x), F.logsigmoid(x)], -1)
        else:
            reg_output = torch.stack([F.sigmoid(-x), F.sigmoid(x)], -1)
        assert reg_output.shape == (bs, 2), f'{bs},{reg_output.shape}'  
        return reg_output
    
    def get_outputs(self):
        self.forward()
        
        lengths = self.data['seq_lengths']
        batch = len(lengths)
        max_len = max(lengths)
        dim = self.prob.shape[-1]

        self.logits = torch.zeros(batch, max_len, dim).to(self.prob.device)

        start = 0
        for i, length in enumerate(lengths):
            end = start + length
            sequence = self.prob[start:end, :] 
            self.logits[i, :length, :] = sequence
            start = end
        
        return [self.logits, self.ratio] 
    
    def get_loss(self, outputs):
        m_loss = {m: 0.0 for m in self.modal}

        # unimodal losses
        for m in self.modal:
            m_loss[m] += self.criterion(self.prob_m[m].squeeze(), self.labels).item()
        
        return {
            'amn_loss': self.amn_loss.squeeze()
        }
        
    def apply_modulation(self, outputs):
        ratio = outputs[1]
        params_dict = self.net.get_encoder_params()
        coeff = {
            m: 1 - F.tanh(self.alpha * F.relu(ratio[m], inplace=True)) if ratio[m] > 1 else 1
            for m in self.modal
        }

        for m in self.modal:
            for params in params_dict[m]:
                params.grad = params.grad * coeff[m]
                if self.args.normalize:
                    params.grad += torch.zeros_like(params.grad).normal_(0, params.grad.std().item() + 1e-8)
