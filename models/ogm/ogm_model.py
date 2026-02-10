import torch
import torch.nn as nn
from .fusion_modules import SumFusion, ConcatFusion


class AVClassifierModel(nn.Module):
    def __init__(self, args, base_encoder):
        super().__init__()
        self.device = args.device
        self.fusion_method = args.fusion_method
        output_dim=args.output_dim
        self.modal = args.modal
        self.method = args.method
        self.alpha = args.alpha
        self.input_dim_a = args.embedding_dim['A']
        self.input_dim_v = args.embedding_dim['V']
        self.input_dim_l = args.embedding_dim['L']
        
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        
        self.hidden_dim = 512

        if self.fusion_method == 'sum':
            self.fusion_module = SumFusion(input_dim=self.hidden_dim, output_dim=output_dim)
        elif self.fusion_method == 'concat':
            self.fusion_module = ConcatFusion(input_dim=self.hidden_dim * 3, output_dim=output_dim)
        else:
            raise NotImplementedError('Incorrect fusion method: {}!'.format(self.fusion_method))

        self.audio_net = base_encoder(self.input_dim_a, self.hidden_dim, dropout=0.5, bidirectional=args.bi_enc, batch_first=args.batch_first)
        self.visual_net = base_encoder(self.input_dim_v, self.hidden_dim, dropout=0.5, bidirectional=args.bi_enc, batch_first=args.batch_first)
        self.lexical_net = base_encoder(self.input_dim_l, self.hidden_dim, dropout=0.5, bidirectional=args.bi_enc, batch_first=args.batch_first)
        
    def set_input(self, input):
        """
        Unpack input data from the dataloader
        """
        acoustic = input['A_feat']
        lexical = input['L_feat']
        visual = input['V_feat']
            
        self.umask = input['umask']  # [batch, seqlen]
        self.label = input['target']
        self.missing_index = [input[f'{m}_miss_index'] for m in self.modal]
        self.missing_index = torch.stack(self.missing_index, dim=-1).to(self.device)  # [?, len(modal)]

        # A modality
        self.A_miss_index = self.missing_index[:, :, 0].unsqueeze(2)
        self.A_miss = acoustic * self.A_miss_index
        # V modality
        self.V_miss_index = self.missing_index[:, :, 1].unsqueeze(2)
        self.V_miss = visual * self.V_miss_index
        # L modality
        self.L_miss_index = self.missing_index[:, :, 2].unsqueeze(2)
        self.L_miss = lexical * self.L_miss_index

    def forward(self):
        a = self.audio_net(self.A_miss)
        v = self.visual_net(self.V_miss)
        t = self.lexical_net(self.L_miss)

        self.a, self.v, self.t, self.logits = self.fusion_module(a, v, t)
        
        self.unimodal_logits = self.get_unimodal_logits(a, v, t)
    
    def get_outputs(self):
        self.forward()
        return [self.logits] 
    
    def get_loss(self, outputs):
        """Calculate the loss for back propagation"""
        return {}
    
    def get_unimodal_logits(self, a, v, t):
        unimodal_logits = {}
        features = {'A': a, 'V': v, 'L': t}
        
        if self.fusion_method == 'sum':
            if 'A' in self.modal: unimodal_logits['A'] = self.fusion_module.fc_x(a)
            if 'V' in self.modal: unimodal_logits['V'] = self.fusion_module.fc_y(v)
            if 'L' in self.modal: unimodal_logits['L'] = self.fusion_module.fc_z(t)

        elif self.fusion_method == 'concat':
            weight = self.fusion_module.fc_out.weight # [output_dim, total_input_dim]
            bias = self.fusion_module.fc_out.bias     # [output_dim]
            
            idx = 0
            num_modals = len(self.modal)
            
            for m in ['A', 'V', 'L']:
                if m not in self.modal: continue
                
                feat = features[m]
                feat_dim = feat.size(2)
                sub_weight = weight[:, idx : idx + feat_dim]
                logit_m = torch.matmul(feat, sub_weight.t()) + (bias / num_modals)
                unimodal_logits[m] = logit_m
                idx += feat_dim
                
        return unimodal_logits
    
    def apply_modulation(self, outputs):
        modality_nums = len(self.modal)
        scores = {}
        ratios = {}
        coeffs = {}
        
        minscore = float('inf')
        
        for m in self.modal:
            logits_m = self.unimodal_logits[m]
            
            if modality_nums == 2:
                score_modality = sum([self.softmax(logits_m)[i][self.label[i]] for i in range(logits_m.size(0))])
            elif modality_nums == 3:
                score_modality = sum([self.softmax(torch.cos(logits_m))[i][self.label[i]] if self.label[i] == torch.argmax(logits_m[i]) else 0 for i in range(logits_m.size(0))])
            else:
                raise("Wrong number of modalitys for OGM, it should be 2 or 3, but given {:0}".format(modality_nums))
            
            try:
                scores[m] = score_modality.detach().clone()
            except:
                continue
        minscore = min(score_modality, minscore)
        
        ##Calculate the ratios
        for m in self.modal:
            ratios[m] = scores[m].detach().clone()
            if modality_nums == 2:
                for modality_another in self.modal:
                    if modality_another == m: 
                        continue
        
                    ratios[m] /= (scores[modality_another]+ 1e-5) # prevent OOM
            if modality_nums == 3:
                ratios[m] /= (minscore + 1e-5)
                
        #Calculate the coeffects
        for m in self.modal:
            if ratios[m] > 1 :
                coeffs[m] = max(1 - self.tanh(self.alpha * self.relu(ratios[m])),0)
            else:
                coeffs[m] = 1
                
        for name, parms in self.named_parameters():
            layer = str(name)
            for m in self.modal:
                if m in layer and len(parms.grad.size()) != 1: ##Don't change the grad of bias for layer
                    if self.method == 'OGM_GE':  # bug fixed
                        parms.grad = parms.grad * coeffs[m] + \
                                    torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)
                    elif self.method == 'OGM':
                        parms.grad *= coeffs[m] 