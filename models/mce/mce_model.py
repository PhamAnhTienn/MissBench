import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.mce.networks.lstm import LSTMEncoder
from models.mce.networks.textcnn import TextCNN
from models.mce.transformer import Transformer
from models.mce.networks.classifier import FcClassifier
from models.loss import MaskedCELoss, MaskedMSELoss

num_modals = 3


class MCEModel(nn.Module):
    def update_unimodal(self, model_dict):
        self.uni_A = model_dict['A']
        self.uni_L = model_dict['L']
        self.uni_V = model_dict['V']
        for network in [self.uni_A, self.uni_L, self.uni_V]:
            for name, param in network['encoder'].named_parameters():
                param.requires_grad = False
            for name, param in network['classifier'].named_parameters():
                param.requires_grad = False
            network['encoder'].to(self.device).eval()
            network['classifier'].to(self.device).eval()

    def unimodal_forward(self, x, modal_name):
        if modal_name == 'A':
            encoder, classifier = self.uni_A['encoder'], self.uni_A['classifier']
        elif modal_name == 'L':
            encoder, classifier = self.uni_L['encoder'], self.uni_L['classifier']
        elif modal_name == 'V':
            encoder, classifier = self.uni_V['encoder'], self.uni_V['classifier']
        else: raise
        with torch.no_grad():
            output = encoder(x)
            output, _ = classifier(output)
        return output

    def __init__(self, args, base_encoder):
        super().__init__()
        self.device = args.device
        self.modal = args.modal
        self.input_dim_a = args.embedding_dim['A']
        self.input_dim_v = args.embedding_dim['V']
        self.input_dim_l = args.embedding_dim['L']
        self.output_dim = args.output_dim
        
        self.model_names = ['netA', 'netV', 'netL', 'netInter', 'netCls', 'netCls_a', 'netCls_v', 'netCls_l']
        self.modal_set = [(1,0,0), (0,1,0), (0,0,1), (1,1,0), (1,0,1), (0,1,1), (1,1,1)]
        
        missing_rate = args.modal_MR

        if len(missing_rate) == 1:
            missing_rate = missing_rate * 3

        missing_rate = np.array(missing_rate, dtype=float)
        self.modal_weight_org = 1.0 / (1.0 - missing_rate)

        # acoustic model
        self.netA = LSTMEncoder(self.input_dim_a, args.embd_size_a, embd_method=args.embd_method_a)
        # lexical model
        self.netL = TextCNN(self.input_dim_l, args.embd_size_l)
        # visual model
        self.netV = LSTMEncoder(self.input_dim_v, args.embd_size_v, embd_method=args.embd_method_v)

        # cross attention
        assert args.embd_size_a == args.embd_size_l == args.embd_size_v
        # Trong __init__
        self.netA_pos = nn.Parameter(torch.zeros(1, args.embd_size_a))
        self.netL_pos = nn.Parameter(torch.zeros(1, args.embd_size_l))
        self.netV_pos = nn.Parameter(torch.zeros(1, args.embd_size_v))
        self.netInter = Transformer(args.embd_size_a, mlp_dim=2048)

        # classifier model
        cls_layers = list(map(lambda x: int(x), args.cls_layers.split(',')))
        self.netCls = FcClassifier(args.embd_size_a + args.embd_size_l + args.embd_size_v, cls_layers, output_dim=args.output_dim, dropout=args.dropout_rate, use_bn=args.bn)
        self.netCls_a = FcClassifier(args.embd_size_a, cls_layers, output_dim=args.output_dim, dropout=args.dropout_rate, use_bn=args.bn)
        self.netCls_l = FcClassifier(args.embd_size_l, cls_layers, output_dim=args.output_dim, dropout=args.dropout_rate, use_bn=args.bn)
        self.netCls_v = FcClassifier(args.embd_size_v, cls_layers, output_dim=args.output_dim, dropout=args.dropout_rate, use_bn=args.bn)
            
        if args.regression:
            self.task_criterion = MaskedMSELoss(args)
        else:
            self.task_criterion = MaskedCELoss(args)
        self.task_weight = args.task_weight
        self.rec_weight = args.rec_weight
        self.shap_weight = args.shap_weight
        self.sep_weight = args.sep_weight
        self.sub_weight = args.sub_weight
        self.use_a = args.use_a
        self.use_b = args.use_b

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
        
        if self.training:
            self.modal_weight = torch.from_numpy(self.modal_weight_org).float().to(self.device)
        
        # A modality
        self.A_miss_index = self.missing_index[:, 0].unsqueeze(1).unsqueeze(2)
        self.A_miss = acoustic * self.A_miss_index
        # self.A_reverse = acoustic * -1 * (self.A_miss_index - 1)
        # V modality
        self.V_miss_index = self.missing_index[:, 1].unsqueeze(1).unsqueeze(2)
        self.V_miss = visual * self.V_miss_index
        # self.V_reverse = visual * -1 * (self.V_miss_index - 1)
        # L modality
        self.L_miss_index = self.missing_index[:, 2].unsqueeze(1).unsqueeze(2)
        self.L_miss = lexical * self.L_miss_index
        # self.L_reverse = lexical * -1 * (self.L_miss_index - 1)

    def pretrain_unimodal(self, unimodal):
        # get utt level representattion
        if unimodal == 'A':
            need_train = True if self.missing_index[:, 0].sum() > 0 else False
            self.feat_A_r = self.netA(self.A_miss[self.missing_index[:, 0].bool()])
            self.logits_a, _ = self.netCls_a(self.feat_A_r)
            loss = self.task_criterion(self.logits_a, self.label[self.missing_index[:, 0].bool()])
        elif unimodal == 'L':
            need_train = True if self.missing_index[:, 2].sum() > 0 else False
            self.feat_L_r = self.netL(self.L_miss[self.missing_index[:, 2].bool()])
            self.logits_l, _ = self.netCls_l(self.feat_L_r)
            loss = self.task_criterion(self.logits_l, self.label[self.missing_index[:, 2].bool()])
        elif unimodal == 'V':
            need_train = True if self.missing_index[:, 1].sum() > 0 else False
            self.feat_V_r = self.netV(self.V_miss[self.missing_index[:, 1].bool()])
            self.logits_v, _ = self.netCls_v(self.feat_V_r)
            loss = self.task_criterion(self.logits_v, self.label[self.missing_index[:, 1].bool()])

        if need_train:
            self.loss_CE = loss
            self.optimizer.zero_grad()  
            self.loss_CE.backward()
            self.optimizer.step()
            for model in self.model_names:
                torch.nn.utils.clip_grad_norm_(getattr(self, 'net'+model).parameters(), 1.0)
        else:
            self.loss_CE = 0
        self.loss_rec = 0
        self.loss_shap = 0
        self.loss_sep = 0
        self.loss_subfuse = 0

        if unimodal == 'A':
            modal_model_dict = {'encoder': self.netA, 'classifier': self.netCls_a}
        elif unimodal == 'L':
            modal_model_dict = {'encoder': self.netL, 'classifier': self.netCls_l}
        elif unimodal == 'V':
            modal_model_dict = {'encoder': self.netV, 'classifier': self.netCls_v}
        return need_train, modal_model_dict

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # get utt level representattion
        # print(self.A_miss)
        self.feat_A_miss = self.netA(self.A_miss)
        # print(self.feat_A_miss)
        self.feat_L_miss = self.netL(self.L_miss)
        self.feat_V_miss = self.netV(self.V_miss)

        self.feat_fusion_miss = torch.stack([self.feat_A_miss, self.feat_V_miss, self.feat_L_miss], dim=1)
        multimodal_pos = torch.stack((self.netA_pos, self.netV_pos, self.netL_pos), dim=1)
        self.feat_fusion_rec = self.netInter(self.feat_fusion_miss, multimodal_pos)

        # fusion miss
        # self.feat_fusion_miss = torch.cat([self.feat_A_miss, self.feat_L_miss, self.feat_V_miss], dim=-1)
        self.feat_A_r, self.feat_V_r, self.feat_L_r = self.feat_fusion_rec[:, 0], self.feat_fusion_rec[:, 1], self.feat_fusion_rec[:, 2]
        self.feat_fusion_r = torch.cat([self.feat_A_r, self.feat_V_r, self.feat_L_r], dim=-1)
        self.logits, _ = self.netCls(self.feat_fusion_r)
        self.pred = F.softmax(self.logits, dim=-1)

    def get_outputs(self):
        self.forward()
        return [self.logits]

    def get_loss(self, outputs):
        """Calculate the loss for back propagation"""
        B = self.logits.shape[0]
        multimodal_pos = torch.stack((self.netA_pos, self.netV_pos, self.netL_pos), dim=1)
        
        self.logits_a, _ = self.netCls_a(self.feat_A_r)
        self.logits_l, _ = self.netCls_l(self.feat_L_r)
        self.logits_v, _ = self.netCls_v(self.feat_V_r)

        sep_loss = torch.zeros(3).float().to(self.device)
        
        valid_a = self.missing_index[:, 0].bool()
        if valid_a.sum() > 0:
            sep_loss[0] = self.task_criterion(
                self.logits_a[valid_a],
                self.label[valid_a],    
                self.umask[valid_a]     
            ) #*self.task_weight
        else:
            sep_loss[0] = 0.0  
        
        valid_v = self.missing_index[:, 1].bool()
        if valid_v.sum() > 0:
            sep_loss[1] = self.task_criterion(
                self.logits_v[valid_v],
                self.label[valid_v],
                self.umask[valid_v]
            ) #*self.task_weight
        else:
            sep_loss[1] = 0.0
        
        valid_l = self.missing_index[:, 2].bool()
        if valid_l.sum() > 0:
            sep_loss[2] = self.task_criterion(
                self.logits_l[valid_l],
                self.label[valid_l],
                self.umask[valid_l]
            ) #*self.task_weight
        else:
            sep_loss[2] = 0.0

        # subset loss
        rec_loss = torch.zeros(B,3).float().to(self.device)
        subfuse_loss = torch.zeros(B,1).float().to(self.device)
        
        # individual prediction
        # [B]
        a_pred = F.softmax(self.logits_a, dim=-1).argmax(dim=1)
        l_pred = F.softmax(self.logits_l, dim=-1).argmax(dim=1)
        v_pred = F.softmax(self.logits_v, dim=-1).argmax(dim=1)
        # a_pred = F.softmax(self.unimodal_forward(self.A_miss, 'A'), dim=-1).argmax(dim=1)
        # l_pred = F.softmax(self.unimodal_forward(self.L_miss, 'L'), dim=-1).argmax(dim=1)
        # v_pred = F.softmax(self.unimodal_forward(self.V_miss, 'V'), dim=-1).argmax(dim=1)

        # sample level [B, num_modal] ×
        # batch level [num_modal] √
        independent_contrib = torch.stack([
            ((a_pred == self.label) * self.missing_index[:, 0]).sum().float(),
            ((l_pred == self.label) * self.missing_index[:, 2]).sum().float(),
            ((v_pred == self.label) * self.missing_index[:, 1]).sum().float()
        ], dim=0)

        bs_subsets = self.gen_bs_subsets(self.missing_index, self.device)
        bs_individual_contrib = dict()
        bs_multimodal_contrib = dict()
        for value in self.modal_set:
            if sum(value) > 1:
                bs_multimodal_contrib[value] = 0
            else:
                bs_individual_contrib[value] = 0
            
        # for each sample
        for b in range(B):
            # shape: subsets - (len_subsets, len_modal)
            subsets = bs_subsets[b]

            rec_loss_a, rec_loss_l, rec_loss_v = torch.zeros(1).float().to(self.device), torch.zeros(1).float().to(self.device), torch.zeros(1).float().to(self.device)
            for _subset in subsets:
                if torch.equal(_subset, self.missing_index[b].int()):
                    fused_pred = self.pred[b:b+1]
                else:
                    a_f = self.feat_A_r[b:b+1] if _subset[0] == 1 else torch.zeros((1, self.feat_A_r.shape[1])).to(self.device)
                    l_f = self.feat_L_r[b:b+1] if _subset[2] == 1 else torch.zeros((1, self.feat_L_r.shape[1])).to(self.device)
                    v_f = self.feat_V_r[b:b+1] if _subset[1] == 1 else torch.zeros((1, self.feat_V_r.shape[1])).to(self.device)

                    f_rec = self.netInter(torch.stack((a_f, v_f, l_f), dim=1), multimodal_pos)
                    a_f_rec, v_f_rec, l_f_rec = f_rec[:, 0], f_rec[:, 1], f_rec[:, 2]
                    # Decoder
                    fused_pred, _ = self.netCls(torch.cat([a_f_rec, v_f_rec, l_f_rec], dim=-1))
                    # compute subset fused loss
                    subfuse_loss[b:b+1] += self.task_criterion(fused_pred, self.label[b:b+1], self.umask[b:b+1]) #*self.task_weight
                    
                    # compute auxiliary completion loss
                    # rec_mask&_modal_weight shape: (1,modal_num)
                    # rec_mask = (self.missing_index[b] - _subset).view(1,-1)
                    # _modal_weight = self.modal_weight.view(1,-1)
                    # rec_loss[b] += F.l1_loss(
                    #     torch.stack([self.feat_A_r[b:b+1], self.feat_L_r[b:b+1], self.feat_V_r[b:b+1]], dim=1) * _modal_weight * rec_mask,
                    #     torch.stack([a_f_rec, l_f_rec, v_f_rec], dim=1) * _modal_weight * rec_mask
                    # )
                    rec_mask = (self.missing_index[b] - _subset)
                    rec_loss_a += F.l1_loss(self.feat_A_r[b:b+1], a_f_rec) * rec_mask[0]
                    rec_loss_l += F.l1_loss(self.feat_L_r[b:b+1], l_f_rec) * rec_mask[2]
                    rec_loss_v += F.l1_loss(self.feat_V_r[b:b+1], v_f_rec) * rec_mask[1]
                
                # 计算两个张量中元素相等的位置，并求和得到相等元素的总数
                pred = F.softmax(fused_pred, dim=-1).argmax(dim=1)
                if (_subset).sum() > 1:
                    bs_multimodal_contrib[tuple(_subset.cpu().numpy())] += (pred == self.label[b:b+1]).sum().float()
                else:
                    bs_individual_contrib[tuple(_subset.cpu().numpy())] += (pred == self.label[b:b+1]).sum().float()
            rec_loss[b] = torch.cat([rec_loss_a, rec_loss_v, rec_loss_l], dim=0)

        # torch.Size([num_modal])
        bs_shap_value = self.shapley_value(bs_multimodal_contrib, bs_individual_contrib, self.device)
        shap_mask = (bs_shap_value < independent_contrib).int().to(self.device)
        
        available_modals = self.missing_index.sum(dim=0).float()
        available_modals = torch.clamp(available_modals, min=1.0)
        shap_loss = (independent_contrib - bs_shap_value) * shap_mask / available_modals

        # sample_mod_count&sample_exist_mask shape (b, 1)
        sample_subset_count = 2**self.missing_index.sum(dim=1, keepdim=True) - 1
        sample_exist_mask = (sample_subset_count > 0).float()
        sample_mod_count = sample_subset_count * sample_exist_mask + (1 - sample_exist_mask)
        
        # torch.Size([128, 1]) torch.Size([2]) torch.Size([128, 2]) torch.Size([128, 1])
        if self.use_a:
            rec_loss = rec_loss * self.modal_weight.view(1,-1)
            sep_loss = sep_loss * self.modal_weight
        if self.use_b:
            rec_loss = rec_loss * shap_loss.view(1,-1)
            sep_loss = sep_loss * shap_loss
        
        # rec_loss = rec_loss * shap_loss.view(1,-1)
        self.loss_rec = (rec_loss / (sample_mod_count)).sum() / sample_exist_mask.sum()

        self.loss_shap = shap_loss.sum()
        # sep_loss = sep_loss * shap_loss

        self.loss_sep = sep_loss.mean()
        
        self.loss_subfuse = (subfuse_loss / (sample_mod_count)).sum() / sample_exist_mask.sum()

        return {
            'recon_loss': self.loss_rec * self.rec_weight,
            'sep_loss': self.loss_sep * self.sep_weight,
            'subfuse_loss': self.loss_subfuse * self.sub_weight
        }

    def cal_shap(self):
        """Calculate the loss for back propagation"""
        B = self.logits.shape[0]
        multimodal_pos = torch.stack((self.netA_pos, self.netV_pos, self.netL_pos), dim=1)
        
        self.logits_a, _ = self.netCls_a(self.feat_A_r)
        self.logits_l, _ = self.netCls_l(self.feat_L_r)
        self.logits_v, _ = self.netCls_v(self.feat_V_r)
        
        # individual prediction
        # [B]
        a_pred = F.softmax(self.logits_a, dim=-1).argmax(dim=1)
        l_pred = F.softmax(self.logits_l, dim=-1).argmax(dim=1)
        v_pred = F.softmax(self.logits_v, dim=-1).argmax(dim=1)
        # a_pred = F.softmax(self.unimodal_forward(self.A_miss, 'A'), dim=-1).argmax(dim=1)
        # l_pred = F.softmax(self.unimodal_forward(self.L_miss, 'L'), dim=-1).argmax(dim=1)
        # v_pred = F.softmax(self.unimodal_forward(self.V_miss, 'V'), dim=-1).argmax(dim=1)

        # sample level [B, num_modal] ×
        # batch level [num_modal] √
        self.independent_contrib = torch.stack([
            ((a_pred == self.label) * self.missing_index[:, 0]).sum().float(),
            ((l_pred == self.label) * self.missing_index[:, 2]).sum().float(),
            ((v_pred == self.label) * self.missing_index[:, 1]).sum().float()
        ], dim=0)

        bs_subsets = self.gen_bs_subsets(self.missing_index, self.device)
        bs_individual_contrib = dict()
        bs_multimodal_contrib = dict()
        for value in self.modal_set:
            if sum(value) > 1:
                bs_multimodal_contrib[value] = 0
            else:
                bs_individual_contrib[value] = 0
            
        # for each sample
        for b in range(B):
            # shape: subsets - (len_subsets, len_modal)
            subsets = bs_subsets[b]

            for _subset in subsets:
                if torch.equal(_subset, self.missing_index[b].int()):
                    fused_pred = self.pred[b:b+1]
                else:
                    a_f = self.feat_A_r[b:b+1] if _subset[0] == 1 else torch.zeros((1, self.feat_A_r.shape[1])).to(self.device)
                    l_f = self.feat_L_r[b:b+1] if _subset[2] == 1 else torch.zeros((1, self.feat_L_r.shape[1])).to(self.device)
                    v_f = self.feat_V_r[b:b+1] if _subset[1] == 1 else torch.zeros((1, self.feat_V_r.shape[1])).to(self.device)

                    f_rec = self.netInter(torch.stack((a_f, v_f, l_f), dim=1), multimodal_pos)
                    a_f_rec, v_f_rec, l_f_rec = f_rec[:, 0], f_rec[:, 1], f_rec[:, 2]
                    # Decoder
                    fused_pred, _ = self.netCls(torch.cat([a_f_rec, v_f_rec, l_f_rec], dim=-1))
                
                # 计算两个张量中元素相等的位置，并求和得到相等元素的总数
                pred = F.softmax(fused_pred, dim=-1).argmax(dim=1)
                if (_subset).sum() > 1:
                    bs_multimodal_contrib[tuple(_subset.cpu().numpy())] += (pred == self.label[b:b+1]).sum().float()
                else:
                    bs_individual_contrib[tuple(_subset.cpu().numpy())] += (pred == self.label[b:b+1]).sum().float()
        
        # torch.Size([num_modal])
        self.bs_shap_value = self.shapley_value(bs_multimodal_contrib, bs_individual_contrib, self.device)

    def gen_subsets(self, lst):
        # 计算子集的总数（2的列表长度次方减去1，因为不包括空集）
        num_subsets = 2 ** len(lst) - 1
        # 使用位运算生成所有子集  
        subsets = [[lst[i] for i, subset_bit in enumerate(bin(subset)[2:].zfill(len(lst))) if subset_bit == '1'] for subset in range(1, num_subsets + 1)]
        return subsets

    def indices_to_tensor(self, indices_list, L, device):
        B = len(indices_list)
        # 创建一个全False的tensor，dtype为torch.bool
        tensor = torch.zeros((B, L), dtype=torch.int)
        # 使用scatter_方法根据索引设置True
        for i, indices in enumerate(indices_list):
            if indices:  # 确保indices不是空列表
                tensor[i].scatter_(0, torch.tensor(indices, dtype=torch.long), 1)
        return tensor.to(device)
    
    def gen_bs_subsets(self, modal_mask, device):
        # modal_mask shape: B, len_modal
        B, modal_num = modal_mask.shape
        # for each sample
        bs_subsets = []
        for b in range(B):
            # 使用nonzero获取非零元素的索引，并将其从tensor转为numpy然后转为list
            indices = self.gen_subsets(modal_mask[b].nonzero(as_tuple=True)[0].tolist())
            bs_subsets.append(self.indices_to_tensor(indices, num_modals, device))
        return bs_subsets

    def shapley_value(self, multimodal_contrib, individual_contrib, device):
        values = torch.zeros((num_modals)).to(device)
            
        for idx in range(num_modals):
            # the number of avaliable modalitites
            #L = int(torch.sum((torch.sum(self.missing_index, dim=0) > 0).float()).item())
            L = int(torch.sum((self.missing_index.sum(dim=0) > 0).float()).item())
            # retreive i-th modality individual contribution
            idx_list = [0] * num_modals
            idx_list[idx] = 1
            contrib_idx = individual_contrib[tuple(idx_list)]

            # compute marginal contribution
            # check if the i-th modality is avaliable and not only avaliable independently
            for key in multimodal_contrib:
                cardinal = sum(key) - 1
                
                if L - cardinal - 1 < 0:
                    continue
                
                # (contribution with i-th modality) - (contribution only i-th modality)
                contrib_diff = multimodal_contrib[key] - contrib_idx
                assert cardinal > 0
                weight = (math.factorial(cardinal) * math.factorial(L-cardinal-1) / math.factorial(L)) # Weight = |S|!(n-|S|-1)!/n!
                values[idx] += weight * contrib_diff
        
            # Add the term corresponding to the empty set
            values[idx] += contrib_idx / L
        return values