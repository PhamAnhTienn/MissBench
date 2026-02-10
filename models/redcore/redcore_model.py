import numpy as np
import torch
import torch.nn as nn
from models.redcore.networks.transformer import Transformer
from models.redcore.networks.classifier import FcClassifier
from models.redcore.networks.autoencoder import ResidualXE, ResidualAE
from models.loss import MaskedCELoss, MaskedMSELoss


class RedCoreModel(nn.Module):
    def __init__(self, args, base_encoder):
        super().__init__()
        self.device = args.device
        self.modal = args.modal
        self.input_dim_a = args.embedding_dim['A']
        self.input_dim_v = args.embedding_dim['V']
        self.input_dim_l = args.embedding_dim['L']
        self.model_names = ['netA', 'netV', 'netL', 'netC', 'netAE', 'netAE_cycle', 'netCls', 'netxenc_al2v', 'netxenc_av2l', 'netxenc_vl2a', 'netCls_a', 'netCls_v', 'netCls_l']
        self.lossA = torch.tensor(0.0, device=self.device)
        self.lossV = torch.tensor(0.0, device=self.device)
        self.lossL = torch.tensor(0.0, device=self.device)
        self.loss_beta = args.beta   #0.95
        self.beta = np.array([1, 1, 1])
        self.iter = 0
        self.interval_i = args.ii
        self.etaext = args.etaext
        self.eta = args.eta
        self.mse_weight = args.mse_weight

        # acoustic model
        #self.netA = LSTMEncoder(self.input_dim_a, opt.embd_size_a, embd_method=opt.embd_method_a)
        self.netA = Transformer(self.input_dim_a, 3, 8, args.embd_size_a)
        self.netA.initialize_parameters()
        # lexical model
        #self.netL = TextCNN(self.input_dim_l, args.embd_size_l)
        self.netL = Transformer(self.input_dim_l, 3, 8, args.embd_size_l)
        self.netL.initialize_parameters()
        # visual model
        #self.netV = LSTMEncoder(self.input_dim_v, args.embd_size_v, args.embd_method_v)
        self.netV = Transformer(self.input_dim_v, 3, 8, args.embd_size_v)
        self.netV.initialize_parameters()
        # AE model
        AE_layers = list(map(lambda x: int(x), args.AE_layers.split(',')))
        AE_input_dim = args.embd_size_a + args.embd_size_v + args.embd_size_l
        self.netAE = ResidualAE(AE_layers, args.n_blocks, AE_input_dim, dropout=0, use_bn=False)
        if args.share_weight:
            self.netAE_cycle = self.netAE
            self.model_names.pop(-1)
        else:
            self.netAE_cycle = ResidualAE(AE_layers, args.n_blocks, AE_input_dim, dropout=0, use_bn=False)
        cls_layers = list(map(lambda x: int(x), args.cls_layers.split(',')))
        cls_input_size = AE_layers[-1] * args.n_blocks
        self.netC = FcClassifier(cls_input_size, cls_layers, output_dim=args.output_dim, dropout=args.dropout_rate, use_bn=args.bn)
        self.feature_dim = 32
        self.netxenc_al2v = ResidualXE(AE_layers, args.n_blocks, args.embd_size_a + args.embd_size_l, args.embd_size_v, dropout=0, use_bn=False)
        self.netxenc_av2l = ResidualXE(AE_layers, args.n_blocks, args.embd_size_a + args.embd_size_v, args.embd_size_l, dropout=0, use_bn=False)
        self.netxenc_vl2a = ResidualXE(AE_layers, args.n_blocks, args.embd_size_v + args.embd_size_l, args.embd_size_a, dropout=0, use_bn=False)
        self.smax_fc = FcClassifier(args.embd_size_a + args.embd_size_l + args.embd_size_v, cls_layers, output_dim=args.output_dim, dropout=args.dropout_rate, use_bn=args.bn)
        self.smax_fc_a = FcClassifier(args.embd_size_a, cls_layers, output_dim=args.output_dim, dropout=args.dropout_rate, use_bn=args.bn)
        self.smax_fc_v = FcClassifier(args.embd_size_v, cls_layers, output_dim=args.output_dim, dropout=args.dropout_rate, use_bn=args.bn)
        self.smax_fc_l = FcClassifier(args.embd_size_l, cls_layers, output_dim=args.output_dim, dropout=args.dropout_rate, use_bn=args.bn)
        
        if args.regression:
            self.task_criterion = MaskedMSELoss(args)
        else:
            self.task_criterion = MaskedCELoss(args)
        self.task_weight = args.task_weight
        self.criterion_mse = MaskedMSELoss(args)
            
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
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # get utt level representattion
        self.feat_A_miss, self.fmu_a, self.flogvar_a = self.netA(self.A_miss)
        self.feat_V_miss, self.fmu_v, self.flogvar_v = self.netV(self.V_miss)
        self.feat_L_miss, self.fmu_l, self.flogvar_l = self.netL(self.L_miss)
        
        # fusion miss
        self.feat_fusion_miss = torch.cat([self.feat_A_miss, self.feat_L_miss, self.feat_V_miss], dim=-1)
        # calc reconstruction of teacher's output
        self.recon_fusion, self.latent = self.netAE(self.feat_fusion_miss)
        self.recon_cycle, self.latent_cycle = self.netAE_cycle(self.recon_fusion)


        #generate the feature of missing modalities
        self.gen_a, latent_a = self.netxenc_vl2a(torch.cat([self.feat_V_miss, self.feat_L_miss], dim=-1))
        self.gen_l, latent_l = self.netxenc_av2l(torch.cat([self.feat_A_miss, self.feat_V_miss], dim=-1))
        self.gen_v, latent_v = self.netxenc_al2v(torch.cat([self.feat_A_miss, self.feat_L_miss], dim=-1))

        bs = self.feat_A_miss.shape[0]
        self.feat_A_r = self.A_miss_index.reshape(bs, 1) * self.feat_A_miss - (self.A_miss_index.reshape(bs, 1) - 1) * self.gen_a 
        self.feat_L_r = self.L_miss_index.reshape(bs, 1) * self.feat_L_miss - (self.L_miss_index.reshape(bs, 1) - 1) * self.gen_l 
        self.feat_V_r = self.V_miss_index.reshape(bs, 1) * self.feat_V_miss - (self.V_miss_index.reshape(bs, 1) - 1) * self.gen_v 

        self.feat_A_re = self.A_miss_index * (self.feat_A_miss - self.gen_a)
        self.feat_L_re = self.L_miss_index * (self.feat_L_miss - self.gen_l)
        self.feat_V_re = self.V_miss_index * (self.feat_V_miss - self.gen_v)
        self.feat_fusion_r = torch.cat([self.feat_A_r, self.feat_L_r, self.feat_V_r], dim=-1)
        self.logits, _ = self.smax_fc(self.feat_fusion_r)

        self.logits_a, _ = self.smax_fc_a(self.feat_A_r)
        self.logits_v, _ = self.smax_fc_v(self.feat_V_r)
        self.logits_l, _ = self.smax_fc_l(self.feat_L_r)
    
    def get_outputs(self):
        self.forward()
        return [self.logits]       
        
    def get_loss(self, outputs):
        """Calculate the loss for back propagation"""
        bs = self.A_miss_index.shape[0]
        a_index = self.A_miss_index.reshape(bs, 1)
        v_index = self.V_miss_index.reshape(bs, 1)
        l_index = self.L_miss_index.reshape(bs, 1)
        self.loss_CE_a = self.task_weight * self.task_criterion(self.logits_a, self.label, self.umask)
        self.loss_CE_v = self.task_weight * self.task_criterion(self.logits_v, self.label, self.umask)
        self.loss_CE_l = self.task_weight * self.task_criterion(self.logits_l, self.label, self.umask)
        lambda1 = 0.0008


        kld_fa = -1 * lambda1 * torch.sum((1 + self.flogvar_a - self.fmu_a.pow(2) - self.flogvar_a.exp()) * a_index) / bs
        kld_fv = -1 * lambda1 * torch.sum((1 + self.flogvar_v - self.fmu_v.pow(2) - self.flogvar_v.exp()) * v_index) / bs
        kld_fl = -1 * lambda1 * torch.sum((1 + self.flogvar_l - self.fmu_l.pow(2) - self.flogvar_l.exp()) * l_index) / bs


        bs_a = sum(self.A_miss_index)
        bs_v = sum(self.V_miss_index)
        bs_l = sum(self.L_miss_index)

        
        loss_mse_a = self.criterion_mse(self.gen_a*a_index, self.feat_A_miss*a_index, self.umask)
        if bs_a > 0:
            self.lossA_r = loss_mse_a / bs_a
        else:
            self.lossA_r = torch.tensor(0.0, device=self.device, requires_grad=True)

        loss_mse_v = self.criterion_mse(self.gen_v*v_index, self.feat_V_miss*v_index, self.umask)
        if bs_v > 0:
            self.lossV_r = loss_mse_v / bs_v
        else:
            self.lossV_r = torch.tensor(0.0, device=self.device, requires_grad=True)

        loss_mse_l = self.criterion_mse(self.gen_l*l_index, self.feat_L_miss*l_index, self.umask)
        if bs_l > 0:
            self.lossL_r = loss_mse_l / bs_l
        else:
            self.lossL_r = torch.tensor(0.0, device=self.device, requires_grad=True)

        # training parameter of pytorch module
        if self.training:
            if self.lossA_r.item() == 0:
                lossA_update = self.lossA.detach()
            else:
                lossA_update = self.lossA_r.detach()
            if self.lossV_r.item() == 0:
                lossV_update = self.lossV.detach()
            else:
                lossV_update = self.lossV_r.detach()
            if self.lossL_r.item() == 0:
                lossL_update = self.lossL.detach()
            else:
                lossL_update = self.lossL_r.detach()
                
            self.lossA = (1 - self.loss_beta) * self.lossA + self.loss_beta * lossA_update
            self.lossV = (1 - self.loss_beta) * self.lossV + self.loss_beta * lossV_update
            self.lossL = (1 - self.loss_beta) * self.lossL + self.loss_beta * lossL_update

            lossAVL = np.array([self.lossA.item(), self.lossV.item(), self.lossL.item()])
            loss_avg = sum(lossAVL) / 3
            ra = (loss_avg - lossAVL ) / loss_avg

            if self.iter % 500 ==0 :
                self.eta = self.eta * self.etaext
            if self.iter % self.interval_i == 0:
                self.beta = self.beta - self.eta * ra 
                self.beta[0] = max(0.1, self.beta[0])
                self.beta[1] = max(0.1, self.beta[1])
                self.beta[2] = max(0.1, self.beta[2])
                self.beta = self.beta / (sum(self.beta**2)**(0.5))
            self.iter += 1

        self.loss_mse = self.mse_weight * (self.beta[0] * self.lossA_r + self.beta[1] * self.lossV_r + self.beta[2] * self.lossL_r)

        self.loss_cycle = 0
        
        return {
            'loss_CE_a': self.loss_CE_a,
            'loss_CE_v': self.loss_CE_v,
            'loss_CE_l': self.loss_CE_l,
            'kld_fa': kld_fa,
            'kld_fv': kld_fv,
            'kld_fl': kld_fl,
            'recon_loss': self.loss_mse.squeeze()
        }