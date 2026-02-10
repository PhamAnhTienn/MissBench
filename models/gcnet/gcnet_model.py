import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, GraphConv

from models.gcnet.module import *
from models.gcnet.graph import batch_graphify
from models.loss import MaskedMSELoss


class GraphNetwork(torch.nn.Module):
    def __init__(self, num_features, num_relations, time_attn, hidden_size=64, dropout=0.5, no_cuda=False):
        """
        The Speaker-level context encoder in the form of a 2 layer GCN.
        """
        super(GraphNetwork, self).__init__()
        self.no_cuda = no_cuda 
        self.time_attn = time_attn
        self.hidden_size = hidden_size

        ## graph modeling
        self.conv1 = RGCNConv(num_features, hidden_size, num_relations)
        self.conv2 = GraphConv(hidden_size, hidden_size)

        ## nodal attention
        D_h = num_features+hidden_size
        self.grufusion = nn.LSTM(input_size=D_h, hidden_size=D_h, num_layers=2, bidirectional=True, dropout=dropout)

        ## sequence attention
        self.matchatt = MatchingAttention(2*D_h, 2*D_h, att_type='general2')
        self.linear = nn.Linear(2*D_h, D_h)


    def forward(self, features, edge_index, edge_type, seq_lengths, umask):
        '''
        features: input node features: [num_nodes, in_channels]
        edge_index: [2, edge_num]
        edge_type: [edge_num]
        '''

        ## graph model: graph => outputs
        out = self.conv1(features, edge_index, edge_type) # [num_features -> hidden_size]
        out = self.conv2(out, edge_index) # [hidden_size -> hidden_size]
        outputs = torch.cat([features, out], dim=-1) # [num_nodes, num_features(16)+hidden_size(8)]

        ## change utterance to conversation: (outputs->outputs)
        outputs = outputs.reshape(-1, outputs.size(1)) # [num_utterance, dim]
        outputs = utterance_to_conversation(outputs, seq_lengths, umask, self.no_cuda) # [seqlen, batch, dim]
        outputs = outputs.reshape(outputs.size(0), outputs.size(1), 1, -1) # [seqlen, batch, ?, dim]

        ## outputs -> outputs:
        seqlen = outputs.size(0)
        batch = outputs.size(1)
        outputs = torch.reshape(outputs, (seqlen, batch, -1)) # [seqlen, batch, dim]
        outputs = self.grufusion(outputs)[0] # [seqlen, batch, dim]

        ## outputs -> hidden:
        ## sequence attention => [seqlen, batch, d_h]
        if self.time_attn:
            alpha = []
            att_emotions = []
            for t in outputs: # [bacth, dim]
                # att_em: [batch, mem_dim] # alpha_: [batch, 1, seqlen]
                att_em, alpha_ = self.matchatt(outputs, t, mask=umask)
                att_emotions.append(att_em.unsqueeze(0)) # [1, batch, mem_dim]
                alpha.append(alpha_[:,0,:]) # [batch, seqlen]
            att_emotions = torch.cat(att_emotions, dim=0) # [seqlen, batch, mem_dim]
            hidden = F.relu(self.linear(att_emotions)) # [seqlen, batch, D_h]
        else:
            alpha = []
            hidden = F.relu(self.linear(outputs)) # [seqlen, batch, D_h]

        return hidden # [seqlen, batch, D_h]

        
'''
adim, tdim, vdim: input feature dim
D_e: hidder feature dimensions of encoder
D_g, D_p, D_h, D_a, graph_hidden_size
'''
class GCNetModel(nn.Module):

    def __init__(self, args, base_encoder: nn.Module):
        
        super(GCNetModel, self).__init__()

        self.modal = args.modal
        self.no_cuda = args.device == "cpu"
        self.device = args.device
        self.loss_recon = args.loss_recon
        self.recon_loss_weight = args.recon_loss_weight
        self.reccls_flag = args.reccls_flag
        
        self.input_dim = sum([args.embedding_dim[m] for m in self.modal])
        self.D_e = args.hidden
        self.graph_hidden_size = args.hidden // 2

        # Definition for base encoder
        self.encoder = base_encoder(self.input_dim, self.D_e, dropout=args.dropout, bidirectional=args.bi_enc, batch_first=args.batch_first)
        self.emb_dim = self.encoder.output_dim

        ## Definition for graph model
        ## [modality_type=3(AVL); time_order=3(past, now, future)]
        self.n_speakers = args.dataset_speaker_dict[args.dataset]
        self.window_past = args.windowp
        self.window_future = args.windowf
        self.time_attn = args.time_attn

        ## gain graph models for 'temporal' and 'speaker'
        n_relations = 3
        self.graph_net_temporal = GraphNetwork(self.emb_dim, n_relations, self.time_attn, self.graph_hidden_size, args.dropout, self.no_cuda)
        n_relations = self.n_speakers ** 2
        self.graph_net_speaker = GraphNetwork(self.emb_dim, n_relations, self.time_attn, self.graph_hidden_size, args.dropout, self.no_cuda)

        ## classification and reconstruction
        D_h = self.emb_dim + self.graph_hidden_size
        self.smax_fc  = nn.Linear(D_h, args.output_dim)
        self.linear_rec = nn.Linear(D_h, self.input_dim)
        
        ## recon loss
        self.loss = MaskedMSELoss(args, reduction='mean')

    def forward(self):
        """
        if batch_first [batch, seqlen, dim] else [seqlen, batch, dim]
        inputs -> [?, ?, dim]
        qmask -> [?, ?]
        umask -> [?, ?]
        seq_lengths -> each conversation lens
        """
        outputs = self.encoder(self.inputs) # [?, ?, emb_dim]

        ## add graph model
        features, edge_index, edge_type, edge_type_mapping = batch_graphify(outputs, self.qmask, self.seq_lengths, self.n_speakers, 
                                                             self.window_past, self.window_future, 'temporal')
        assert len(edge_type_mapping) == 3
        hidden1 = self.graph_net_temporal(features, edge_index, edge_type, self.seq_lengths, self.umask)
        features, edge_index, edge_type, edge_type_mapping = batch_graphify(outputs, self.qmask, self.seq_lengths, self.n_speakers, 
                                                             self.window_past, self.window_future, 'speaker')
        assert len(edge_type_mapping) == self.n_speakers ** 2
        hidden2 = self.graph_net_speaker(features, edge_index, edge_type, self.seq_lengths, self.umask)
        hidden = hidden1 + hidden2

        ## for classification
        log_prob = self.smax_fc(hidden) # [?, ?, n_classes]

        ## for reconstruction
        rec_outputs = self.linear_rec(hidden)

        return log_prob, rec_outputs, hidden
        
    def set_input(self, inputs):
        self.target_input = [inputs[f'{m}_feat'] for m in self.modal]
        # self.target_input = torch.cat(self.target_input, dim=-1).to(self.device)  # [?, ?, sum(modal_dim)]
        
        self.inputs = [inputs[f'{m}_feat']*inputs[f'{m}_miss_index'].unsqueeze(-1) for m in self.modal]
        self.inputs = torch.cat(self.inputs, dim=-1).to(self.device)  # [?, ?, sum(modal_dim)]
        
        self.miss_index = [inputs[f'{m}_miss_index'] for m in self.modal]  # [?, ?, len(modal)]
        self.miss_index = torch.stack(self.miss_index, dim=-1).to(self.device)  # [?, ?, len(modal)]

        self.qmask = inputs['qmask']  # [?, ?]
        self.umask = inputs['umask']  # [?, ?]
        self.seq_lengths = inputs['seq_lengths']  # [batch]
        
    def get_outputs(self):
        if self.reccls_flag: # whether use reconstruction features 
            _, recon_input_features, _ = self.forward()
            assert recon_input_features.shape == self.inputs.shape
            self.inputs = recon_input_features 
            log_prob, _, hidden = self.forward()
        else:
            log_prob, recon_input_features, hidden = self.forward()
            
        return [log_prob, recon_input_features, hidden]
    
    def get_loss(self, outputs):
        """
        outputs: [logits, reconstructions, hidden_states]
        Get reconstruction loss
        """
        if not self.loss_recon:
            return {}
        
        recon = outputs[1] # [?, ?, dim]
        target = self.target_input  # list of [?, ?, dim]
        umask = self.umask

        miss_index = self.miss_index[self.umask.bool()] # [num_valid, len(modal)]

        loss_recon = []
        cum_dim = 0
        # Calculate reconstruction loss for each modality
        for idx, target_feat in enumerate(target):
            dim = target_feat.shape[-1]
            recon_feat = recon[:, :, cum_dim:cum_dim+dim]
            assert recon_feat.shape == target_feat.shape, f'recon: {recon_feat.shape}, target: {target_feat.shape}'
            modal_miss_index = miss_index[:,idx].unsqueeze(-1)
            # Calculate reconstruction loss on samples that misses current modality only
            loss = self.loss(recon_feat, target_feat, umask) * (1 - modal_miss_index)
            loss_recon.append(loss)
            cum_dim += dim
            
        loss_recon = torch.mean(torch.cat(loss_recon, dim=-1))
        
        return {"recon_loss": loss_recon * self.recon_loss_weight}
