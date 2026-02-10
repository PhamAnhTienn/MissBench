import os
import pickle
import math
import torch
import random
from data.prepare_idt import generate_modal_mask

def load_data(data_dir, dataset, modal_MR, ratio=0.1):
    data_path = os.path.join(data_dir, f'{dataset}_word.pkl')
    data_raw = pickle.load(open(data_path, 'rb'))

    for set_name in data_raw:
        lengths = [len(data_raw[set_name][vid]) for vid in data_raw[set_name]]
        missing_idx = generate_modal_mask(modal_MR, lengths, list(data_raw[set_name].keys()))
        for vid in data_raw[set_name]:
            for idx in range(len(data_raw[set_name][vid])):
                data_raw[set_name][vid][idx]['A_miss_index'] = missing_idx['A'][vid][idx]
                data_raw[set_name][vid][idx]['L_miss_index'] = missing_idx['L'][vid][idx]
                data_raw[set_name][vid][idx]['V_miss_index'] = missing_idx['V'][vid][idx]
                
    data = {set_name: [] for set_name in data_raw}
    for set_name in data_raw:
        for vid in data_raw[set_name]:
            data[set_name].append(data_raw[set_name][vid])
    random.shuffle(data['train'])
    if 'valid' not in data:
        valid_size = int(ratio * len(data['train']))
        data['valid'] = data['train'][:valid_size]
        data['train'] = data['train'][valid_size:]
    
    # Con to Non-Conversion: flatten the data
    for set_name in data:
        data[set_name] = [utt for vid in data[set_name] for utt in vid]
        
    embedding_dim = {
        'A': data['train'][0]['A_feat'].shape[-1],
        'L': data['train'][0]['L_feat'].shape[-1],
        'V': data['train'][0]['V_feat'].shape[-1]
    }

    return data, embedding_dim

class Dataloader:
    """Dataloader for Non-Conversational Dataset
    batch: batch of utterances
    A_lengths: audio feature length(over time) for every sample
    L_lengths: lexical feature length(over time) for every sample
    V_lengths: visual feature length(over time) for every sample
    umask: mask for valid utterance positions (all 1s since no padding between utterances)
    qmask: mask for speaker identities (no padding between utterances)
    """

    def __init__(self, data, batch_size, embedding_dim, regression=False, batch_first=False, device='cuda'):
        super(Dataloader, self).__init__()
        self.data = data
        self.regression = regression
        self.batch_size = batch_size
        self.num_batches = math.ceil(len(data) / self.batch_size)
        self.embedding_dim = embedding_dim
        self.batch_first = batch_first
        self.device = device

    def __len__(self):
        return self.num_batches

    def __getitem__(self, index):
        batch = self.raw_batch(index)
        return self.padding(batch)

    def raw_batch(self, index):
        assert index < self.num_batches, "batch_idx %d > %d" % (
            index, self.num_batches)
        batch = self.data[index *
                          self.batch_size: (index + 1) * self.batch_size]
        return batch

    def padding(self, samples):
        
        batch_size = len(samples)
        A_mx = max([len(s['A_feat']) for s in samples])
        L_mx = max([len(s['L_feat']) for s in samples])
        V_mx = max([len(s['V_feat']) for s in samples])
        
        A_feat_tensor = torch.zeros((batch_size, A_mx, self.embedding_dim['A']))
        L_feat_tensor = torch.zeros((batch_size, L_mx, self.embedding_dim['L']))
        V_feat_tensor = torch.zeros((batch_size, V_mx, self.embedding_dim['V']))
        
        A_miss_index = torch.tensor([s['A_miss_index'] for s in samples]).long()
        L_miss_index = torch.tensor([s['L_miss_index'] for s in samples]).long()
        V_miss_index = torch.tensor([s['V_miss_index'] for s in samples]).long()
        qmask = torch.tensor([[s['speaker']] for s in samples]).long() # no padding
        umask = torch.ones((batch_size, 1)).long()
        if self.regression:
            label = [float(s['reg_label']) for s in samples]
            target = torch.tensor(label).float()
        else:
            label = [int(s['cls_label']) for s in samples]
            target = torch.tensor(label).long()
        
        for i, s in enumerate(samples):
            len_A = len(s["A_feat"])
            len_V = len(s["V_feat"])
            len_L = len(s["L_feat"])
            
            tmp_A = torch.tensor(s["A_feat"])
            tmp_L = torch.tensor(s["L_feat"])
            tmp_V = torch.tensor(s["V_feat"])
            A_feat_tensor[i, :len_A, :] = tmp_A
            L_feat_tensor[i, :len_L, :] = tmp_L
            V_feat_tensor[i, :len_V, :] = tmp_V

        data = {
            'A_feat': A_feat_tensor,
            'L_feat': L_feat_tensor,
            'V_feat': V_feat_tensor,
            'A_miss_index': A_miss_index,
            'L_miss_index': L_miss_index,
            'V_miss_index': V_miss_index,
            'qmask': qmask,
            'umask': umask,
            'target': target
        }
        
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                data[k] = v.to(self.device)
        
        if not self.batch_first:
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    data[k] = v.transpose(0, 1).contiguous()  # [seq_len, batch, ...]

        return data

    def shuffle(self):
        random.shuffle(self.data)