import os
import pickle
import math
import torch
import random
from data.utils import word2utt_data
from data.prepare_idt import generate_modal_mask


def load_data(data_dir, dataset, modal_MR, ratio=0.1):
    try:
        data_path = os.path.join(data_dir, f'{dataset}_utt.pkl')
        data_raw = pickle.load(open(data_path, 'rb'))
    except:
        data_path = os.path.join(data_dir, f'{dataset}_word.pkl')
        data_raw = pickle.load(open(data_path, 'rb'))
        data_raw = word2utt_data(data_raw)
        pickle.dump(data_raw, open(data_path.replace('_word', '_utt'), 'wb'))

    data = {set_name: [] for set_name in data_raw}
    for set_name in data_raw:
        lengths = [len(data_raw[set_name][vid]) for vid in data_raw[set_name]]
        missing_idx = generate_modal_mask(modal_MR, lengths, list(data_raw[set_name].keys()))
        for vid in data_raw[set_name]:
            sample = {
                'A_feat': [utt['A_feat'] for utt in data_raw[set_name][vid]],
                'L_feat': [utt['L_feat'] for utt in data_raw[set_name][vid]],
                'V_feat': [utt['V_feat'] for utt in data_raw[set_name][vid]],
                'speaker': [utt['speaker'] for utt in data_raw[set_name][vid]],
                'uid': [utt['uid'] for utt in data_raw[set_name][vid]],
                'A_miss_index': missing_idx['A'][vid],
                'L_miss_index': missing_idx['L'][vid],
                'V_miss_index': missing_idx['V'][vid],
                'vid': vid,
                'cls_label': [utt['cls_label'] for utt in data_raw[set_name][vid]],
                'reg_label': [utt['reg_label'] for utt in data_raw[set_name][vid]]}
            data[set_name].append(sample)
    random.shuffle(data['train'])
    if 'valid' not in data:
        valid_size = int(ratio * len(data['train']))
        data['valid'] = data['train'][:valid_size]
        data['train'] = data['train'][valid_size:]
        
    embedding_dim = {
        'A': data['train'][0]['A_feat'][0].shape[-1],
        'L': data['train'][0]['L_feat'][0].shape[-1],
        'V': data['train'][0]['V_feat'][0].shape[-1]
    }

    return data, embedding_dim


class Dataloader:
    """Dataloader for Conversational Dataset
    batch: batch of conversations
    seq_lengths: lengths of each conversation 
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
        mx = max([len(s['A_feat']) for s in samples])

        A_feat_tensor = torch.zeros((batch_size, mx, self.embedding_dim['A']))
        L_feat_tensor = torch.zeros((batch_size, mx, self.embedding_dim['L']))
        V_feat_tensor = torch.zeros((batch_size, mx, self.embedding_dim['V']))
        A_miss_index = torch.zeros((batch_size, mx)).long()
        L_miss_index = torch.zeros((batch_size, mx)).long()
        V_miss_index = torch.zeros((batch_size, mx)).long()
        qmask = torch.zeros((batch_size, mx)).long()
        umask = torch.zeros((batch_size, mx)).long()
        target = torch.zeros((batch_size, mx))
        seq_lengths = []

        for i, s in enumerate(samples):
            cur_len = len(s["A_feat"])

            tmp_A = torch.stack([torch.tensor(a) for a in s["A_feat"]])
            tmp_L = torch.stack([torch.tensor(l) for l in s["L_feat"]])
            tmp_V = torch.stack([torch.tensor(v) for v in s["V_feat"]])
            A_feat_tensor[i, :cur_len, :] = tmp_A
            L_feat_tensor[i, :cur_len, :] = tmp_L
            V_feat_tensor[i, :cur_len, :] = tmp_V

            A_miss_index[i, :cur_len] = torch.tensor(s["A_miss_index"]).long()
            L_miss_index[i, :cur_len] = torch.tensor(s["L_miss_index"]).long()
            V_miss_index[i, :cur_len] = torch.tensor(s["V_miss_index"]).long()

            qmask[i, :cur_len] = torch.tensor(s["speaker"])
            umask[i, :cur_len] = torch.ones(cur_len).long()

            if self.regression:
                label = [float(x) for x in s['reg_label']]
                target[i, :cur_len] = torch.tensor(label).float()
            else:
                label = [int(x) for x in s['cls_label']]
                target[i, :cur_len] = torch.tensor(label).long()
                
            seq_lengths.append(cur_len)

        data = {
            'A_feat': A_feat_tensor,
            'L_feat': L_feat_tensor,
            'V_feat': V_feat_tensor,
            'A_miss_index': A_miss_index,
            'L_miss_index': L_miss_index,
            'V_miss_index': V_miss_index,
            'qmask': qmask,
            'umask': umask,
            'seq_lengths': seq_lengths,
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