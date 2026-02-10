def word2utt_feature(x, norm_method='mean'):
    """
        x [feature_len, feature_dim] : np.ndarray -> [feature_dim]
    """
    if norm_method == 'first':
        return x[0, :]
    if norm_method == 'mean':
        return x.mean(axis=0)
    if norm_method == 'max':
        return x.max(axis=0)
    if norm_method == 'min':
        return x.min(axis=0)
    if norm_method == 'sum':
        return x.sum(axis=0)
    
def word2utt_data(data):
    """
        data: dict of conversations
    """
    for split in data:
        for vid in data[split]:
            for utt in range(len(data[split][vid])):
                A_feat = data[split][vid][utt]['A_feat']
                V_feat = data[split][vid][utt]['V_feat']
                L_feat = data[split][vid][utt]['L_feat']
                
                data[split][vid][utt]['A_feat'] = word2utt_feature(A_feat, norm_method='mean')
                data[split][vid][utt]['V_feat'] = word2utt_feature(V_feat, norm_method='mean')
                data[split][vid][utt]['L_feat'] = word2utt_feature(L_feat, norm_method='mean')
    return data