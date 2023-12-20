import pickle
import numpy as np
import os


def get_v_a_data(data_args, feats_path):
    
    if not os.path.exists(feats_path):
        raise Exception('Error: The directory of features is empty.')    

    feats = load_feats(data_args, feats_path)
    data = padding_feats(data_args, feats)
    
    return data 
    
def load_feats(data_args, video_feats_path):

    with open(video_feats_path, 'rb') as f:
        video_feats = pickle.load(f)

    train_feats = [np.array(video_feats[x]) for x in data_args['train_data_index']]
    dev_feats = [np.array(video_feats[x]) for x in data_args['dev_data_index']]
    test_feats = [np.array(video_feats[x]) for x in data_args['test_data_index']]
    
    
    outputs = {
        'train': train_feats,
        'dev': dev_feats,
        'test': test_feats
    }

    return outputs

def padding(feat, max_length, padding_mode = 'zero', padding_loc = 'end'):
    """
    padding_mode: 'zero' or 'normal'
    padding_loc: 'start' or 'end'
    """
    assert padding_mode in ['zero', 'normal']
    assert padding_loc in ['start', 'end']

    length = feat.shape[0]
    if length > max_length:
        return feat[:max_length, :]

    if padding_mode == 'zero':
        pad = np.zeros([max_length - length, feat.shape[-1]])
    elif padding_mode == 'normal':
        mean, std = feat.mean(), feat.std()
        pad = np.random.normal(mean, std, (max_length - length, feat.shape[1]))
    
    if padding_loc == 'start':
        feat = np.concatenate((pad, feat), axis = 0)
    else:
        feat = np.concatenate((feat, pad), axis = 0)

    return feat

def padding_feats(data_args, feats):
    
    max_seq_len = data_args['max_seq_len']

    p_feats = {}

    for dataset_type in feats.keys():
        f = feats[dataset_type]

        tmp_list = []
        length_list = []
        
        for x in f:
            x_f = np.array(x) 
            x_f = x_f.squeeze(1) if x_f.ndim == 3 else x_f

            length_list.append(len(x_f))
            p_feat = padding(x_f, max_seq_len)
            tmp_list.append(p_feat)

        p_feats[dataset_type] = {
            'feats': tmp_list,
            'lengths': length_list
        }

    return p_feats    
