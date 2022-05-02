import torch
import numpy as np
from torch.utils.data.dataloader import default_collate

default_collate = default_collate

def pad_sequence(seq, max_len, padding_value = 0):
    try:
        seq_len, col = seq.shape
        padding = np.zeros((max_len - seq_len, col)) + padding_value
    except:
        seq_len = seq.shape[0]
        padding = np.zeros((max_len - seq_len, )) + padding_value

    padding_seq = np.concatenate([padding, seq])

    return padding_seq

def transformer_collate(samples):
    max_len = 0
    for sample in samples:
        seq_len, col = sample['past_cat_feature'].shape
        if max_len < seq_len:
            max_len = seq_len
    
    past_cat_feature = []
    past_num_feature = []
    past_answerCode = []
    now_cat_feature = []
    now_num_feature = []
    now_answerCode = []

    for sample in samples:
        past_cat_feature += [pad_sequence(sample['past_cat_feature'] + 1, max_len = max_len, padding_value = 0)]
        past_num_feature += [pad_sequence(sample['past_num_feature'], max_len = max_len, padding_value = 0)]
        past_answerCode += [pad_sequence(sample['past_answerCode'] + 1, max_len = max_len, padding_value = 0)]
        now_cat_feature += [pad_sequence(sample['now_cat_feature'] + 1, max_len = max_len, padding_value = 0)]
        now_num_feature += [pad_sequence(sample['now_num_feature'], max_len = max_len, padding_value = 0)]
        now_answerCode += [pad_sequence(sample['now_answerCode'], max_len = max_len, padding_value = -1)]

    return {
        'past_cat_feature' : torch.tensor(past_cat_feature, dtype = torch.long),
        'past_num_feature' : torch.tensor(past_num_feature, dtype = torch.float32), 
        'past_answerCode' : torch.tensor(past_answerCode, dtype = torch.long), 
        'now_cat_feature' : torch.tensor(now_cat_feature, dtype = torch.long), 
        'now_num_feature' : torch.tensor(now_num_feature, dtype = torch.float32), 
        'now_answerCode' : torch.tensor(now_answerCode, dtype = torch.float32),
        }
