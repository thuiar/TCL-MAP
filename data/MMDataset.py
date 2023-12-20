from torch.utils.data import Dataset
import torch
import numpy as np

__all__ = ['MMDataset']


class MMDataset(Dataset):
        
    def __init__(self, label_ids, text_feats, video_feats, audio_feats, cons_text_feats, condition_idx):
        
        
        self.label_ids = label_ids
        self.text_feats = text_feats
        self.cons_text_feats = cons_text_feats
        self.condition_idx = condition_idx
        self.video_feats = video_feats
        self.audio_feats = audio_feats
        self.size = len(self.text_feats)

    def __len__(self):
        return self.size

    def __getitem__(self, index):

        sample = {
            'label_ids': torch.tensor(self.label_ids[index]), 
            'text_feats': torch.tensor(self.text_feats[index]),
            'video_feats': torch.tensor(self.video_feats['feats'][index]),
            'audio_feats': torch.tensor(self.audio_feats['feats'][index]),
            'cons_text_feats': torch.tensor(self.cons_text_feats[index]),
            'condition_idx': torch.tensor(self.condition_idx[index])
        } 
        return sample


