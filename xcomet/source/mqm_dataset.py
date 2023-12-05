import torch
from torch.utils.data import Dataset

class MQMDataset(Dataset):
    def __init__(self, ):
        pass
    
    def __getitem__(self, index):
        ...
        return tokenized_sequence, per_token_tags, mqm_score
    
    def __len__(self):
        pass


def collate_fn(batch):
    pass