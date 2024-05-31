import torch
import pandas as pd

from typing import Callable
from torch.utils.data import Dataset

class MQMDataset(Dataset):
    def __init__(self, path: str):
        self.path = path
        self.data = pd.read_csv(path)
    
    def filter(self, predicate: Callable):
        self.data = self.data[self.data.apply(predicate, axis=1)]

    def __getitem__(self, index):
        return self.data.iloc[index].to_dict()
    
    def __len__(self):
        return len(self.data)