from crdm.utils.ParseFileNames import parse_fname
from torch.utils.data import Dataset
import numpy as np
import torch

class PixelLoader(Dataset):
    
    def __init__(self, constant, monthly, target):

        info = parse_fname(constant)

        const_shape = np.memmap(constant, dtype='float32', mode='r').size
        num_consts = 16 if info['rmFeatures'] == 'True' else 18
        num_samples = int(const_shape/num_consts)

        self.constant = np.memmap(constant, dtype='float32', mode='c', shape=(num_samples, num_consts))
        self.constant = np.nan_to_num(self.constant, nan=-0.5)

        self.monthly = np.memmap(monthly, dtype='float32', mode='c', shape=(num_samples, int(info['nMonths']), 12))
        self.monthly = np.nan_to_num(self.monthly, nan=-0.5)

        self.target = np.memmap(target, dtype='int8', mode='c')
    
    def __len__(self):
        return len(self.target)
    
    def __getitem__(self, idx):
        return {'const': torch.tensor(self.constant[idx]), 
                'mon': torch.tensor(self.monthly[idx]),
                'target': self.target[idx]}
