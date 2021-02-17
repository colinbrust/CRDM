from crdm.utils.ParseFileNames import parse_fname
from crdm.utils.ImportantVars import WEEKLY_VARS, MONTHLY_VARS
from torch.utils.data import Dataset
import numpy as np
import torch


class PixelLoader(Dataset):
    
    def __init__(self, constant, weekly, monthly, target, init):

        info = parse_fname(constant)

        const_shape = np.memmap(constant, dtype='float32', mode='r').size
        num_consts = 11
        num_samples = int(const_shape/num_consts)
        nMonths = int(info['nWeeks'])//4

        weekly_size = len(WEEKLY_VARS) + 1 if init == 'True' else len(WEEKLY_VARS)

        self.constant = np.memmap(constant, dtype='float32', mode='c', shape=(num_samples, num_consts))

        self.weekly = np.memmap(weekly, dtype='float32', mode='c', shape=(num_samples, int(info['nWeeks']), weekly_size))

        self.monthly = np.memmap(monthly, dtype='float32', mode='c', shape=(num_samples, nMonths, len(MONTHLY_VARS)))

        self.target = np.memmap(target, dtype='float32', mode='c', shape=(num_samples, 8))
    
    def __len__(self):
        return len(self.target)
    
    def __getitem__(self, idx):

        # self.monthly = np.nan_to_num(self.monthly, =-0.5)
        return {'const': torch.tensor(self.constant[idx]),
                'mon': torch.tensor(self.monthly[idx]),
                'week': torch.tensor(self.weekly[idx]),
                'target': torch.tensor(self.target[idx])}
