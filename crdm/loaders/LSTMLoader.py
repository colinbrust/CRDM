import numpy as np
import os
import pickle
import torch
from torch.utils.data import Dataset


class LSTMLoader(Dataset):

    def __init__(self, dirname, train=True, categorical=False, n_weeks=30, sample=None, mx_lead=12):

        print('Train: {}\nCategorical: {}\nWeek History: {}'.format(train, categorical, n_weeks))

        with open(os.path.join(dirname, 'shps.p'), 'rb') as f:
            shps = pickle.load(f)

        x = 'train_x.dat' if train else 'test_x.dat'
        y = 'train_y.dat' if train else 'test_y.dat'

        self.categorical = categorical
        self.mx_lead = mx_lead
        self.sample = sample

        self.xtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        if categorical:
            self.ytype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
        else:
            self.ytype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        self.x = np.memmap(os.path.join(dirname, x), dtype='float32', shape=shps[x], mode='r')
        self.x = self.x[..., -n_weeks:]
        self.y = np.memmap(os.path.join(dirname, y), dtype='float32', shape=shps[y], mode='r')

        # Batch, seq, feature
        self.x = self.x.swapaxes(1, 2)

        # if sample is not None:
        #     samps = np.random.randint(0, len(self.x), sample)
        #     self.x = self.x[samps]
        #     self.y = self.y[samps]

    def __len__(self):
        return len(self.x) if self.sample is not None else self.sample

    def __getitem__(self, idx):

        idx = np.random.randint(0, len(self.x), 1) if self.sample else idx
        x = self.x[idx]
        y = self.y[idx, :self.mx_lead]

        y = y*5 if self.categorical else y

        # Copy so we only have to copy one slice into memory rather than the entire train/test dataset.
        return self.xtype(x.copy()), self.ytype(y.copy())
