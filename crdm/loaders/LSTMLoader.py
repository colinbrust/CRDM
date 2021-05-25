import numpy as np
import os
import pickle
import torch
from torch.utils.data import Dataset


class LSTMLoader(Dataset):

    def __init__(self, dirname, train=True, categorical=False, n_weeks=30, sample=None, mx_lead=12, even_sample=False):

        print('Train: {}\nCategorical: {}\nWeek History: {}'.format(train, categorical, n_weeks))

        with open(os.path.join(dirname, 'shps.p'), 'rb') as f:
            shps = pickle.load(f)

        x = 'train_x.dat' if train else 'test_x.dat'
        y = 'train_y.dat' if train else 'test_y.dat'

        self.categorical = categorical
        self.mx_lead = mx_lead
        self.sample = sample
        self.even_sample = even_sample

        self.xtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        if categorical:
            self.ytype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
        else:
            self.ytype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        self.x = np.memmap(os.path.join(dirname, x), dtype='float32', shape=shps[x], mode='r')
        self.x = self.x[..., -n_weeks:]
        # Batch, seq, feature
        self.x = self.x.swapaxes(1, 2)

        self.y = np.memmap(os.path.join(dirname, y), dtype='float32', shape=shps[y], mode='r')

        if self.even_sample:
            print('Evenly Sampling')
            yr = (self.y.ravel() * 5).astype(np.int)
            p = np.zeros_like(yr).astype(np.float64)
            c = np.bincount(yr)
            weights = (1/c)/6

            for i in range(6):
                temp = np.equal(yr, i)
                np.putmask(p, temp, weights[i])

            self.p = np.sum(p.reshape(self.y.shape), axis=1)

            idx = np.random.choice(range(len(self.x)), int(len(self.x)/2), p=self.p, replace=True)
            self.x = self.x[idx]
            self.y = self.y[idx]

    def __len__(self):
        return self.sample if self.sample is not None else len(self.x)

    def __getitem__(self, idx):

        idx = np.random.randint(0, len(self.x), 1) if self.sample else idx

        x = self.x[idx].squeeze()
        y = self.y[idx, :self.mx_lead].squeeze()
        
        y = y*5 if self.categorical else y

        # Copy so we only have to copy one slice into memory rather than the entire train/test dataset.
        return self.xtype(x.copy()), self.ytype(y.copy())
