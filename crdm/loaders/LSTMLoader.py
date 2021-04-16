import numpy as np
import os
import pickle
import torch
from torch.utils.data import Dataset

dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


class LSTMLoader(Dataset):

    def __init__(self, dirname, train=True):

        with open(os.path.join(dirname, 'shps.p'), 'rb') as f:
            shps = pickle.load(f)

        x = 'train_x.dat' if train else 'test_x.dat'
        y = 'train_y.dat' if train else 'test_y.dat'

        self.x = np.memmap(os.path.join(dirname, x), dtype='float32', shape=shps[x], mode='r')
        self.y = np.memmap(os.path.join(dirname, y), dtype='float32', shape=shps[y], mode='r')

        # Batch, seq, feature
        self.x = self.x.swapaxes(1, 2)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):

        x = self.x[idx]
        y = self.y[idx]

        # Copy so we only have to copy one slice into memory rather than the entire train/test dataset.
        return dtype(x.copy()), dtype(y.copy())
