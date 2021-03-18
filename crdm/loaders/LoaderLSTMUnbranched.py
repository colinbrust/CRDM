from crdm.loaders.ReadConvLSTM import AggregateAllSpatial
from crdm.utils.ImportantVars import DIMS
import glob
from itertools import cycle
import numpy as np
import os
import torch
from torch.utils.data import Dataset


class PixelLoader(Dataset):

    def __init__(self, target_dir, in_features, n_weeks=5, batch_size=256, test=False, cuda=True):

        self.targets = glob.glob(os.path.join(target_dir, '*.dat'))
        if test:
            self.targets = [x for x in self.targets if ('/2007' in x or '/2015' in x or '/2017' in x)]
        else:
            self.targets = [x for x in self.targets if not ('/2007' in x or '/2015' in x or '/2017' in x)]
        # self.targets = self.targets[95:100]
        pix_list = []

        for target in self.targets:
            for lead_time in [2, 4, 6, 8]:
                try:
                    print('{} for {} week lead time'.format(target, lead_time))
                    pixels = AggregateAllSpatial(target, in_features, lead_time=lead_time, n_weeks=25, memmap=True)
                    pix_list.append(pixels)
                except AssertionError as e:
                    print('{} - Skipping {} for {} week lead time.'.format(e, target, lead_time))

        self.pix_list = pix_list
        np.random.shuffle(self.pix_list)

        self.size = len(self.pix_list)

        self.pix_list = cycle(self.pix_list)
        self.cuda = cuda
        self.in_features = in_features
        self.n_weeks = n_weeks
        self.batch_size = batch_size
        self.dt = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor

    def __len__(self):
        return self.size * self.batch_size

    def __getitem__(self, idx):

        pixels = next(self.pix_list)
        features = pixels.premake_features()
        n_features, n_weeks, size = features.shape

        target = np.memmap(pixels.target, dtype='int8', mode='c')
        indices = np.random.choice(size, size=self.batch_size, replace=False)

        features = np.take(features, indices, axis=-1)
        features = features.swapaxes(1, 2)
        features = features.swapaxes(0, 2)

        target = target[indices]/5

        return self.dt(features), self.dt(target)
