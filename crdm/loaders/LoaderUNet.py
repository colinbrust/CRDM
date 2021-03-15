from crdm.loaders.ReadUNet import AggregateAllSpatial
from crdm.utils.ImportantVars import DIMS
import glob
from itertools import cycle
import numpy as np
import os
import torch
from torch.utils.data import Dataset


class CroppedLoader(Dataset):

    def __init__(self, target_dir, in_features, n_weeks=5, crop_size=32, batch_size=256, test=False, cuda=True):

        self.targets = glob.glob(os.path.join(target_dir, '*.dat'))
        if test:
            self.targets = [x for x in self.targets if ('/2007' in x or '/2015' in x or '/2017' in x)]
        else:
            self.targets = [x for x in self.targets if not ('/2007' in x or '/2015' in x or '/2017' in x)]
        # self.targets = self.targets[90:105]
        pix_list = []

        for target in self.targets:
            for lead_time in [2, 4, 6, 8]:
                try:
                    print('{} for {} week lead time'.format(target, lead_time))
                    pixels = AggregateAllSpatial(target, in_features, lead_time=lead_time, n_weeks=n_weeks, memmap=True)
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
        self.crop_size = crop_size
        self.batch_size = batch_size

    def __len__(self):
        return self.size * self.batch_size

    def __getitem__(self, idx):

        pixels = next(self.pix_list)
        features = pixels.premake_features()
        n_features, size = features.shape
        features = features.reshape(n_features, *DIMS)
        out = []
        target_out = []

        target = np.memmap(pixels.target, dtype='int8', shape=DIMS, mode='c')

        for i in range(self.batch_size):
            x = np.random.randint(0, DIMS[1] - 32)
            y = np.random.randint(0, DIMS[0] - 32)

            tmp = features[:, y:y + self.crop_size, x:x + self.crop_size]
            target_tmp = target[y:y + self.crop_size, x:x + self.crop_size]

            out.append(tmp)
            target_out.append(target_tmp)

        features = np.array(out)

        targets = np.array(target_out)

        dt = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor

        return dt(features), dt(targets/5)
