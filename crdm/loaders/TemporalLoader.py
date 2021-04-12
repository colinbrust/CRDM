from crdm.utils.ImportantVars import STACK_SHP, TRAIN_INDICES, TEST_INDICES, LENGTH
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
import torch
from itertools import cycle

dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


class DroughtLoader(Dataset):

    def __init__(self, feature_dir, const_dir, train=True, max_lead_time=12, n_weeks=25, batch_size=1024, feats=('pr', 'USDM')):

        p = Path(feature_dir)
        ps = [list(p.glob(x+'.dat'))[0] for x in feats] if feats[0] != '*' else list(p.glob('*.dat'))

        self.shp = STACK_SHP
        self.targets = np.memmap(str(list(p.glob('USDM.dat'))[0]), dtype='float32', shape=self.shp)
        self.features = [np.memmap(str(x), dtype='float32', shape=self.shp) for x in ps]
        self.max_lead_time = max_lead_time
        self.n_weeks = n_weeks
        self.const_dir = const_dir
        self.consts = self._make_constants()
        self.batch_size = batch_size

        self.indices = TRAIN_INDICES if train else TEST_INDICES
        self.complete_ts = [list(range(x, x+max_lead_time)) for x in range(1, len(self.targets))]
        self.complete_ts = [x for x in self.complete_ts if all(y in self.indices for y in x)]
        self.complete_ts = [x for x in self.complete_ts if x[0] >= self.n_weeks]
        self.sz = len(self.complete_ts)
        np.random.shuffle(self.complete_ts)
        self.complete_ts = cycle(self.complete_ts)

    def _make_constants(self):

        consts = [str(x) for x in list(Path(self.const_dir).iterdir())]
        consts = [np.memmap(x, dtype='float32', shape=LENGTH) for x in consts]
        consts = np.array(consts)

        return consts

    def __len__(self):
        return self.sz * self.batch_size

    def __getitem__(self, idx):

        idx_list = next(self.complete_ts)
        feature_range = list(range(idx_list[0] - self.n_weeks, idx_list[0]))

        pixels = np.random.randint(0, LENGTH, self.batch_size)

        feats = []
        for feature in self.features:
            feats.append(np.take(feature[feature_range], pixels, axis=-1))

        feats = np.array(feats)
        feats = np.swapaxes(feats, 0, 2)

        targets = np.take(self.targets[idx_list], pixels, axis=-1)
        targets = np.swapaxes(targets, 0, 1)

        consts = np.take(self.consts, pixels, axis=-1)
        consts = np.swapaxes(consts, 0, 1)

        return dtype(feats), dtype(consts), dtype(targets)





