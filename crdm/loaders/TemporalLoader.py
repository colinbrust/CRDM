from crdm.utils.ImportantVars import STACK_SHP, TRAIN_INDICES, TEST_INDICES, LENGTH
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
import torch

dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


class DroughtLoader(Dataset):

    def __init__(self, feature_dir, const_dir, train=True, max_lead_time=12, n_weeks=25, feats=('pr', 'USDM')):

        p = Path(feature_dir)
        ps = [list(p.glob(x+'.dat'))[0] for x in feats] if feats[0] != '*' else list(p.glob('*.dat'))

        self.shp = STACK_SHP
        self.targets = np.memmap(str(list(p.glob('USDM.dat'))[0]), dtype='float32', shape=self.shp)
        self.features = [np.memmap(str(x), dtype='float32', shape=self.shp) for x in ps]
        self.max_lead_time = max_lead_time
        self.n_weeks = n_weeks
        self.const_dir = const_dir
        self.consts = self._make_constants()

        self.indices = TRAIN_INDICES if train else TEST_INDICES
        self.complete_ts = [list(range(x, x+max_lead_time)) for x in range(1, len(self.targets))]
        self.complete_ts = [x for x in self.complete_ts if all(y in self.indices for y in x)]
        self.complete_ts = [x for x in self.complete_ts if x[0] >= self.n_weeks]

    def _make_constants(self):

        consts = [str(x) for x in list(Path(self.const_dir).iterdir())]
        consts = [np.memmap(x, dtype='float32', shape=LENGTH) for x in consts]
        consts = np.array(consts)

        return consts

    def __len__(self):
        return len(self.complete_ts)

    def __getitem__(self, idx):

        idx_list = self.complete_ts[idx]
        feature_range = list(range(idx_list[0] - self.n_weeks, idx_list[0]))

        pixel = np.random.randint(0, LENGTH, 1)

        feats = []
        for feature in self.features:
            feats.append(np.take(feature[feature_range], pixel, axis=-1))

        feats = np.array(feats)
        feats = feats.squeeze()
        targets = np.take(self.targets[idx_list], pixel, axis=-1)
        consts = np.take(self.consts, pixel, axis=-1)

        return dtype(feats), dtype(consts), dtype(targets)





