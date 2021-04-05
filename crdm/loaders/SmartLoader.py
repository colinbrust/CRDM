from crdm.utils.ImportantVars import STACK_SHP, TRAIN_INDICES, TEST_INDICES, LENGTH, DIMS
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
import torch

dtype = torch.cuda.FloatTensor


class SmartLoader(Dataset):

    def __init__(self, feature_dir, const_dir, train=True, max_lead_time=12,
                 n_weeks=25, pixel=False, crop_size=16, feats=['*']):

        p = Path(feature_dir)
        ps = [list(p.glob(x+'.dat'))[0] for x in feats] if feats[0] != '*' else list(p.glob('*.dat'))

        self.shp = STACK_SHP if pixel else (STACK_SHP[0], *DIMS)
        self.targets = np.memmap(str(list(p.glob('USDM.dat'))[0]), dtype='float32', shape=self.shp)
        self.features = [np.memmap(x, dtype='float32', shape=self.shp) for x in ps]
        self.indices = TRAIN_INDICES if train else TEST_INDICES
        self.max_lead_time = max_lead_time
        self.n_weeks = n_weeks
        self.pixel = pixel
        self.crop_size = crop_size
        self.const_dir = const_dir

        self.complete_ts = [list(range(x, x+max_lead_time)) for x in range(1, len(self.targets))]
        self.complete_ts = [x for x in self.complete_ts if all(y in self.indices for y in x)]
        self.complete_ts = [x for x in self.complete_ts if x[0] >= self.n_weeks]
        self.complete_ts = self.complete_ts * 512 

        self.consts = self._make_constants()

    def _make_constants(self):

        shp = LENGTH if self.pixel else DIMS
        consts = [str(x) for x in list(Path(self.const_dir).iterdir())]
        consts = [np.memmap(x, dtype='float32', shape=shp) for x in consts]
        consts = np.array(consts)

        return consts

    def pixel_loader(self, idx):
        idx_list = self.complete_ts[idx]
        feature_end = idx_list[0]
        feature_start = feature_end - self.n_weeks
        feature_range = list(range(max(0, feature_start), feature_end))

        pixel = np.random.randint(0, LENGTH, 1)

        feats = []
        for feature in self.features:
            feats.append(np.take(feature[feature_range], pixel, axis=-1))

        feats = np.array(feats)
        targets = np.take(self.targets[idx_list], pixel, axis=-1)
        consts = np.take(self.consts, pixel, axis=-1)

        return dtype(feats), dtype(consts), dtype(targets)

    def crop_loader(self, idx):
        idx_list = self.complete_ts[idx]
        feature_end = idx_list[0]
        feature_start = feature_end - self.n_weeks
        feature_range = list(range(max(0, feature_start), feature_end))

        x = np.random.randint(0, DIMS[1] - self.crop_size)
        y = np.random.randint(0, DIMS[0] - self.crop_size)

        feats = []
        for feature in self.features:
            feats.append(feature[feature_range, y:y+self.crop_size, x:x+self.crop_size])

        feats = np.array(feats)
        targets = self.targets[idx_list, y:y+self.crop_size, x:x+self.crop_size]
        consts = self.consts[:, y:y+self.crop_size, x:x+self.crop_size]

        return feats, consts, targets

    def __len__(self):
        return len(self.complete_ts)

    def __getitem__(self, idx):
        
        if self.pixel:
            return self.pixel_loader(idx)
        else:
            arr, consts, targets = self.crop_loader(idx)
            arr = np.transpose(arr, (1, 0, 2, 3))
            return dtype(arr), dtype(consts), dtype(targets)




