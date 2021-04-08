from crdm.utils.ImportantVars import STACK_SHP, TRAIN_INDICES, TEST_INDICES, LENGTH, DIMS
import numpy as np
from pathlib import Path
from scipy.ndimage import rotate
import pickle
from torch.utils.data import Dataset
import torch
import os

dtype = torch.cuda.FloatTensor


class DroughtLoader(Dataset):

    def __init__(self, feature_dir, const_dir, train_dir, train=True, max_lead_time=12,
                 n_weeks=25, pixel=False, crop_size=16, feats=('pr', 'USDM'), spatial=False):

        p = Path(feature_dir)
        ps = [list(p.glob(x+'.dat'))[0] for x in feats] if feats[0] != '*' else list(p.glob('*.dat'))

        self.shp = STACK_SHP if pixel else (STACK_SHP[0], *DIMS)
        self.targets = np.memmap(str(list(p.glob('USDM.dat'))[0]), dtype='float32', shape=self.shp)
        self.features = [np.memmap(str(x), dtype='float32', shape=self.shp) for x in ps]
        self.max_lead_time = max_lead_time
        self.n_weeks = n_weeks
        self.pixel = pixel
        self.crop_size = crop_size
        self.const_dir = const_dir
        self.transform = True if train else False
        self.spatial = spatial
        self.consts = self._make_constants()

        self.indices = self._read_pickle(os.path.join(train_dir, 'train.p')) if train else self._read_pickle(os.path.join(train_dir, 'test.p'))
        self.indices = np.array(self.indices)
        
        self.complete_ts = [list(range(x, x+max_lead_time)) for x in range(1, len(self.targets))]
        self.complete_ts = [x for x in self.complete_ts if all(y in TRAIN_INDICES+TEST_INDICES for y in x)]
        self.complete_ts = [x for x in self.complete_ts if x[0] >= self.n_weeks]
        self.complete_ts = [(x, self.indices[:, y]) for x in self.complete_ts for y in range(self.indices.shape[1])]
        np.random.shuffle(self.complete_ts)

    def _make_constants(self):

        shp = LENGTH if self.pixel else DIMS
        consts = [str(x) for x in list(Path(self.const_dir).iterdir())]
        consts = [np.memmap(x, dtype='float32', shape=shp) for x in consts]
        consts = np.array(consts)

        return consts

    def crop_loader(self, idx):
        idx_list, location = self.complete_ts[idx]
        feature_end = idx_list[0]
        feature_start = feature_end - self.n_weeks
        feature_range = list(range(max(0, feature_start), feature_end))

        x, y = location
        feats = []

        for feature in self.features:
            feats.append(feature[feature_range, y:y+self.crop_size, x:x+self.crop_size])

        feats = np.array(feats)
        shp = feats.shape
        missing_1, missing_2 = self.crop_size - shp[-2], self.crop_size - shp[-1]
        npad = ((0, 0), (0, 0), (0, missing_1), (0, missing_2))
        feats = np.pad(feats, pad_width=npad, mode='constant', constant_values=-1.5)

        npad = ((0, 0), (0, missing_1), (0, missing_2))
        targets = self.targets[idx_list, y:y+self.crop_size, x:x+self.crop_size]
        targets = np.pad(targets, pad_width=npad, mode='constant', constant_values=-1.5)

        consts = self.consts[:, y:y+self.crop_size, x:x+self.crop_size]
        consts = np.pad(consts, pad_width=npad, mode='constant', constant_values=-1.5)

        return feats, consts, targets

    @staticmethod
    def _read_pickle(p):
        with open(p, 'rb') as f:
            f = pickle.load(f)
        return f

    @staticmethod
    def _transforms(feat, const, target):

        tfm = np.random.choice(['none', 'rotate', 'ud', 'lr', 'lrud'])

        if tfm == 'none':
            pass
        elif tfm == 'rotate':
            rotation = np.random.randint(-180, 180, 1)
            feat = rotate(feat, axes=(-2, -1), angle=int(rotation), reshape=False, mode='constant', cval=-1.5, order=0)
            const = rotate(const, axes=(-2, -1), angle=int(rotation), reshape=False, mode='constant', cval=-1.5, order=0)
            target = rotate(target, axes=(-2, -1), angle=int(rotation), reshape=False, mode='constant', cval=0, order=0)
        elif tfm == 'ud':
            feat, const, target = feat[:, :, ::-1, :], const[:, ::-1, :], target[:, ::-1, :]
        elif tfm == 'lr':
            feat, const, target = feat[:, :, :, ::-1], const[:, :, ::-1], target[:, :, ::-1]
        else:
            feat, const, target = feat[:, :, ::-1, ::-1], const[:, ::-1, ::-1], target[:, ::-1, ::-1]

        return feat, const, target

    def __len__(self):
        return 20000
        # return len(self.complete_ts)

    def __getitem__(self, idx):

        if self.pixel:
            return self.pixel_loader(idx)
        else:
            arr, consts, targets = self.crop_loader(idx)
            arr = np.transpose(arr, (1, 0, 2, 3))

            if self.transform:
                arr, consts, targets = self._transforms(arr, consts, targets)

            return dtype(arr.copy()), dtype(consts.copy()), dtype(targets.copy())
