from crdm.loaders.AggregateAllSpatial import AggregateAllSpatial
from crdm.utils.ImportantVars import DIMS, WEEKLY_VARS, MONTHLY_VARS
import glob
import numpy as np
import os
from torch.utils.data import Dataset


class CroppedLoader(Dataset):

    def __init__(self, target_dir, in_features, n_weeks=5, crop_size=32, batch_size=256, test=False):

        self.targets = glob.glob(os.path.join(target_dir, '*.dat'))
        if test:
            self.targets = [x for x in self.targets if ('/2007' in x or '/2015' in x or '/2017' in x)]
        else:
            self.targets = [x for x in self.targets if not ('/2007' in x or '/2015' in x or '/2017' in x)]

        np.random.shuffle(self.targets)

        self.in_features = in_features
        self.n_weeks = n_weeks
        self.crop_size = crop_size
        self.batch_size = batch_size

    def __len__(self):
        return len(self.targets) * 4  # * 4 for 4 different lead times.

    def __getitem__(self, idx):

        # TODO: Make targets and lead_time generators so I can iterate through everything in one epoch.
        test = AggregateAllSpatial(target, in_features, lead_time=2, n_weeks=5, memmap=True)

        features = test.premake_features()
        n_features, n_weeks, size = features.shape
        features = features.reshape(n_features, n_weeks, *DIMS)
        out = []

        for i in range(batch_size):
            x = np.random.randint(0, DIMS[1] - 32)
            y = np.random.randint(0, DIMS[0] - 32)

            tmp = features[:, :, y:y + crop_size, x:x + crop_size]
            out.append(tmp)

        crop = np.array(out)
        crop = crop.swapaxes(1, 2)