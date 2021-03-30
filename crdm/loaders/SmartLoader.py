from crdm.utils.ImportantVars import STACK_SHP, TRAIN_INDICES, TEST_INDICES, LENGTH
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset


class SmartLoader(Dataset):

    def __init__(self, feature_dir, train=True, max_lead_time=12, n_weeks=25):

        p = Path(feature_dir)

        self.targets = np.memmap(str(list(p.glob('USDM.dat'))[0]), dtype='float32', shape=STACK_SHP)
        self.features = [np.memmap(x, dtype='float32', shape=STACK_SHP) for x in p.iterdir()]
        self.indices = TRAIN_INDICES if train else TEST_INDICES
        self.max_lead_time = max_lead_time
        self.n_weeks = n_weeks

        self.complete_ts = [list(range(x, x+max_lead_time)) for x in range(1, len(self.targets))]
        self.complete_ts = [x for x in self.complete_ts if all(y in self.indices for y in x)]

    def pixel_loader(self):
        pass

    def crop_loader(self):
        pass

    def __len__(self):
        return len(self.complete_ts)

    def __getitem__(self, idx):

        # TODO: Make this just implement either pixel loader or crop loader depending on the type of model I'm training.
        # Edge case of IDX == 0
        idx_list = self.complete_ts[idx]
        feature_end = idx_list[0] - 1
        feature_start = feature_end - self.n_weeks
        feature_range = list(range(max(0, feature_start), feature_end))

        pixel = np.random.randint(0, LENGTH, 1)

        feats = []
        for feature in self.features:
            feats.append(np.take(feature[feature_range], pixel, axis=-1))

        feats = np.array(feats)
        targets = np.take(self.targets[idx_list], pixel, axis=-1)

        return feats, targets
