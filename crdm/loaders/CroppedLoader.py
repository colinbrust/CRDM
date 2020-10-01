from crdm.loaders.AggregateTrainingPixels import PremakeTrainingPixels
from crdm.utils.ImportantVars import DIMS
from functools import partial
import glob
import numpy as np
import os
import torch
from torch.utils.data import Dataset


def crop(img, x_crop, y_crop, size):
    img = img[y_crop:y_crop+size, x_crop:x_crop+size]
    return img


class CroppedLoader(Dataset):
    
    def __init__(self, target_dir, in_features, lead_time=1, n_months=5, 
                 crop_size=64, crops_per_img=50, rm_features=False):

        self.crop_size = crop_size
        self.crops_per_img = crops_per_img

        targets = glob.glob(os.path.join(target_dir, '*.dat'))
        self.targets = [x for x in targets if not('/2015' in x or '/2016' in x)] if rm_features else targets
        self.agg_list = []

        for target in self.targets:
            try:

                agg = PremakeTrainingPixels(target, in_features, lead_time, n_months)
                if rm_features:
                    agg.remove_lat_lon()
                self.agg_list.append(agg)

            except AssertionError as e:
                print(e)

    
    def __len__(self):
        return len(self.agg_list) * self.crops_per_img
    
    def __getitem__(self, idx):

        x = np.random.randint(0, DIMS[1] - self.crop_size)
        y = np.random.randint(0, DIMS[0] - self.crop_size)
        
        agg = self.agg_list[idx//self.crops_per_img]

        arr_out = []
        for img in sorted([*agg.monthlys, *agg.constants]):
            tmp = np.memmap(img, dtype='float32', mode='r', shape=DIMS)
            tmp = np.nan_to_num(tmp, nan=-0.5)
            tmp = crop(tmp, x, y, self.crop_size)
            arr_out.append(tmp)

        month = int(os.path.basename(agg.target)[4:6])
        month = month * 0.01
        month = np.ones_like(arr_out[0]) * month

        day_diff = agg._get_day_diff()
        day_diff = day_diff * 0.01
        day_diff = np.ones_like(arr_out[0]) * day_diff

        arr_out.append(month)
        arr_out.append(day_diff)

        feats = torch.Tensor(arr_out)

        target = np.memmap(agg.target, dtype='int8', mode='r', shape=DIMS)
        target = crop(target, x, y, self.crop_size)

        return {'feats': feats, 
                'target': target}


