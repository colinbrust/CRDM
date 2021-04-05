from crdm.utils.ImportantVars import STACK_SHP, TRAIN_INDICES, TEST_INDICES, LENGTH, DIMS
import numpy as np
import os
from pathlib import Path
import pickle
import rasterio as rio
from skimage.util.shape import view_as_windows
from torch.utils.data import Dataset
import torch


class DroughtLoader(Dataset):

    def __init__(self, result_dir, feature_dir, const_dir, train=True, pixel=False, overlap=4):

        self.metadata = self._read_pickle(os.path.join(result_dir, 'metadata.p'))
        self.model = os.path.join(result_dir, 'model.p')
        self.indices = TRAIN_INDICES if train else TEST_INDICES
        self.max_lead_time = self.metadata['max_lead_time']
        self.n_weeks = self.metadata['n_weeks']
        self.pixel = pixel
        self.crop_size = self.metadata['crop_size']
        self.feats = self.metadata['feats']
        self.num_features = 17 if self.feats[0] == '*' else len(self.feats)
        self.const_dir = const_dir
        self.overlap = overlap

        p = Path(feature_dir)
        ps = [list(p.glob(x+'.dat'))[0] for x in self.feats] if self.feats[0] != '*' else list(p.glob('*.dat'))

        self.shp = STACK_SHP if pixel else (STACK_SHP[0], *DIMS)
        self.targets = np.memmap(str(list(p.glob('USDM.dat'))[0]), dtype='float32', shape=self.shp)
        self.features = [np.memmap(str(x), dtype='float32', shape=self.shp) for x in ps]
        self.constants = self._make_constants()

    def _make_constants(self):

        shp = LENGTH if self.pixel else DIMS
        consts = [str(x) for x in list(Path(self.const_dir).iterdir())]
        consts = [np.memmap(x, dtype='float32', shape=shp) for x in consts]
        consts = np.array(consts)

        return consts

    @staticmethod
    def _read_pickle(p):

        with open(p, 'rb') as f:
            dat = pickle.load(f)
        return dat

    def make_indices(self):
        col_indices, row_indices = list(range(0, DIMS[1], self.overlap)), list(range(0, DIMS[0], self.overlap))

    def model_setup(self):
        pass

    def run_model(self):
        pass

    def mosaic(self):
        pass

    def to_tiff(self):

        out_dst = rio.open(
            os.path.join(out_dir, os.path.basename(target[0]).replace('_USDM.dat', '_preds.tif')),
            'w',
            driver='GTiff',
            height=DIMS[0],
            width=DIMS[1],
            count=1,
            dtype='float32',
            transform=rio.Affine(9000.0, 0.0, -12048530.45, 0.0, -9000.0, 5568540.83),
            crs='+proj=cea +lon_0=0 +lat_ts=30 +x_0=0 +y_0=0 +ellps=WGS84 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs'
        )

        out_dst.write(np.expand_dims(out.reshape(DIMS), axis=0))
        out_dst.close()

