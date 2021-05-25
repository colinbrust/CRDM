from crdm.utils.ImportantVars import DIMS
import numpy as np
import os
from pathlib import Path
import rasterio as rio
from scipy.stats import mode

opt = ["data/models/small_search/ensemble_46/preds", "data/models/small_search/ensemble_29/preds",
       "data/models/small_search/ensemble_22/preds", "data/models/small_search/ensemble_7/preds",
       "data/models/small_search/ensemble_32/preds", "data/models/small_search/ensemble_47/preds",
       "data/models/small_search/ensemble_39/preds", "data/models/small_search/ensemble_41/preds",
       "data/models/small_search/ensemble_0/preds", "data/models/small_search/ensemble_23/preds"]


def save_arrays(out_dir, data, name, dtype):

    out_dst = rio.open(
        os.path.join(out_dir, name),
        'w',
        driver='GTiff',
        height=DIMS[0],
        width=DIMS[1],
        count=12,
        dtype=dtype,
        transform=rio.Affine(9000.0, 0.0, -12048530.45, 0.0, -9000.0, 5568540.83),
        crs='+proj=cea +lon_0=0 +lat_ts=30 +x_0=0 +y_0=0 +ellps=WGS84 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs'
    )

    out_dst.write(data.astype(np.int8 if dtype == 'int8' else np.float32))
    out_dst.close()


def average_estimates(out_dir, holdout='None'):

    # opt = [x for x in Path(pth).glob('ensemble*')]
    # opt = [list(x.glob('preds*')) for x in opt]
    # opt = [x[0].as_posix() for x in opt if len(x) > 0]

    options = [os.path.basename(x) for x in Path(opt[0]).glob('*' + holdout + '.tif')]

    for day in options:
        print(day)
        arr = []
        for model in opt:
            raster = rio.open(os.path.join(model, day))
            preds = raster.read([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
            arr.append(preds)
            raster.close()

        save_arrays(os.path.join(out_dir, 'mean'), np.mean(arr, axis=0), day, 'float32')
        save_arrays(os.path.join(out_dir, 'min'), np.min(arr, axis=0), day, 'float32')
        save_arrays(os.path.join(out_dir, 'max'), np.max(arr, axis=0), day, 'float32')
        save_arrays(os.path.join(out_dir, 'sd'), np.std(arr, axis=0), day, 'float32')







