from crdm.utils.ImportantVars import DIMS
import numpy as np
import os
from pathlib import Path
import rasterio as rio
from scipy.stats import mode

opt = ['data/models/ensemble_12/preds_32', 'data/models/ensemble_38/preds_35', 'data/models/ensemble_20/preds_30',
       'data/models/ensemble_17/preds_30', 'data/models/ensemble_29/preds_35', 'data/models/ensemble_51/preds_31',
       'data/models/ensemble_101/preds_37', 'data/models/ensemble_46/preds_33', 'data/models/ensemble_34/preds_26',
       'data/models/ensemble_3/preds_23', 'data/models/ensemble_5/preds_27', 'data/models/ensemble_24/preds_31',
       'data/models/ensemble_56/preds_30']


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


def average_estimates(pth, out_dir, holdout='None'):

    opt = [x for x in Path(pth).glob('ensemble*')]
    opt = [list(x.glob('preds*')) for x in opt]
    opt = [x[0].as_posix() for x in opt if len(x) > 0]

    options = [os.path.basename(x) for x in Path(opt[0]).glob('*' + holdout + '.tif')]

    for day in options:
        print(day)
        arr = []
        for model in opt:
            raster = rio.open(os.path.join(model, day))
            preds = raster.read([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
            arr.append(preds)
            raster.close()

        save_arrays(os.path.join(out_dir, 'median'), np.median(arr, axis=0), day, 'int8')
        save_arrays(os.path.join(out_dir, 'min'), np.min(arr, axis=0), day, 'int8')
        save_arrays(os.path.join(out_dir, 'max'), np.max(arr, axis=0), day, 'int8')
        save_arrays(os.path.join(out_dir, 'sd'), np.std(arr, axis=0), day, 'float32')







