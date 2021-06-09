from dc.utils.ImportantVars import DIMS
import numpy as np
import os
from pathlib import Path
import rasterio as rio


def save_arrays(out_dir: str, data: np.array, name: str, dtype: str):

    """
    :param out_dir: Path to directory that images will be saved in
    :param data: 12 x 284 x 622 np array to that will be saved as a raster.
    :param name: Name to use for new file.
    :param dtype: Data type to save array as. Either 'int8' or 'float32'.
    :return: Saves new image to the out_dir.
    """
    if not os.path.exists(out_dir):
        print('{} does not exist. Creating directory.'.format(out_dir))
        os.mkdir(out_dir)

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


def average_estimates(pth: str, out_dir: str, holdout: str = 'None'):
    """
    :param pth: Path to directory containing ensemble estimates.
    :param out_dir: Path to directory that images will be saved in
    :param holdout: Either 'None' or one of the weekly or monthly variables from ImportantVars.py
    :return: Saves out mean, min, max, sd, and median of model estimates.
    """

    opt = [x for x in Path(pth).glob('ensemble*')]
    options = [os.path.basename(x) for x in Path(opt[0]).glob('*' + holdout + '.tif')]

    for day in options:
        print(day)
        arr = []
        for model in opt:
            raster = rio.open(os.path.join(model, day))
            preds = raster.read(list(range(1, 13)))
            arr.append(preds)
            raster.close()

        save_arrays(os.path.join(out_dir, 'mean'), np.mean(arr, axis=0), day, 'float32')
        save_arrays(os.path.join(out_dir, 'min'), np.min(arr, axis=0), day, 'float32')
        save_arrays(os.path.join(out_dir, 'max'), np.max(arr, axis=0), day, 'float32')
        save_arrays(os.path.join(out_dir, 'sd'), np.std(arr, axis=0), day, 'float32')
        save_arrays(os.path.join(out_dir, 'median'), np.median(arr, axis=0), day, 'float32')








