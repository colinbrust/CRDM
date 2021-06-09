from dc.utils.ImportantVars import DIMS, LENGTH
import datetime as dt
import dateutil.relativedelta as rd
from functools import lru_cache
import numpy as np
import os
from pathlib import Path
import rasterio as rio
from sklearn.metrics import mean_squared_error
from tqdm import tqdm


def strip_date(f: str) -> str:
    """
    :param f: Filename formatted as 'YYYYMMDD_variable.extension'
    :return: Date string stripped from beginning of file.
    """
    return os.path.basename(f).split('_')[0]


def get_target_dates(date: dt.date, lead_range: int) -> List[str]:
    """
    Get list of dates for target images.
    :param date: Date of first USDM image target.
    :param lead_range: Number of weeks in the future to stack USDM images.
    :return: List of dates of USDM images to use.
    """
    dates = [str(date + rd.relativedelta(weeks=x)) for x in range(1, lead_range + 1)]
    dates = [x.replace('-', '') for x in dates]
    return list(sorted(dates))


@lru_cache(maxsize=1000)
def get_targets(count: int, f: str, target_dir: str) -> np.array:
    """
    Stack USDM images into a timeseries.
    :param count: Number of USDM images to include in timeseries
    :param f: Filename of first USDM image.
    :param target_dir: Directory containing USDM memmaps.
    :return: np.array of USDM timeseries.
    """
    date = dt.datetime.strptime(strip_date(f), '%Y%m%d').date()
    target_dates = get_target_dates(date, count)
    pth = Path(target_dir)

    targets = []
    for d in target_dates:
        targets += [x.as_posix() for x in pth.glob(d+'*')]

    return np.array([np.memmap(x, shape=DIMS, dtype='int8') for x in targets])


def save_arrays(out_path: str, data: np.array):

    """
    Save np array as geotiff
    :param out_path: Path to save geotiff out as.
    :param data: np array to be saved as geotiff.
    :return: Saves out geotiff.
    """

    out_dst = rio.open(
        out_path,
        'w',
        driver='GTiff',
        height=DIMS[0],
        width=DIMS[1],
        count=1,
        dtype='float32',
        transform=rio.Affine(9000.0, 0.0, -12048530.45, 0.0, -9000.0, 5568540.83),
        crs='+proj=cea +lon_0=0 +lat_ts=30 +x_0=0 +y_0=0 +ellps=WGS84 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs'
    )

    out_dst.write(data.astype(np.float32))
    out_dst.close()


def get_spatial_error(base_dir: str = './data/models/preds', target_dir: str = './data/out_classes/memmap'):

    f_list = list(sorted([x.as_posix() for x in Path(base_dir).glob('*None.tif')]))

    preds = []
    targs = []

    pred_val = []
    targ_val = []

    for f in f_list:
        print(f)
        target = get_targets(12, f, target_dir).reshape(12, LENGTH)
        pred = np.round(rio.open(f).read(list(range(1, 13)))).astype(np.int8).reshape(12, LENGTH)
        pred = np.clip(pred, 0, 5)

        if strip_date(f)[:4] in ['2007', '2014', '2017']:
            print('Val')
            pred_val.append(pred)
            targ_val.append(target)
        else:
            preds.append(pred)
            targs.append(target)

    pred_arr = np.mean(np.array(preds), axis=1)
    targ_arr = np.mean(np.array(targs), axis=1)

    pred_arr_val = np.mean(np.array(pred_val), axis=1)
    targ_arr_val = np.mean(np.array(targ_val), axis=1)

    mse_arr = []
    r_arr = []

    mse_arr_val = []
    r_arr_val = []

    for pix in tqdm(range(pred_arr.shape[-1])):

        mse_arr.append(mean_squared_error(targ_arr[:, pix], pred_arr[:, pix]))
        r_arr.append(np.corrcoef(targ_arr[:, pix], pred_arr[:, pix])[0, 1])

        mse_arr_val.append(mean_squared_error(targ_arr_val[:, pix], pred_arr_val[:, pix]))
        r_arr_val.append(np.corrcoef(targ_arr_val[:, pix], pred_arr_val[:, pix])[0, 1])

    save_arrays('./data/plot_data/figs/mse_train.tif', np.array(mse_arr).reshape(DIMS)[np.newaxis])
    save_arrays('./data/plot_data/figs/r_train.tif', np.array(r_arr).reshape(DIMS)[np.newaxis])

    save_arrays('./data/plot_data/figs/mse_val.tif', np.array(mse_arr_val).reshape(DIMS)[np.newaxis])
    save_arrays('./data/plot_data/figs/r_val.tif', np.array(r_arr_val).reshape(DIMS)[np.newaxis])

