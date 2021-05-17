from crdm.utils.ImportantVars import DIMS
import datetime as dt
import dateutil.relativedelta as rd
from functools import lru_cache
import numpy as np
import os
import pandas as pd
from pathlib import Path
import pickle
import rasterio as rio
from sklearn.metrics import mean_squared_error, r2_score


def read_raster(arr):

    out = []

    for i in range(1, arr.count + 1):
        out.append(arr.read(i))

    return np.array(out)


def strip_date(f):
    return os.path.basename(f).split('_')[0]


def get_target_dates(date, lead_range):
    dates = [str(date + rd.relativedelta(weeks=x)) for x in range(1, lead_range + 1)]
    dates = [x.replace('-', '') for x in dates]
    return list(sorted(dates))


@lru_cache(maxsize=1000)
def get_targets(count, f, target_dir):

    date = dt.datetime.strptime(strip_date(f), '%Y%m%d').date()
    target_dates = get_target_dates(date, count)
    pth = Path(target_dir)

    targets = []
    for d in target_dates:
        targets += [x.as_posix() for x in pth.glob(d+'*')]

    return np.array([np.memmap(x, shape=DIMS, dtype='int8') for x in targets])


def calc_error(arr, f, target_dir):

    targets = get_targets(arr.count, f, target_dir)
    raster = read_raster(arr)

    mse = [mean_squared_error(targets[i], raster[i]) for i in range(arr.count)]
    r2 = [r2_score(targets[i], raster[i]) for i in range(arr.count)]

    return mse, r2


def get_model_runs(base_dir, target_dir):

    pth = Path(base_dir)

    out = []
    for sub in pth.iterdir():
        for subsub in sub.iterdir():
            unq = set([os.path.basename(x.as_posix()).split('_')[-1].replace('.p', '') for x in subsub.glob('preds_*')])
            for match in unq:
                print(subsub.as_posix(), match)

                metadata = [x.as_posix() for x in subsub.glob('metadata_'+match+'.p')][0]
                with open(metadata, 'rb') as f:
                    metadata = pickle.load(f)

                preds = [y.as_posix() for y in ([x for x in subsub.glob('*_'+match)][0]).iterdir()]
                mx_lead = metadata['mx_lead']
                for k, v in metadata.items():
                    metadata[k] = [v] * mx_lead
                for pred in preds:

                    arr = rio.open(pred)
                    mse, r2 = calc_error(arr, pred, target_dir)

                    metadata['mse'], metadata['r2'] = mse, r2
                    df = pd.DataFrame(metadata)
                    df['lead_time'] = list(range(1, mx_lead+1))
                    df['pred'] = os.path.basename(pred)
                    out.append(df)

    out_dat = pd.concat(out, ignore_index=True)
    out_dat.to_csv('./data/err_search.csv', index=False)

f = '/mnt/e/PycharmProjects/DroughtCast/data/models/global_norm/model0/preds_0/20170704_preds_None.tif'
target_dir = '/mnt/e/PycharmProjects/DroughtCast/data/targets'