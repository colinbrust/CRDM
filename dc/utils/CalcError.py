from dc.utils.ImportantVars import DIMS, LENGTH
import datetime as dt
import dateutil.relativedelta as rd
from functools import lru_cache
import numpy as np
import os
import pandas as pd
from pathlib import Path
import pickle
import rasterio as rio
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix
from tqdm import tqdm


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


def save_arrays(out_path, data):

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


def get_spatial_error(base_dir='./data/models/preds',
                      target_dir='./data/out_classes/memmap'):

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



def get_all_error(base_dir='./data/models/best/model_00/preds',
                  target_dir='./data/targets',
                  locs='./data/models/final_model/locs.p',
                  out_dir='./data/plot_data/tables'):

    f_list = [x.as_posix() for x in Path(base_dir).glob('*None.tif')]
    locs = pickle.load(open(locs, 'rb'))

    train = []
    train_targ = []
    train_base = []

    test = []
    test_targ = []
    test_base = []

    validation = []
    val_targ = []
    val_base = []

    for f in sorted(f_list):
        try:
            date = strip_date(f)
            print(date)

            targets = get_targets(12, f, target_dir).reshape(12, LENGTH)
            pred = np.round(rio.open(f).read(list(range(1, 13)))).astype(np.int8).reshape(12, LENGTH)
            pred = np.clip(pred, 0, 5)
            baseline = next(Path(target_dir).glob(date+'*')).as_posix()
            baseline = np.array([np.memmap(baseline, dtype=np.int8)] * 12)

            if date[:4] in ['2007', '2014', '2017']:
                validation.append(pred)
                val_targ.append(targets)
                val_base.append(baseline)
            else:
                train.append(pred[:, locs['train']])
                test.append(pred[:, locs['test']])

                train_targ.append(targets[:, locs['train']])
                test_targ.append(targets[:, locs['test']])

                train_base.append(baseline[:, locs['train']])
                test_base.append(baseline[:, locs['test']])

        except ValueError as e:
            print(e)
            continue

    train = np.array(train).swapaxes(0, 1).reshape(12, -1)
    train_targ = np.array(train_targ).swapaxes(0, 1).reshape(12, -1)
    # train_base = np.array(train_base).swapaxes(0, 1).reshape(12, -1)

    test = np.array(test).swapaxes(0, 1).reshape(12, -1)
    test_targ = np.array(test_targ).swapaxes(0, 1).reshape(12, -1)
    # test_base = np.array(test_base).swapaxes(0, 1).reshape(12, -1)

    validation = np.array(validation).swapaxes(0, 1).reshape(12, -1)
    val_targ = np.array(val_targ).swapaxes(0, 1).reshape(12, -1)
    # val_base = np.array(val_base).swapaxes(0, 1).reshape(12, -1)

    train_mse = []
    train_cor = []

    test_mse = []
    test_cor = []

    val_mse = []
    val_cor = []

    for lead_time in range(12):
        train_cor.append(round(np.corrcoef(train[lead_time, :], train_targ[lead_time, :])[0, 1], 4))
        train_mse.append(round(mean_squared_error(train[lead_time, :], train_targ[lead_time, :]), 4))

        test_cor.append(round(np.corrcoef(test[lead_time, :], test_targ[lead_time, :])[0, 1], 4))
        test_mse.append(round(mean_squared_error(test[lead_time, :], test_targ[lead_time, :]), 4))

        val_cor.append(round(np.corrcoef(validation[lead_time, :], val_targ[lead_time, :])[0, 1], 4))
        val_mse.append(round(mean_squared_error(validation[lead_time, :], val_targ[lead_time, :]), 4))

    dat = {'train_cor': train_cor,
           'train_mse': train_mse,
           'test_cor': test_cor,
           'test_mse': test_mse,
           'val_cor': val_cor,
           'val_mse': val_mse}

    df = pd.DataFrame(dat)
    df.to_csv(os.path.join(out_dir, 'complete_err.csv'), index=False)


def calc_error(arr, f, target_dir):

    targets = get_targets(arr.count, f, target_dir)
    raster = read_raster(arr)

    mse = [mean_squared_error(targets[i], raster[i]) for i in range(arr.count)]
    r2 = [r2_score(targets[i], raster[i]) for i in range(arr.count)]

    return mse, r2


def get_model_runs(base_dir, target_dir):

    mx_lead = 12
    pth = Path(base_dir)

    out = []
    for sub in pth.glob('model*'):
        # for subsub in sub.glob('preds*'):
        for pred in sub.glob('*.tif'):
            print(pred)
            info = [pred.as_posix()] * mx_lead
            metadata = {'name': info}

            arr = rio.open(pred)
            mse, r2 = calc_error(arr, pred, target_dir)

            metadata['mse'], metadata['r2'] = mse, r2
            df = pd.DataFrame(metadata)
            df['lead_time'] = list(range(1, mx_lead+1))
            df['pred'] = os.path.basename(pred)
            out.append(df)

    out_dat = pd.concat(out, ignore_index=True)
    out_dat.to_csv('./data/ensemble_results.csv', index=False)
