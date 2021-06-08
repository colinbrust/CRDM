from dc.utils.ImportantVars import DIMS
import datetime as dt
from dateutil import relativedelta as rd
import numpy as np
import os
from pathlib import Path
import rasterio as rio


def calc_entire_ts_error(pred_dir, true_dir, variable, ann_dir, mon_dir):

    preds = list(sorted([x.as_posix() for x in Path(pred_dir).glob('*'+variable+'.tif')]))

    # Get dates of all target images
    dates = [os.path.basename(x).split('_')[0] for x in preds]
    last_date = dt.datetime.strptime(dates[-1], '%Y%m%d').date()
    tail_dates = [str(last_date + rd.relativedelta(weeks=x)).replace('-', '') for x in range(1, 12)]
    dates += tail_dates

    targ_path = Path(true_dir)
    targets = list(sorted([next(targ_path.glob(date+'*')).as_posix() for date in dates]))

    targ_arr = []

    for i in range(len(targets) - 11):
        tmp = np.array([np.memmap(x, shape=DIMS, dtype='int8') for x in targets[i:i+12]])
        targ_arr.append(tmp)

    targets = np.array(targ_arr)

    month_indices = np.array([os.path.basename(x)[4:6] for x in preds])
    preds = np.array([rio.open(x).read([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]) for x in preds])

    targets = targets.reshape((len(targets), 12, -1))
    preds = preds.reshape((len(preds), 12, -1))

    for month in set(month_indices):
        arr_stack = []
        idx = np.where(month_indices == month)[0]
        for lead_time in range(12):
            tmp_targ = targets[idx, lead_time, :]
            tmp_pred = preds[idx, lead_time, :]

            arr_stack.append(((tmp_targ-tmp_pred)**2).mean(axis=0).reshape(DIMS))

        err = np.array(arr_stack)

        out_dst = rio.open(
            os.path.join(mon_dir, 'err_' + variable + '_' + month + '.tif'),
            'w',
            driver='GTiff',
            height=DIMS[0],
            width=DIMS[1],
            count=len(err),
            dtype='float32',
            transform=rio.Affine(9000.0, 0.0, -12048530.45, 0.0, -9000.0, 5568540.83),
            crs='+proj=cea +lon_0=0 +lat_ts=30 +x_0=0 +y_0=0 +ellps=WGS84 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs'
        )

        out_dst.write(err.astype(np.float32))
        out_dst.close()

    arr_stack = []
    for lead_time in range(12):
        tmp_targ = targets[:, lead_time, :]
        tmp_pred = preds[:, lead_time, :]

        arr_stack.append(((tmp_targ-tmp_pred)**2).mean(axis=0).reshape(DIMS))

    err = np.array(arr_stack)

    out_dst = rio.open(
        os.path.join(ann_dir, 'err_'+variable+'.tif'),
        'w',
        driver='GTiff',
        height=DIMS[0],
        width=DIMS[1],
        count=len(err),
        dtype='float32',
        transform=rio.Affine(9000.0, 0.0, -12048530.45, 0.0, -9000.0, 5568540.83),
        crs='+proj=cea +lon_0=0 +lat_ts=30 +x_0=0 +y_0=0 +ellps=WGS84 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs'
    )

    out_dst.write(err.astype(np.float32))
    out_dst.close()

pred_dir = './data/models/best/model_00/preds'
true_dir = './data/targets'
ann_dir = './data/err_maps/annual'
mon_dir = './data/err_maps/monthly'
for v in ['None'] + list(holdouts.keys()):
    calc_entire_ts_error(pred_dir, true_dir, v, ann_dir, mon_dir)
