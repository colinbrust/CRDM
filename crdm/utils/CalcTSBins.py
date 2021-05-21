from crdm.utils.ImportantVars import DIMS
import datetime as dt
from dateutil import relativedelta as rd
import numpy as np
import os
import pandas as pd
from pathlib import Path
import rasterio as rio


def calc_ts_bins(pred_dir, target_dir, mask='./data/pix_mask.dat', out_dir='./data/err_maps'):

    mask = np.memmap(mask, shape=DIMS, dtype='int8').astype(np.bool)
    mask = np.where(mask == True, False, True)

    f_list = sorted([x.as_posix() for x in Path(pred_dir).glob('*None.tif')])
    preds = np.array([rio.open(x).read([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]) for x in f_list])

    dates = [os.path.basename(x).split('_')[0] for x in f_list]
    last_date = dt.datetime.strptime(dates[-1], '%Y%m%d').date()
    tail_dates = [str(last_date + rd.relativedelta(weeks=x)).replace('-', '') for x in range(1, 12)]
    dates += tail_dates

    targ_path = Path(target_dir)
    targets = list(sorted([next(targ_path.glob(date+'*')).as_posix() for date in dates]))

    targ_arr = []

    for i in range(len(targets) - 11):
        tmp = np.array([np.memmap(x, shape=DIMS, dtype='int8') for x in targets[i:i+12]])
        targ_arr.append(tmp)

    targets = np.array(targ_arr)

    out = pd.DataFrame()
    for day in range(len(preds)):
        print(dates[day])

        for lead_time in range(12):
            tmp_pred = np.round(preds[day, lead_time, ...].copy()).astype(np.int)
            np.putmask(tmp_pred, mask, 6)
            tmp_targ = targets[day, lead_time, ...].copy()
            np.putmask(tmp_targ, mask, 6)

            pred = np.bincount(tmp_pred.ravel())
            targ = np.bincount(tmp_targ.ravel())
            tmp_df = pd.DataFrame({'pred': pred, 'targ': targ, 'lt': lead_time,
                                   'day': dates[day], 'category': [0, 1, 2, 3, 4, 5, 6]})
            out = pd.concat([out, tmp_df], ignore_index=True)

    out.to_csv(os.path.join(out_dir, 'bin_counts.csv'), index=False)
