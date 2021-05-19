from crdm.utils.ImportantVars import DIMS
import datetime as dt
from dateutil import relativedelta as rd
import numpy as np
import os
from pathlib import Path
import rasterio as rio

def calc_ts_bins(pred_dir, target_dir):

    f_list = [x.as_posix() for x in Path(pred_dir).glob('*None.tif')]
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

    for day in range(len(preds)):
        for lead_time in range(12):
            print(preds)