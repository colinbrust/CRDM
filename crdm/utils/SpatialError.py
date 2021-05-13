from crdm.utils.ImportantVars import DIMS, LENGTH
import datetime as dt
from dateutil import relativedelta as rd
import numpy as np
import os
from pathlib import Path
import rasterio as rio
from sklearn.metrics import mean_squared_error as mse

pred_dir = './data/old/old/model3/preds_1'
true_dir = './data/targets'
variable = 'None'


def calc_entire_ts_error(pred_dir, true_dir, variable):

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
    preds = np.array([rio.open(x).read([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]) for x in preds])

    targets = targets.reshape((len(targets), 12, -1))
    preds = preds.reshape((len(preds), 12, -1))

    for lead_time in range(12):
        tmp_targ = targets[:, lead_time, :]
        tmp_pred = preds[:, lead_time, :]
        test = []
        for pix in range(LENGTH):
            print(pix)
            test.append(mse(tmp_targ[:, pix], tmp_pred[:, pix]))
        break