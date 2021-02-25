from datetime import datetime as dt
from datetime import timedelta
import numpy as np
import os
import pickle

# TODO: Rerun MakeStack.py so that it includes variables in info.dat
class DataStack(object):

    def __init__(self, data_dir):

        self.data_dir = data_dir

        self.weekly, self.weekly_info = self.read('weekly')
        self.monthly, self.monthly_info = self.read('monthly')
        self.annual, self.annual_info = self.read('annual')
        self.constant, self.constant_info = self.read('constant')
        self.target, self.target_info = self.read('target')

    def read(self, variable):
        dat = os.path.join(self.data_dir, variable+'.dat')
        info = os.path.join(self.data_dir, variable+'_info.dat')

        with open(info, 'rb') as pick:
            info = pickle.load(pick)

        dat = np.memmap(dat, dtype='float32', shape=info['shp'])
        return dat, info

    def check_complete(self, target):
        pass

    def sample(self):
        pass

    def holdout_variable(self, variable):
        pass

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass
date_str = os.path.basename(target.as_posix())[:8]
date = dt.strptime(date_str, '%Y%M%d').date()
date_range = [str(date - timedelta(weeks=x)).replace('-', '') for x in range(1, 9)]
mon_range = set([x[:6] for x in date_range])

weekly_candidates = [x + '_' + y + '.dat' for x in date_range for y in WEEKLY_VARS]