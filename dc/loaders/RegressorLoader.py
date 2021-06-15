import datetime as dt
from dateutil.relativedelta import relativedelta
from dc.utils.ImportantVars import LENGTH
import numpy as np
from pathlib import Path
import rasterio as rio
from torch.utils.data import Dataset


class RegressorLoader(Dataset):
    def __init__(self, ens_path: str = '/mnt/anx_lagr4/drought/models',
                 target_path: str = '/mnt/anx_lagr4/drought/out_classes/out_memmap',
                 mx_lead: int = 12, train: bool = True):

        test_years = ['2007', '2014', '2017']
        ens_dirs = ['ensemble_01', 'ensemble_02', 'ensemble_03', 'ensemble_04', 'ensemble_05',
                    'ensemble_06', 'ensemble_07', 'ensemble_08', 'ensemble_09', 'ensemble_10']

        self.ens_path = Path(ens_path)
        self.target_path = Path(target_path)
        self.days = self.get_days()

        self.days = [x for x in self.days if x[:4] not in test_years] if train else [x for x in self.days if x[:4] in test_years]
        all_files = []

        for day in self.days:
            for d in ens_dirs:
                tmp = self.ens_path.joinpath(d).joinpath('preds').joinpath(day + '_preds_None.tif')
                all_files.append(tmp)

        self.all_files = all_files
        self.mx_lead = mx_lead

        self.options = [(x, y) for x in self.days for y in range(mx_lead)]
    
    def make_ensemble(self, day, lead_time):

        f_list = [str(x) for x in self.all_files if day in str(x)]
        arrs = np.array([rio.open(x).read([lead_time+1]) for x in f_list])

        return arrs

    def get_days(self):
        files = [x for x in self.ens_path.joinpath('ensemble_01/preds').glob('*None.tif')]
        days = list(sorted(set([x.name[:8] for x in files])))

        return days

    def get_target(self, day, lead_time):
        target_day = str(dt.datetime.strptime(day, '%Y%m%d').date() + relativedelta(weeks=lead_time)).replace('-', '')
        pth = str(next(iter(self.target_path.glob(target_day+'*'))))

        return np.memmap(pth, dtype='int8', mode='r')/5

    def __len__(self):
        return len(self.options)

    def __getitem__(self, idx):

        day, lead_time = self.options[idx]
        
        x = self.make_ensemble(day, lead_time)
        x = x.squeeze().reshape(10, LENGTH)
        x = x.swapaxes(0, 1)

        y = self.get_target(day, lead_time)

        return x.copy(), y.copy(), lead_time
