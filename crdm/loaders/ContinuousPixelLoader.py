from crdm.utils.FilterMissing import find_missing_dates
from crdm.utils.ImportantVars import LENGTH, MONTHLY_VARS, WEEKLY_VARS
import datetime as dt
import dateutil.relativedelta as rd
from functools import partial
from itertools import chain
import numpy as np
import os
from pathlib import Path
import re
from torch.utils.data import Dataset
import torch
from typing import List, Tuple


class PixelLoader(Dataset):

    def __init__(self, target_dir: str, feature_dir: str, pixel_per_img: int, lead_time: int, n_weeks: int = 17):

        # target_dir = '/mnt/e/PycharmProjects/CRDM/data/out_classes/out_memmap'
        # feature_dir = '/mnt/e/PycharmProjects/CRDM/data/in_features'

        self.target_dir = target_dir
        self.feature_dir = feature_dir
        self.pixel_per_img = pixel_per_img
        self.lead_time = lead_time
        self.n_weeks = n_weeks
        self.n_months = n_weeks // 4

        self.valid_dates = find_missing_dates(os.path.join(feature_dir, 'weekly_mem'))

        # Get list of months from 2003-01-01 to 2019-12-01
        self.valid_months = [dt.date(2003, 1, 1) + rd.relativedelta(months=x) for x in range(1, 17*12)]
        self.valid_months = [str(x).replace('-', '') for x in self.valid_months]
        self.targets = [str(x) for x in Path(self.target_dir).glob('*.dat')]
        self.targets = self._get_target_feature_pairs()
        self.keys = list(self.targets.keys())
        self.constants = [str(x) for x in Path(os.path.join(feature_dir, 'constant_mem')).iterdir()]
        self.indices = [np.random.randint(low=0, high=LENGTH, size=self.pixel_per_img) for _ in range(len(self.keys))]

    def _get_target_feature_pairs(self) -> dict:
        """
        Get a list of feature dates to use given the target image.
        """
        month_list = [str(x) for x in Path(os.path.join(self.feature_dir, 'monthly_mem')).iterdir()]
        week_list = [str(x) for x in Path(os.path.join(self.feature_dir, 'weekly_mem')).iterdir()]
        annual_list = [str(x) for x in Path(os.path.join(self.feature_dir, 'annual_mem')).iterdir()]

        out = {}
        for target in self.targets:

            # find the date that is 'lead_time' weeks away from the target
            d = dt.datetime.strptime(os.path.basename(target)[:8], '%Y%m%d').date()
            target_date = d
            d = d - rd.relativedelta(weeks=self.lead_time)
            init_date = d

            # Find the input feature image dates for weekly and monthly features.
            dates = [str((d - rd.relativedelta(weeks=x)) - rd.relativedelta(days=1)) for x in range(self.n_weeks)]
            start_month = dt.datetime.strptime(dates[0][:-2] + '01', '%Y-%m-%d').date()
            months = [str(start_month - rd.relativedelta(months=x)) for x in range(self.n_months)]

            dates = [x.replace('-', '') for x in dates]
            months = [x.replace('-', '') for x in months]

            # If all necessary images are present, add the dates to the list
            if all([x in self.valid_months for x in months]) and all([x in self.valid_dates for x in dates]):

                year = sorted(dates)[-1][:4]
                weeks = [x for x in week_list for y in dates if y in x]
                weeks_out = []
                for var in WEEKLY_VARS:
                    filt = sorted([x for x in weeks if var + '.dat' in x])
                    weeks_out.append(filt)
                weeks_out = list(chain(*weeks_out))

                months = [x for x in month_list for y in months if y in x]
                months_out = []
                for var in MONTHLY_VARS:
                    filt = sorted([x for x in months if var + '.dat' in x])
                    months_out.append(filt)
                months_out = list(chain(*months_out))

                annual = [x for x in annual_list if year in x]
                init_drought = self._get_init_drought_status(str(init_date).replace('-', ''))
                doy_data = self._get_doy_data(target_date, init_date)

                out[target] = {'months': months_out,
                               'weeks': weeks_out,
                               'annual': annual[0],
                               'init_drought': init_drought,
                               'doy_data': doy_data}

        return out

    @staticmethod
    def _get_doy_data(target_date, init_date):

        # Add day of year for target image.
        target_doy = target_date.timetuple().tm_yday
        target_doy = target_doy * 0.001

        # Add day of year for image guess date.
        guess_doy = init_date.timetuple().tm_yday
        guess_doy = guess_doy * 0.001

        day_diff = (target_date - init_date).days
        day_diff = day_diff * 0.001

        return [target_doy, guess_doy, day_diff]

    def _get_init_drought_status(self, date) -> str:

        # Find the date that is lead_time weeks away from the target USDM image.
        out = [x for x in self.targets if date in x]

        assert len(out) == 1, 'No initial USDM images available for this target.'

        return out[0]

    @staticmethod
    def _read_image(img, idx, as_int=False):
        dType = 'int8' if as_int else 'float32'
        out = np.memmap(img, dType, 'c')
        return out[idx]

    def __len__(self):
        return len(self.keys) * self.pixel_per_img

    def __getitem__(self, idx):

        img = idx // self.pixel_per_img
        pix = idx - (img * self.pixel_per_img)
        data = self.indices[img][pix]

        feature_dict = self.targets[self.keys[img]]

        float_func = partial(self._read_image, idx=data, as_int=False)
        int_func = partial(self._read_image, idx=data, as_int=True)
        weeks = list(map(float_func, feature_dict['weeks']))
        weeks = np.array(weeks).reshape((len(WEEKLY_VARS), self.n_weeks))
        months = list(map(float_func, feature_dict['months']))
        months = np.array(months).reshape((len(MONTHLY_VARS), self.n_months))

        annual = self._read_image(feature_dict['annual'], data, False)
        init_drought = self._read_image(feature_dict['init_drought'], data, True)
        const = list(map(float_func, self.constants))

        const = const + feature_dict['doy_data']
        const.append(annual)
        const.append(init_drought/6)

        target = self._read_image(self.keys[img], data, True)

        return {'const': torch.tensor(const),
                'mon': torch.tensor(months),
                'week': torch.tensor(weeks),
                'target': torch.tensor(target)}
