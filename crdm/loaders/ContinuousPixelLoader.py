from crdm.utils.FilterMissing import find_missing_dates
from crdm.utils.ImportantVars import WEEKLY_VARS, MONTHLY_VARS
from crdm.utils.ParseFileNames import parse_fname
import datetime as dt
import dateutil.relativedelta as rd
import numpy as np
import os
from pathlib import Path
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
        self.targets = self._get_date_list()
        self.keys = list(self.targets.keys())

    def _get_date_list(self) -> dict:
        """
        Get a list of feature dates to use given the target image.
        """

        out = {}
        for target in self.targets:

            # find the date that is 'lead_time' weeks away from the target
            d = dt.datetime.strptime(os.path.basename(target)[:8], '%Y%m%d').date()
            d = d - rd.relativedelta(weeks=self.lead_time)

            # Find the input feature image dates for weekly and monthly features.
            dates = [str((d - rd.relativedelta(weeks=x)) - rd.relativedelta(days=1)) for x in range(self.n_weeks)]
            start_month = dt.datetime.strptime(dates[0][:-2] + '01', '%Y-%m-%d').date()
            months = [str(start_month - rd.relativedelta(months=x)) for x in range(self.n_months)]

            dates = [x.replace('-', '') for x in dates]
            months = [x.replace('-', '') for x in months]

            # If all necessary images are present, add the dates to the list
            if all([x in self.valid_months for x in months]) and all([x in self.valid_dates for x in dates]):
                #TODO: Change this so that months and dates are the full image paths rather than just the dates.
                out[target] = {'months': months,
                               'dates': dates}

        return out

    def get_images(self, idx):

        target = self.keys[idx]
        dates = self.targets[target]

    '''
    def get_constants(self):

        # dim = variable x location
        constants = [np.memmap(x, 'float32', 'c') for x in [*self.constants, *self.annuals]]
        constants = np.array(constants)
        constants = np.take(constants, indices, axis=1)

        # Add day of year for target image.
        target_doy = self.target_date.timetuple().tm_yday
        target_doy = target_doy * 0.001
        target_doy = np.ones_like(constants[0]) * target_doy

        # Add day of year for image guess date.
        guess_doy = self.guess_date.timetuple().tm_yday
        guess_doy = guess_doy * 0.001
        guess_doy = np.ones_like(constants[0]) * guess_doy

        day_diff = self._get_day_diff()
        day_diff = day_diff * 0.001
        day_diff = np.ones_like(constants[0]) * day_diff

        drought = np.memmap(self.initial_drought, 'int8', 'c')
        drought = np.take(drought, indices, axis=0)

        constants = np.concatenate((constants, target_doy[np.newaxis]))
        constants = np.concatenate((constants, guess_doy[np.newaxis]))
        constants = np.concatenate((constants, day_diff[np.newaxis]))
        constants = np.concatenate((constants, drought[np.newaxis]))

    def _get_init_drought_status(self) -> str:
        match = '.dat' if self.memmap else '.tif'

        # Find the date that is lead_time weeks away from the target USDM image.
        d = os.path.basename(self.target).replace('_USDM' + match, '')
        d = dt.datetime.strptime(d, '%Y%m%d').date()
        d = d - rd.relativedelta(weeks=self.lead_time)

        out = [str(x) for x in pathlib.Path(os.path.dirname(self.target)).glob(str(d).replace('-', '')+'*')]

        assert len(out) == 1, 'No initial USDM images available for this target.'

        return out[0]

    def _get_day_diff(self) -> int:
        """
        Get the number of days between the feature date and the target image date.
        """

        return int(7 * self.lead_time)

    def _get_monthlys(self) -> List[str]:
        """
        Get a list of monthly image paths to use to predict a given target.
        """
        match = 'monthly_mem' if self.memmap else 'monthly'
        suffix = '.dat' if self.memmap else '.tif'

        p = os.path.join(self.in_features, match)
        out = []
        for x in self.monthly_dates:
            x = [img for img in pathlib.Path(p).glob(x + '_*' + suffix)]
            [out.append(str(y)) for y in x]

        assert_complete(self.monthly_dates, out, weekly=False)
        return sorted(out)

    def _get_weeklys(self) -> List[str]:
        """
        Get a list of weekly image paths to use to predict a given target.
        """
        match = 'weekly_mem' if self.memmap else 'weekly'
        suffix = '.dat' if self.memmap else '.tif'

        p = os.path.join(self.in_features, match)
        out = []
        for x in self.weekly_dates:
            x = [img for img in pathlib.Path(p).glob(x + '_*' + suffix)]
            [out.append(str(y)) for y in x]

        assert_complete(self.weekly_dates, out, weekly=True)
        return sorted(out)

    def _get_annuals(self) -> List[str]:
        """
        Get a list of annual image paths to use to predict a given target.
        """
        match = 'annual_mem' if self.memmap else 'annual'
        suffix = '.dat' if self.memmap else '.tif'

        p = os.path.join(self.in_features, match)
        return [str(img) for img in Path(p).glob(self.annual_date + '_*' + suffix)]

    def _get_constants(self) -> List[str]:
        """
        Get constant feature images.
        """
        match = 'constant_mem' if self.memmap else 'constant'

        p = os.path.join(self.in_features, match)
        return sorted([str(img) for img in pathlib.Path(p).iterdir()])
    '''
    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):

        if self.count <= self.pixel_per_img:
            self.count += 1
        else:
            self.count = 0
            self.target_count += 1

        return {'const': torch.tensor(self.constant[idx]),
                'mon': torch.tensor(self.monthly[idx]),
                'week': torch.tensor(self.weekly[idx]),
                'target': self.target[idx]}
