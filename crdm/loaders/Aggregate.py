from typing import List
from abc import ABC, abstractmethod
import os
import pathlib
import rasterio as rio
import datetime as dt
import numpy as np
import dateutil.relativedelta as rd
from paper2.classification.AssertComplete import assert_complete


class AggregateFeatures(ABC):

    def __init__(self, target: str, in_features: str, lead_time: int, n_months: int = 2) -> None:
        """
        :param target: Path to target flash drought image
        :param in_features: Path to directory containing 'monthly', 'constant' and 'annual' subdirectories
            each containing features
        """
        self.target = target
        self.annual_date = os.path.basename(self.target)[:4] + '0101'
        self.in_features = in_features
        self.n_months = n_months
        self.lead_time = lead_time

        self.dates = self._get_date_list()
        self.annuals = self._get_annuals()
        self.monthlys = self._get_monthlys()
        self.constants = self._get_constants()

        self.stack = None
        # self.stack = self.make_feature_stack()

    def _get_date_list(self) -> List[str]:
        d = os.path.basename(self.target).replace('_USDM.tif', '')
        d = d[:-2] + '01'
        d = dt.datetime.strptime(d, '%Y%m%d').date()

        d = d - rd.relativedelta(months=self.lead_time)

        dates = [str(d - rd.relativedelta(months=x)) for x in range(self.n_months)]
        return [x.replace('-', '') for x in dates]

    def _get_day_diff(self) -> int:

        d_pred = os.path.basename(self.target).replace('_USDM.tif', '')
        d_feat = d_pred[:-2] + '01'

        d_pred = dt.datetime.strptime(d_pred, '%Y%m%d').date()
        d_feat = dt.datetime.strptime(d_feat, '%Y%m%d').date()

        d_feat = d_feat - rd.relativedelta(months=self.lead_time)

        return (d_pred - d_feat).days

    def _get_monthlys(self) -> List[str]:
        p = os.path.join(self.in_features, 'monthly')
        out = []
        for x in self.dates:
            x = [img for img in pathlib.Path(p).glob(x + '_*.tif')]
            [out.append(str(y)) for y in x]

        assert_complete(self.dates, out)
        return sorted(out)

    def _get_annuals(self) -> List[str]:
        p = os.path.join(self.in_features, 'annual')
        return [str(img) for img in pathlib.Path(p).glob(self.annual_date + '_*.tif')]

    def _get_constants(self) -> List[str]:
        p = os.path.join(self.in_features, 'constant')
        return sorted([str(img) for img in pathlib.Path(p).iterdir()])

    @staticmethod
    def _read_img(img) -> np.array:
        src = rio.open(img)
        arr = src.read(1)
        arr = np.where(arr <= -9999, np.nan, arr)
        return arr

    def make_feature_stack(self) -> np.array:
        stack = [*self.monthlys, *self.annuals, *self.constants]
        stack = list(map(self._read_img, stack))

        template = np.ones_like(stack[0])
        month = int(os.path.basename(self.target)[4:6])
        month = template * month * 0.001
        stack.append(month)

        day_diff = self._get_day_diff()
        day_diff = template * day_diff * 0.001
        stack.append(day_diff)

        self.stack = np.array(stack)

    def get_features(self):
        return self.stack
    
    def get_target(self):
        src = rio.open(self.target)
        return src.read(1)


    def get_features_and_target(self):

        return self.get_features(), self.get_target()
