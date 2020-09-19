from typing import List
from abc import ABC, abstractmethod
import os
import pathlib
import rasterio as rio
import datetime as dt
import numpy as np
import dateutil.relativedelta as rd
from crdm.loaders.AssertComplete import assert_complete

class Aggregate(ABC):

    def __init__(self, target: str, in_features: str, lead_time: int, n_months: int = 2, **kwargs) -> None:
        """
        :param target: Path to target flash drought image
        :param in_features: Path to directory containing 'monthly', 'constant' and 'annual' subdirectories
            each containing features
        :param lead_time: How many months in advance we should make the drought prediction.
        :param n_months: How many months we should use as context to make the prediction. 
        :param kwargs: If using the 'AggregatePixels' class, you must include 'size' as an arg. 'size' is the number of 
            samples to include for train/test. Notice both train and test size will be 1/2 of the size specified by 'size'. 
        """
        self.target = target
        self.annual_date = os.path.basename(self.target)[:4] + '0101'
        self.in_features = in_features
        self.n_months = n_months
        self.lead_time = lead_time
        self.kwargs = kwargs

        self.dates = self._get_date_list()
        self.annuals = self._get_annuals()
        self.monthlys = self._get_monthlys()
        self.constants = self._get_constants()

        self.stack = None


    def _get_date_list(self) -> List[str]:
        d = os.path.basename(self.target).replace('_USDM.dat', '')
        d = d[:-2] + '01'
        d = dt.datetime.strptime(d, '%Y%m%d').date()

        d = d - rd.relativedelta(months=self.lead_time)

        dates = [str(d - rd.relativedelta(months=x)) for x in range(self.n_months)]
        return [x.replace('-', '') for x in dates]

    def _get_day_diff(self) -> int:

        d_pred = os.path.basename(self.target).replace('_USDM.dat', '')
        d_feat = d_pred[:-2] + '01'

        d_pred = dt.datetime.strptime(d_pred, '%Y%m%d').date()
        d_feat = dt.datetime.strptime(d_feat, '%Y%m%d').date()

        d_feat = d_feat - rd.relativedelta(months=self.lead_time)

        return (d_pred - d_feat).days

    def _get_monthlys(self) -> List[str]:
        p = os.path.join(self.in_features, 'monthly_memmap')
        out = []
        for x in self.dates:
            x = [img for img in pathlib.Path(p).glob(x + '_*.dat')]
            [out.append(str(y)) for y in x]

        assert_complete(self.dates, out)
        return sorted(out)

    def _get_annuals(self) -> List[str]:
        p = os.path.join(self.in_features, 'annual_memmap')
        return [str(img) for img in pathlib.Path(p).glob(self.annual_date + '_*.dat')]

    def _get_constants(self) -> List[str]:
        p = os.path.join(self.in_features, 'constant_memmap')
        return sorted([str(img) for img in pathlib.Path(p).iterdir()])
