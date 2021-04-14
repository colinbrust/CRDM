from abc import ABC
from crdm.loaders.AssertComplete import assert_complete
import datetime as dt
import dateutil.relativedelta as rd
import os
import pathlib
import pandas as pd
from typing import List


class Aggregate(ABC):
    """
    Class for aggregating all images necessary to make predictions for a given USDM image.
    """
    def __init__(self, targets: List[str], in_features: str, mei: str, n_weeks: int = 17, **kwargs) -> None:
        """
        :param target: Path to target flash drought image :param in_features: Path to directory containing 'monthly',
        'constant' and 'annual' subdirectories each containing features :param lead_time: How many months in advance
        we should make the drought prediction. :param n_months: How many months we should use as context to make the
        prediction. :param kwargs: If using the 'AggregatePixels' class, you must include 'size' as an arg. 'size' is
        the number of samples to include for train/test. Notice both train and test size will be 1/2 of the size
        specified by 'size'.
        """
        self.targets = targets
        self.annual_date = os.path.basename(self.targets[0])[:4] + '0101'
        self.in_features = in_features
        self.n_weeks = n_weeks
        self.kwargs = kwargs
        self.mei = pd.read_csv(os.path.join(in_features, 'mei.csv'))

        self.weekly_dates = self._get_date_list()
        self.annuals = self._get_annuals()
        self.weeklys = self._get_weeklys()
        self.constants = self._get_constants()
        self.initial_drought = self._get_init_drought_status()

        self.stack = None

    def _get_date_list(self) -> List[str]:
        """
        Get a list of feature dates to use given the target image.
        """

        # Find the date that is lead_time weeks away from the target USDM image.
        d = os.path.basename(self.targets[0]).replace('_USDM.dat', '')
        d = dt.datetime.strptime(d, '%Y%m%d').date()
        d = d - rd.relativedelta(weeks=1)
        self.guess_date = d

        # Find the input feature image dates for weekly and monthly features.
        dates = [str(d - rd.relativedelta(weeks=x)) for x in range(self.n_weeks)]
        dates = [x.replace('-', '') for x in dates]
        return dates

    def _get_init_drought_status(self) -> List[str]:

        p = pathlib.Path(os.path.dirname(self.targets[0]))
        out = []

        for x in self.weekly_dates:
            x = next(p.glob(x+'*'))
            out.append(str(x))

        return sorted(out)

    def _get_weeklys(self) -> List[str]:
        """
        Get a list of weekly image paths to use to predict a given target.
        """
        match = 'weekly_mem'
        suffix = '.dat'

        p = os.path.join(self.in_features, match)
        out = []
        for x in self.weekly_dates:
            x = [img for img in pathlib.Path(p).glob(x + '_*' + suffix)]
            [out.append(str(y)) for y in x]

        assert_complete(self.weekly_dates, out)
        return sorted(out)

    def _get_annuals(self) -> List[str]:
        """
        Get a list of annual image paths to use to predict a given target.
        """
        match = 'annual_mem'
        suffix = '.dat'

        p = os.path.join(self.in_features, match)
        return [str(img) for img in pathlib.Path(p).glob(self.annual_date + '_*' + suffix)]

    def _get_constants(self) -> List[str]:
        """
        Get constant feature images.
        """
        match = 'constant_mem'

        p = os.path.join(self.in_features, match)
        return sorted([str(img) for img in pathlib.Path(p).iterdir()])
