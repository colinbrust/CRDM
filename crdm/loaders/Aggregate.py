from abc import ABC
from crdm.loaders.AssertComplete import assert_complete
import datetime as dt
import dateutil.relativedelta as rd
import os
import pathlib
from typing import List


class Aggregate(ABC):
    """
    Class for aggregating all images necessary to make predictions for a given USDM image.
    """
    def __init__(self, target: str, in_features: str, lead_time: int, n_weeks: int = 17, memmap: bool = True,
                 **kwargs) -> None:
        """
        :param target: Path to target flash drought image :param in_features: Path to directory containing 'monthly',
        'constant' and 'annual' subdirectories each containing features :param lead_time: How many months in advance
        we should make the drought prediction. :param n_months: How many months we should use as context to make the
        prediction. :param kwargs: If using the 'AggregatePixels' class, you must include 'size' as an arg. 'size' is
        the number of samples to include for train/test. Notice both train and test size will be 1/2 of the size
        specified by 'size'.
        """
        self.target = target
        self.annual_date = os.path.basename(self.target)[:4] + '0101'
        self.in_features = in_features
        self.n_weeks = n_weeks
        self.n_months = n_weeks // 4
        self.lead_time = lead_time
        self.memmap = memmap
        self.kwargs = kwargs

        self.weekly_dates, self.monthly_dates = self._get_date_list()
        self.annuals = self._get_annuals()
        self.monthlys = self._get_monthlys()
        self.constants = self._get_constants()

        self.stack = None

    def _get_date_list(self) -> List[str]:
        """
        Get a list of feature dates to use given the target image. 
        """
        match = '.dat' if self.memmap else '.tif'
        d = os.path.basename(self.target).replace('_USDM' + match, '')
        d = dt.datetime.strptime(d, '%Y%m%d').date()

        d = d - rd.relativedelta(weeks=self.lead_time)

        dates = [str((d - rd.relativedelta(weeks=x)) - rd.relativedelta(days=1)) for x in range(self.n_weeks)]
        start_month = dt.datetime.strptime(dates[0][:-2] + '01', '%Y-%m-%d').date()

        months = [str(start_month - rd.relativedelta(months=x)) for x in range(self.n_months)]

        dates = [x.replace('-', '') for x in dates]
        months = [x.replace('-', '') for x in months]

        return dates, months

    def _get_day_diff(self) -> int:
        """
        Get the number of days between the feature date and the target image date. 
        """

        return int(7 * self.lead_time)

    def _get_monthlys(self) -> List[str]:
        """
        Get a list of monthly image paths to use to predict a given target. 
        """
        match = 'monthly_memmap' if self.memmap else 'monthly'
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
        match = 'weekly_memmap' if self.memmap else 'weekly'
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
        match = 'annual_memmap' if self.memmap else 'annual'
        suffix = '.dat' if self.memmap else '.tif'

        p = os.path.join(self.in_features, match)
        return [str(img) for img in pathlib.Path(p).glob(self.annual_date + '_*' + suffix)]

    def _get_constants(self) -> List[str]:
        """
        Get constant feature images. 
        """
        match = 'constant_memmap' if self.memmap else 'constant'

        p = os.path.join(self.in_features, match)
        return sorted([str(img) for img in pathlib.Path(p).iterdir()])
