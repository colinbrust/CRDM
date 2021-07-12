import datetime as dt
from dateutil import relativedelta as rd

# List of all model features
WEEKLY_VARS = ['pr', 'rmax', 'rmin', 'sm-rootzone', 'sm-surface', 'srad', 'tmmn', 'tmmx', 'vpd', 'vs']
MONTHLY_VARS = ['ET', 'gpp']

# Dimensions of all training/classification images
DIMS = (284, 622)
LENGTH = 176648

# Dict of the order of each variable
holdouts = dict(zip(WEEKLY_VARS+MONTHLY_VARS, range(len(WEEKLY_VARS+MONTHLY_VARS))))
l = len(holdouts)
holdouts['drought'] = l
holdouts['mei'] = l + 1

_d = dt.date(2000, 1, 4)
_today = dt.date.today()

usdm_dates = []
while _d < _today:
    usdm_dates.append(_d)
    _d = _d + rd.relativedelta(days=7)


# Exception class to use if model doesn't converge
class ConvergenceError(Exception):
    pass

