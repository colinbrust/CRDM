WEEKLY_VARS = ['pr', 'rmax', 'rmin', 'sm-rootzone', 'sm-surface', 'srad', 'tmmn', 'tmmx', 'vpd', 'vs']
MONTHLY_VARS = ['ET', 'gpp']
DIMS = (284, 622)
LENGTH = 176648

holdouts = dict(zip(WEEKLY_VARS+MONTHLY_VARS, range(len(WEEKLY_VARS+MONTHLY_VARS))))
l = len(holdouts)

holdouts['drought'] = l
holdouts['mei'] = l + 1


class ConvergenceError(Exception):
    pass
