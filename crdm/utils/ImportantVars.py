WEEKLY_VARS = ['pr', 'rmax', 'rmin', 'sm-rootzone', 'sm-surface', 'srad', 'tmmn', 'tmmx', 'vpd', 'vs'] # 'fw', 'vod', 'vapor'
MONTHLY_VARS = ['ET', 'gpp']
DIMS = (284, 622)
LENGTH = 176648

holdouts = dict(zip(WEEKLY_VARS+MONTHLY_VARS, range(len(WEEKLY_VARS+MONTHLY_VARS))))
holdouts['drought'] = len(holdouts) 
holdouts['mei'] = len(holdouts) + 1
