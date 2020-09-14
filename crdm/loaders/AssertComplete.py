import os

# VARIABLES = ['aswi', 'ET', 'fw', 'gpp', 'pr', 'rmax', 'rmin', 'sm-rootzone', 'sm-surface', 'srad', 'tmmn', 'tmmx',
#              'vapor', 'vod', 'vpd', 'vs']

VARIABLES = ['ET', 'gpp', 'pr', 'rmax', 'rmin', 'sm-rootzone', 'sm-surface', 'srad', 'tmmn', 'tmmx', 'vpd', 'vs']

def assert_complete(dates, features):

    for v in VARIABLES:
        filt = [x for x in features if v in x]

        for d in dates:
            assert sum([d in x for x in filt]) > 0, '{}_{}.dat is not present.'.format(d, v)