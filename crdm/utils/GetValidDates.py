from crdm.utils.ImportantVars import WEEKLY_VARS, MONTHLY_VARS
import os
from pathlib import Path
from typing import List


def get_valid_dates(weekly_dir, variable) -> List[str]:

    assert variable in ['weekly', 'monthly', 'annual', 'target'], "Type argument must be one of 'weekly', 'annual', " \
                                                                   "'target', or 'monthly' "

    image_list = []

    if variable == 'weekly':
        var_list = WEEKLY_VARS
    elif variable == 'monthly':
        var_list = MONTHLY_VARS
    elif variable == 'target':
        var_list = ['USDM']
    else:
        var_list = ['lc']

    # get a list of weekly dates available for each variable
    for var in var_list:
        f_list = [os.path.basename(str(x)) for x in Path(weekly_dir).glob('*' + var + '.dat')]
        f_list = [x[:8] for x in f_list]
        image_list.append(f_list)

    # return a list of dates where data are available for all variables
    return list(sorted(set.intersection(*[set(ls) for ls in image_list])))