from crdm.utils.ImportantVars import WEEKLY_VARS
import os
from pathlib import Path
from typing import List


def find_missing_dates(weekly_dir) -> List[str]:

    image_list = []

    # get a list of weekly dates available for each variable
    for var in WEEKLY_VARS:
        f_list = [os.path.basename(str(x)) for x in Path(weekly_dir).glob('*' + var + '.dat')]
        f_list = [x[:8] for x in f_list]
        image_list.append(f_list)

    # return a list of dates where data are available for all variables
    return list(sorted(set.intersection(*[set(ls) for ls in image_list])))
