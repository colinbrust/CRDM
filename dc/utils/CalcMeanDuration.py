import matplotlib.pyplot as plt
from numba import njit
from dc.utils.ImportantVars import DIMS, LENGTH
import numpy as np
from pathlib import Path
from typing import List, Dict


@njit
def counter(l: np.array, swap: int = 1) -> List[int]:
    """
    :param l: np.array containing timeseries of -1, 0 or 1 for a given pixel.
    :param swap: Int of either -1 or 1.
    :return: List of ints representing sequential number of weeks of the same drought category.
    """
    out = []
    c = 0
    switch = False
    for i in l:
        if i == 1*swap:
            switch = True
        if switch:
            c += 1
        if i == -1*swap:
            switch = False
            out.append(c)
            c = 0
    return out


def calc_durations(target_dir: str) -> Dict[np.array]:
    """
    Calculate mean duration of drought in various categories.
    :param target_dir: Path to directory containing memmaps of USDM images.
    :return: Dictionary containing durations of category 3, 4 drought and time without a category 0 drought.
    """
    f_list = sorted([x.as_posix() for x in Path(target_dir).glob('*.dat')])
    targets = np.array([np.memmap(x, dtype='int8') for x in f_list])

    fives = np.where(targets == 5, 1, 0)
    fours = np.where(targets == 4, 1, 0)
    ones = np.where(targets == 1, 1, 0)

    # Count how long the average duration of a D3 or D4 drought is.
    five_out = np.zeros(LENGTH)
    for pixel in range(fives.shape[-1]):
        print(pixel)
        five_out[pixel] = np.nanmedian(counter(np.diff(fives[:, pixel])))
    four_out = np.zeros(LENGTH)
    for pixel in range(fours.shape[-1]):
        print(pixel)
        four_out[pixel] = np.nanmedian(counter(np.diff(fours[:, pixel])))

    # Count how average duration of time without a category D0 drought.
    one_out = np.zeros(LENGTH)
    for pixel in range(ones.shape[-1]):
        print(pixel)
        one_out[pixel] = np.nanmedian(counter(np.diff(ones[:, pixel]), swap=-1))

    return {'fives': five_out,
            'fours': four_out,
            'ones': one_out}

# out_counts = []
# for pixel in range(fours.shape[-1]):
#     print(pixel)
#     out_counts += counter(np.diff(fours[:, pixel]))
#
# a = np.sum(fours, axis=0)
# plt.imshow(a.reshape(DIMS))
# plt.colorbar()
# plt.close()
#
# five_out = np.nan_to_num(five_out, nan=0)
# four_out = np.nan_to_num(four_out, nan=0)
# one_out = np.nan_to_num(one_out, nan=0)
#
# plt.imshow(five_out.reshape(DIMS))
# plt.title('Max Duration of D4 Drought (Weeks)')
# plt.colorbar()
#
# plt.imshow(four_out.reshape(DIMS))
# plt.title('Max Duration of D3 Drought (Weeks)')
# plt.colorbar()
#
# plt.imshow(one_out.reshape(DIMS))
# plt.title('Max Duration Without D0 Drought (Weeks)')
# plt.colorbar()