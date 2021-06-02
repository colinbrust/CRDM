import matplotlib.pyplot as plt
from numba import njit
from crdm.utils.ImportantVars import DIMS, LENGTH
import numpy as np
from pathlib import Path


@njit
def counter(l, swap=1):
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


def calc_durations(target_dir):
    f_list = sorted([x.as_posix() for x in Path(target_dir).glob('*.dat')])
    targets = np.array([np.memmap(x, dtype='int8') for x in f_list])

    fives = np.where(targets == 5, 1, 0)
    fours = np.where(targets == 4, 1, 0)
    ones = np.where(targets == 1, 1, 0)

    five_out = np.zeros(LENGTH)
    for pixel in range(fives.shape[-1]):
        print(pixel)
        five_out[pixel] = np.nanmedian(counter(np.diff(fives[:, pixel])))

    four_out = np.zeros(LENGTH)
    for pixel in range(fours.shape[-1]):
        print(pixel)
        four_out[pixel] = np.nanmedian(counter(np.diff(fours[:, pixel])))

    one_out = np.zeros(LENGTH)
    for pixel in range(ones.shape[-1]):
        print(pixel)
        one_out[pixel] = np.nanmedian(counter(np.diff(ones[:, pixel]), swap=-1))

out_counts = []
for pixel in range(fours.shape[-1]):
    print(pixel)
    out_counts += counter(np.diff(fours[:, pixel]))

a = np.sum(fours, axis=0)
plt.imshow(a.reshape(DIMS))
plt.colorbar()
plt.close()

five_out = np.nan_to_num(five_out, nan=0)
four_out = np.nan_to_num(four_out, nan=0)
one_out = np.nan_to_num(one_out, nan=0)

plt.imshow(five_out.reshape(DIMS))
plt.title('Max Duration of D4 Drought (Weeks)')
plt.colorbar()

plt.imshow(four_out.reshape(DIMS))
plt.title('Max Duration of D3 Drought (Weeks)')
plt.colorbar()

plt.imshow(one_out.reshape(DIMS))
plt.title('Max Duration Without D0 Drought (Weeks)')
plt.colorbar()