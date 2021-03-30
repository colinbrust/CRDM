import argparse
from crdm.utils.ImportantVars import MONTHLY_VARS, WEEKLY_VARS
import numpy as np
import os
from pathlib import Path


def save_arrays(target_dir, feature_dir, const_dir, out_dir, variable):

    targets = sorted(Path(target_dir).iterdir())
    fill = np.memmap(list(Path(const_dir).glob(variable+'*'))[0], dtype='float32')

    out = []
    for target in targets:
        print(variable, target)
        date = target.stem.split('_')[0]
        feat = Path(os.path.join(feature_dir, date+'_'+variable+'.dat'))

        if feat.exists():
            arr = np.memmap(feat, dtype='float32')
            out.append(arr)
        else:
            # Fill with the climatological mean if the data are missing.
            out.append(fill)

    arr_out = np.array(out)
    mm = np.memmap(os.path.join(out_dir, variable+'.dat'), dtype='float32', mode='w+', shape=arr_out.shape)
    mm[:] = arr_out[:]

    # Flush to disk.
    del mm


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Convert raster timeseries to a stacked memmap')

    parser.add_argument('-t', '--target_dir', type=str, help='Path to directory with complete timeseries.')
    parser.add_argument('-f', '--feature_dir', type=str, help='Path to directory with the training features.')
    parser.add_argument('-c', '--const_dir', type=str, help='Path to directory with climatological means of vairables.')
    parser.add_argument('-o', '--out_dir', type=str, help='Path to directory that new files will be written to.')

    args = parser.parse_args()

    for variable in MONTHLY_VARS + WEEKLY_VARS:
        save_arrays(args.target_dir, args.feature_dir, args.const_dir, args.out_dir, variable)

