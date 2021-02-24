import argparse
from crdm.utils.GetValidDates import get_valid_dates
from crdm.utils.ImportantVars import MONTHLY_VARS, WEEKLY_VARS, DIMS, LENGTH
import numpy as np
import os
from pathlib import Path
import pickle


def make_temporal_stack(data_dir, out_dir, variable):

    assert variable in ['weekly', 'monthly', 'annual', 'target']

    if variable == 'weekly':
        var_list = WEEKLY_VARS
    elif variable == 'monthly':
        var_list = MONTHLY_VARS
    elif variable == 'target':
        var_list = ['USDM']
    else:
        var_list = ['lc']

    var_list = ['_' + x + '.dat' for x in var_list]

    dates = sorted(get_valid_dates(data_dir, variable))
    files = [x.as_posix() for x in Path(data_dir).glob('*.dat')]

    dt = 'int8' if variable == 'target' else 'float32'

    arr_stack = []

    for date in dates:

        date_arr = []

        for var in var_list:
            f = [x for x in files if date in x and var in x]
            assert len(f) == 1, 'You did something dumb.'
            print(f)
            arr = np.memmap(f[0], dtype=dt)
            assert len(arr) == LENGTH, 'There is a mismatch in memmap length and desired length.'
            date_arr.append(arr)

        date_arr = np.array(date_arr)
        arr_stack.append(date_arr)

    out_arr = np.array(arr_stack)
    out_dict = {'dates': dates,
                'shp': out_arr.shape}

    with open(os.path.join(out_dir, variable+'_info.dat'), 'wb') as pick:
        pickle.dump(out_dict, pick)

    mm = np.memmap(os.path.join(out_dir, variable+'.dat'), dtype='float32', mode='w+', shape=out_arr.shape)
    # Copy .tif array to the memmap.
    mm[:] = out_arr[:]

    # Flush to disk.
    del mm


def make_constant_stack(data_dir, out_dir):

    files = sorted([x.as_posix() for x in Path(data_dir).glob('*.dat')])
    variables = [os.path.basename(x).replace('.dat', '') for x in files]

    assert len(files) == len(variables), 'Something went wrong, all constant features should be uniquely named.'

    arr_stack = []

    for f in files:

        print(f)
        arr = np.memmap(f, dtype='float32')
        assert len(arr) == LENGTH, 'There is a mismatch in memmap length and desired length.'
        arr_stack.append(arr)

    out_arr = np.array(arr_stack)
    out_dict = {'shp': out_arr.shape}

    with open(os.path.join(out_dir, 'constant_info.dat'), 'wb') as pick:
        pickle.dump(out_dict, pick)

    mm = np.memmap(os.path.join(out_dir, 'constant.dat'), dtype='float32', mode='w+', shape=out_arr.shape)
    # Copy .tif array to the memmap.
    mm[:] = out_arr[:]

    # Flush to disk.
    del mm


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Convert individual arrays to one large array.')
    parser.add_argument('-wd', '--week', type=str, help='Path to directory containing weekly images.')
    parser.add_argument('-md', '--month', type=str, help='Path to directory containing monthly images.')
    parser.add_argument('-ad', '--annual', type=str, help='Path to directory containing annual images.')
    parser.add_argument('-cd', '--const', type=str, help='Path to directory containing constant images.')
    parser.add_argument('-td', '--target', type=str, help='Path to directory containing target classes.')
    parser.add_argument('-od', '--out', type=str, help='Path to directory where output data will be written.')

    args = parser.parse_args()

    # Write all datasets out
    make_temporal_stack(data_dir=args.week, out_dir=args.out, variable='weekly')
    make_temporal_stack(data_dir=args.month, out_dir=args.out, variable='monthly')
    make_temporal_stack(data_dir=args.annual, out_dir=args.out, variable='annual')
    make_temporal_stack(data_dir=args.target, out_dir=args.out, variable='target')

    make_constant_stack(data_dir=args.const, out_dir=args.out)
