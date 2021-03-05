import argparse
from crdm.loaders.AggregateTrainingPixels import PremakeTrainingPixels
from crdm.utils.ImportantVars import LENGTH
import glob
import numpy as np
import os
import pickle


def make_lstm_pixel_ts(target_dir, in_features, size, n_weeks, out_dir, rm_years=False, init=True):

    targets = glob.glob(os.path.join(target_dir, '*.dat'))
    targets = [x for x in targets if not ('/2015' in x or '/2017' in x or '/2007' in x)] if rm_years else targets
    targets = sorted(targets)
    weeklys_out = []
    monthlys_out = []
    consts_out = []
    targets_out = []

    for target in targets:

        try:
            # For each image, get new random indices so we don't overfit to certain locations
            indices = np.random.choice(LENGTH, size=size, replace=False)

            for lead_time in [2, 4, 6, 8]:

                agg = PremakeTrainingPixels(target=target, in_features=in_features, lead_time=lead_time, n_weeks=n_weeks,
                                            indices=indices, memmap=True, init=init)
                print(lead_time, target)

                # Make monthly, constant, and target arrays
                weeklys, monthlys, consts = agg.premake_features()
                target = np.memmap(target, 'int8', 'c')
                target = target[indices]

                assert consts.shape == (20, 16384)
                weeklys_out.append(weeklys)
                monthlys_out.append(monthlys)
                consts_out.append(consts)
                targets_out.append(target)

        except AssertionError as e:
            print('{}\n Skipping {}'.format(e, target))

    # Flatten and reorder all the axes
    weeklys_out = np.concatenate([*weeklys_out], axis=2)
    weeklys_out = np.swapaxes(weeklys_out, 0, 2)

    monthlys_out = np.concatenate([*monthlys_out], axis=2)
    monthlys_out = np.swapaxes(monthlys_out, 0, 2)

    consts_out = np.concatenate([*consts_out], axis=1)
    consts_out = np.swapaxes(consts_out, 0, 1)

    targets_out = np.concatenate([*targets_out])

    # Write out training data to numpy memmaps.
    basename = '_trainingType-pixelPremade_nWeeks-{}_leadTime-{}_size-{}_rmYears-{}_init-{}.dat'.format(n_weeks, 'all', size, rm_years, init)

    pick = {}
    for prefix, arr in list(
            zip(['featType-weekly', 'featType-monthly', 'featType-constant', 'featType-target'],
                [weeklys_out, monthlys_out, consts_out, targets_out])):

        out_name = os.path.join(out_dir, prefix + basename)
        pick[prefix] = prefix + basename

        if prefix == 'featType-target':
            mm = np.memmap(out_name, dtype='int8', mode='w+', shape=arr.shape)
        else:
            mm = np.memmap(out_name, dtype='float32', mode='w+', shape=arr.shape)

        # Copy .tif array to the memmap.
        mm[:] = arr[:]

        # Flush to disk.
        del mm

    pick_name = os.path.join(out_dir, 'pickle' + basename)
    out = open(pick_name, 'wb')
    pickle.dump(pick, out)
    out.close()

    return basename


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make training data for LSTM drought model.')

    parser.add_argument('-td', '--target_dir', type=str, help='Directory containing target memmap images.')
    parser.add_argument('-if', '--in_features', type=str,
                        help='Directory containing directorys with memmaps of training features')
    parser.add_argument('-nw', '--n_weeks', type=int, help='Number of week "history" to use as model inputs.')
    parser.add_argument('-sz', '--size', type=int, help='Number of pixels to use to train model.')
    parser.add_argument('-od', '--out_dir', type=str, help='Directory to put new files into.')

    args = parser.parse_args()

    make_lstm_pixel_ts(target_dir=args.target_dir, in_features=args.in_features,
                       n_weeks=args.n_weeks, size=args.size, out_dir=args.out_dir, rm_years=True, init=True)
