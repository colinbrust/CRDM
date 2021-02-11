import argparse
from crdm.loaders.AggregateTrainingPixels import PremakeTrainingPixels
from crdm.utils.ImportantVars import LENGTH
import glob
import numpy as np
import os
import pickle


def make_lstm_pixel_ts(target_dir, in_features, size, n_weeks, out_dir, rm_years=False, init=True):

    targets_tmp = sorted(glob.glob(os.path.join(target_dir, '*.dat')))
    targets = []
    for i in range(len(targets_tmp)):
        if rm_years:
            files = targets_tmp[i:i + 8]
            if any(['/2015' in x or '/2017' in x for x in files]):
                continue
            else:
                targets.append(files)
        else:
            targets.append(targets_tmp[i:i + 8])
    weeklys_out = []
    monthlys_out = []
    consts_out = []
    targets_out = []

    for target in targets:

        try:
            # For each image, get new random indices so we don't overfit to certain locations
            indices = np.random.choice(LENGTH, size=size, replace=False)
            agg = PremakeTrainingPixels(targets=target, in_features=in_features, n_weeks=n_weeks,
                                        indices=indices, memmap=True, init=init)
            print(target[0])

            # Make monthly, constant, and target arrays
            weeklys, monthlys, consts = agg.premake_features()

            imgs = np.array([np.memmap(x, 'int8', 'c') for x in target])
            imgs = np.take(imgs, indices, axis=1)

            weeklys_out.append(weeklys)
            monthlys_out.append(monthlys)
            consts_out.append(consts)
            targets_out.append(imgs)

        except AssertionError as e:
            print('{}\n Skipping {}'.format(e, target))

    # Flatten and reorder all the axes
    weeklys_out = np.concatenate([*weeklys_out], axis=2)
    weeklys_out = np.swapaxes(weeklys_out, 0, 2)

    monthlys_out = np.concatenate([*monthlys_out], axis=2)
    monthlys_out = np.swapaxes(monthlys_out, 0, 2)

    consts_out = np.concatenate([*consts_out], axis=1)
    consts_out = np.swapaxes(consts_out, 0, 1)

    targets_out = np.concatenate([*targets_out], axis=1)
    targets_out = np.swapaxes(targets_out, 0, 1)
    targets_out = targets_out/5

    # Write out training data to numpy memmaps.
    basename = '_trainingType-pixelPremade_nWeeks-{}_size-{}_rmYears-{}_init-{}.dat'.format(n_weeks, size, rm_years, init)

    pick = {}
    for prefix, arr in list(
            zip(['featType-weekly', 'featType-monthly', 'featType-constant', 'featType-target'],
                [weeklys_out, monthlys_out, consts_out, targets_out])):

        out_name = os.path.join(out_dir, prefix + basename)
        pick[prefix] = prefix + basename

        mm = np.memmap(out_name, dtype='float32', mode='w+', shape=arr.shape)

        # Copy .tif array to the memmap.
        mm[:] = arr[:]

        # Flush to disk.
        del mm

    pick_name = os.path.join(out_dir, 'pickle' + basename)
    out = open(pick_name, 'wb')
    pickle.dump(pick, out)
    out.close()

    return pick_name


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make training data for LSTM drought model.')

    parser.add_argument('-td', '--target_dir', type=str, help='Directory containing target memmap images.')
    parser.add_argument('-if', '--in_features', type=str,
                        help='Directory containing directorys with memmaps of training features')
    parser.add_argument('-nw', '--n_weeks', type=int, help='Number of week "history" to use as model inputs.')
    parser.add_argument('-sz', '--size', type=int, help='Number of pixels to use to train model.')
    parser.add_argument('-od', '--out_dir', type=str, help='Directory to put new files into.')
    parser.add_argument('--rm', dest='remove', action='store_true', help='Remove data from years 2015 and 2017.')
    parser.add_argument('--no-rm', dest='remove', action='store_false', help='Do not remove any training features')

    parser.add_argument('--init', dest='init', action='store_true', help='Use initial drought condition as model input.')
    parser.add_argument('--no-init', dest='init', action='store_false', help='Do not use initial drought condition as model input..')
    parser.set_defaults(init=True)

    args = parser.parse_args()

    make_lstm_pixel_ts(target_dir=args.target_dir, in_features=args.in_features, n_weeks=args.n_weeks,
                       size=args.size, out_dir=args.out_dir, rm_years=args.remove, init=args.init)
