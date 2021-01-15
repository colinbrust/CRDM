import argparse
from crdm.loaders.AggregateTrainingPixels import PremakeTrainingPixels
from crdm.utils.ImportantVars import LENGTH
import glob
import numpy as np
import os


def make_lstm_pixel_ts(target_dir, in_features, lead_time, size, n_months, out_dir, rm_features=False):

    targets = glob.glob(os.path.join(target_dir, '*.dat'))
    targets = [x for x in targets if not('/2015' in x or '/2016' in x)] if rm_features else targets

    weeklys_out = []
    monthlys_out = []
    consts_out = []
    targets_out = []

    for target in targets:
        
        try:
            # For each image, get new random indices so we don't overfit to certain locations
            indices = np.random.choice(LENGTH, size=size, replace=False)
            agg = PremakeTrainingPixels(target=target, in_features=in_features, lead_time=lead_time, n_months=n_months, indices=indices)

            if rm_features:
                agg.remove_lat_lon()

            print(target)

            # Make monthly, constant, and target arrays
            weeklys, monthlys, constants = agg.premake_features()
            target = np.memmap(target, 'int8', 'c')
            target = target[indices]

            weeklys_out.append(weeklys)
            monthlys_out.append(monthlys)
            consts_out.append(constants)
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
    basename = '_trainingType-pixelPremade_nMonths-{}_leadTime-{}_size-{}_rmFeatures-{}.dat'.format(n_months, lead_time, size, rm_features)


    for prefix, arr in list(zip(['featType-monthly', 'featType-constant', 'featType-target'], [arrs_out, consts_out, targets_out])):

        if prefix == 'featType-target':
            mm = np.memmap(os.path.join(out_dir, prefix+basename), dtype='int8', mode='w+', shape=arr.shape)
        else:
            mm = np.memmap(os.path.join(out_dir, prefix+basename), dtype='float32', mode='w+', shape=arr.shape)
            
        # Copy .tif array to the memmap.
        mm[:] = arr[:]

        # Flush to disk.
        del mm
    
    return basename


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Make training data for LSTM drought model.')

    parser.add_argument('-td', '--target_dir', type=str, help='Directory containing target memmap images.')
    parser.add_argument('-if', '--in_features', type=str, help='Direcotry containing directorys with memmaps of training features')
    parser.add_argument('-lt', '--lead_time', type=int, help='Number of months in advance to make predictions.')
    parser.add_argument('-nm', '--n_months', type=int, help='Number of months "history" to use as model inputs.')
    parser.add_argument('-sz', '--size', type=int, help='Number of pixels to use to train model.')
    parser.add_argument('-od', '--out_dir', type=str, help='Directory to put new files into.')
    parser.add_argument('--rm', dest='remove', action='store_true', help='Remove lat/lon inputs and years 2015 and 2016.')
    parser.add_argument('--no-rm', dest='remove', action='store_false', help='Do not remove any training features')
    parser.set_defaults(remove=False)

    args = parser.parse_args()

    make_lstm_pixel_ts(target_dir=args.target_dir, in_features=args.in_features, lead_time=args.lead_time,
                       n_months=args.n_months, size=args.size, out_dir=args.out_dir, rm_features=args.remove)