import numpy as np
import pickle
import glob
import os
from crdm.loaders.Aggregate import Aggregate
from crdm.utils.ImportantVars import VARIABLES


# make list of tuples where elem[0] is the sequence of features and elem[1] is the output class
# make nxm array for input to LSTM where n is a variable and m is the sequence length (12 months)
# Have a dense layer after the end of the LSTM that incorporates the constant information that doesn't change with time
class PremakeTrainingPixels(Aggregate):

    def premake_features(self) -> np.array:
        # Make sure you have pixel indices to slice by.
        assert 'indices' in self.kwargs, "'indices' must be included as a kwarg when instantiating the AggregatePixels class."
        assert len(self.kwargs['indices']) <= 161040, "'indices' must be smaller than 161040 (the number of 9km pixels in the CONUS domain)."

        indices = self.kwargs['indices']
        arrs = []

        # Read one variable at a time so that tensors are all formatted the same for training.
        for v in VARIABLES:
            filt = sorted([x for x in self.monthlys if v+'.dat' in x])
            tmp = np.array([np.memmap(x, 'float32', 'c') for x in filt])
            arrs.append(tmp)

        # dim = variable x timestep x location
        arrs = np.array(arrs)
        # Slice out only training indices
        arrs = np.take(arrs, indices, axis=2)

        # dim = variable x location
        constants = [np.memmap(x, 'float32', 'c') for x in [*self.constants, *self.annuals]]
        constants = np.array(constants)
        constants = np.take(constants, indices, axis=1)

        # Add month 
        month = int(os.path.basename(self.target)[4:6])
        month = month * 0.01
        month = np.ones_like(constants[0]) * month

        day_diff = self._get_day_diff()
        day_diff = day_diff * 0.01
        day_diff = np.ones_like(constants[0]) * day_diff

        constants = np.concatenate((constants, month[np.newaxis]))
        constants = np.concatenate((constants, day_diff[np.newaxis]))

        return arrs, constants

# If it turns out we need more training data, rerun this with a size > 1000
def make_pixel_ts(target_dir, in_features, lead_time, size, n_months, out_dir):

    targets = glob.glob(os.path.join(target_dir, '*.dat'))

    arrs_out = []
    consts_out = []
    targets_out = []

    for target in targets:
        
        try:
            # For each image, get new random indices so we don't overfit to certain locations
            indices = np.random.choice(161040, size=size, replace=False)
            agg = PremakeTrainingPixels(target=target, in_features=in_features, lead_time=lead_time, n_months=n_months, indices=indices)
            print(target)

            # Make monthly, constant, and target arrays
            arrs, consts = agg.premake_features()
            target = np.memmap(target, 'int8', 'c')
            target = target[indices]

            arrs_out.append(arrs)
            consts_out.append(consts)
            targets_out.append(target)

        except AssertionError as e:
            print('{}\n Skipping {}'.format(e, target))
    
    # Flatten and reorder all the axes
    arrs_out = np.concatenate([*arrs_out], axis=2)
    arrs_out = np.swapaxes(arrs_out, 0, 2)

    consts_out = np.concatenate([*consts_out], axis=1)
    consts_out = np.swapaxes(consts_out, 0, 1)

    targets_out = np.concatenate([*targets_out])

    # Write out training data to numpy memmaps.
    basename = '_pixelPremade_nMonths-{}_leadTime-{}_size-{}.dat'.format(n_months, lead_time, size)

    for prefix, arr in list(zip(['monthly', 'constant', 'target'], [arrs_out, consts_out, targets_out])):

        if prefix == 'target':
            mm = np.memmap(os.path.join(out_dir, prefix+basename), dtype='int8', mode='w+', shape=arr.shape)
        else:
            mm = np.memmap(os.path.join(out_dir, prefix+basename), dtype='float32', mode='w+', shape=arr.shape)
            
        # Copy .tif array to the memmap.
        mm[:] = arr[:]

        # Flush to disk.
        del mm

