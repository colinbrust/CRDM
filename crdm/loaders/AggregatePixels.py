import numpy as np
from crdm.loaders.Aggregate import Aggregate
from crdm.utils.ImportantVars import WEEKLY_VARS, MONTHLY_VARS
from typing import List


# make list of tuples where elem[0] is the sequence of features and elem[1] is the output class
# make nxm array for input to LSTM where n is a variable and m is the sequence length (12 months)
# Have a dense layer after the end of the LSTM that incorporates the constant information that doesn't change with time
class PremakeTrainingPixels(Aggregate):

    def make_pixel_stack(self, indices):
        out = []

        vs = WEEKLY_VARS + MONTHLY_VARS
        # Read one variable at a time so that tensors are all formatted the same for training.
        for v in vs:
            filt = sorted([x for x in self.weeklys if v + '.dat' in x])
            tmp = np.array([np.memmap(x, 'float32', 'c') for x in filt])
            out.append(tmp)

        # dim = variable x timestep x location
        out = np.array(out)

        # Slice out only training indices
        out = np.take(out, indices, axis=2)

        return out

    def premake_features(self, indices) -> np.array:
        # Make sure you have pixel indices to slice by.

        weeklys = self.make_pixel_stack(indices)

        # dim = variable x location
        constants = [np.memmap(x, 'float32', 'c') for x in [*self.constants, *self.annuals]]
        constants = np.array(constants)
        constants = np.take(constants, indices, axis=1)

        # Add day of year for image guess date.
        guess_doy = self.guess_date.timetuple().tm_yday
        guess_doy = (guess_doy - 1) / (366 - 1)
        guess_doy = np.ones_like(constants[0]) * guess_doy
        constants = np.concatenate((constants, guess_doy[np.newaxis]))
        constants = np.repeat(np.expand_dims(constants, 1), self.n_weeks, 1)

        mei = self.mei[self.mei.date.isin(self.weekly_dates)].value
        mei = np.expand_dims(mei, -1)
        mei = np.repeat(mei, len(indices), -1)

        drought = np.array([np.memmap(x, 'int8', 'r') for x in self.initial_drought])
        drought = np.take(drought, indices, axis=1)
        drought = 2 * drought / 5 - 1

        weeklys = np.concatenate((weeklys, drought[np.newaxis]))
        weeklys = np.concatenate((weeklys, mei[np.newaxis]))
        weeklys = np.vstack((weeklys, constants))

        targets = np.array([np.memmap(x, 'int8', 'r') for x in self.targets])
        targets = np.take(targets, indices, axis=1)

        return weeklys, targets

    def sample_evenly(self) -> List[int]:

        targs = np.array([np.memmap(x, 'int8', 'r') for x in self.targets])

        indices = []
        for category in [5, 4, 3, 2, 1, 0]:
            locs = list(np.where(np.any(targs == category, axis=0))[0])

            if len(locs) == 0:
                continue
            else:
                locs = list(np.random.choice(np.setdiff1d(locs, indices), self.sample_size))
                indices = indices + locs

        return list(np.unique(indices))
