import numpy as np
import pickle
import glob
import os
from crdm.loaders.Aggregate import Aggregate
from crdm.utils.ImportantVars import LENGTH, WEEKLY_VARS, MONTHLY_VARS
from crdm.utils.ParseFileNames import parse_fname


# make list of tuples where elem[0] is the sequence of features and elem[1] is the output class
# make nxm array for input to LSTM where n is a variable and m is the sequence length (12 months)
# Have a dense layer after the end of the LSTM that incorporates the constant information that doesn't change with time
class PremakeTrainingPixels(Aggregate):

    def make_pixel_stack(self, indices, weekly):
        out = []

        VARIABLES = WEEKLY_VARS if weekly else MONTHLY_VARS
        images = self.weeklys if weekly else self.monthlys

        # Read one variable at a time so that tensors are all formatted the same for training.
        for v in VARIABLES:
            filt = sorted([x for x in images if v + '.dat' in x])
            tmp = np.array([np.memmap(x, 'float32', 'c') for x in filt])
            out.append(tmp)

        # dim = variable x timestep x location
        out = np.array(out)

        # Slice out only training indices
        out = np.take(out, indices, axis=2)

        return out

    def premake_features(self) -> np.array:
        # Make sure you have pixel indices to slice by.
        assert 'indices' in self.kwargs, "'indices' must be included as a kwarg when instantiating the " \
                                         "AggregatePixels class. "
        assert len(self.kwargs['indices']) <= LENGTH, "'indices' must be smaller than 161040 (the number of 9km " \
                                                      "pixels in the CONUS domain). "

        indices = self.kwargs['indices']

        weeklys = self.make_pixel_stack(indices, True)
        monthlys = self.make_pixel_stack(indices, False)

        # dim = variable x location
        constants = [np.memmap(x, 'float32', 'c') for x in [*self.constants, *self.annuals]]
        constants = np.array(constants)
        constants = np.take(constants, indices, axis=1)

        target_doy = self.target_date.timetuple().tm_yday
        target_doy = (target_doy - 1) / (366 - 1)
        target_doy_week = np.ones_like(weeklys[0]) * target_doy
        target_doy = np.ones_like(constants[0]) * target_doy

        # Add day of year for image guess date.
        guess_doy = self.guess_date.timetuple().tm_yday
        guess_doy = (guess_doy - 1) / (366 - 1)
        guess_doy_week = np.ones_like(weeklys[0]) * guess_doy
        guess_doy = np.ones_like(constants[0]) * guess_doy

        day_diff = self._get_day_diff()
        day_diff = (day_diff - 7) / (84 - 7)
        day_diff_week = np.ones_like(weeklys[0]) * day_diff
        day_diff = np.ones_like(constants[0]) * day_diff

        constants = np.concatenate((constants, target_doy[np.newaxis]))
        constants = np.concatenate((constants, guess_doy[np.newaxis]))
        constants = np.concatenate((constants, day_diff[np.newaxis]))

        try:
            if self.kwargs['init']:

                drought = np.array([np.memmap(x, 'int8', 'c') for x in self.initial_drought])
                drought = np.take(drought, indices, axis=1)
                # Scale between -1 and 1
                drought = 2 * drought/5 - 1
                weeklys = np.concatenate((weeklys, drought[np.newaxis]))

            else:
                print('Not including initial drought state.')

        except KeyError:
            drought = np.array([np.memmap(x, 'int8', 'c') for x in self.initial_drought])
            drought = np.take(drought, indices, axis=1)
            # Scale between -1 and 1
            drought = 2 * drought / 5 - 1
            weeklys = np.concatenate((weeklys, drought[np.newaxis]))

        weeklys = np.vstack((weeklys, target_doy_week[np.newaxis]))
        weeklys = np.vstack((weeklys, guess_doy_week[np.newaxis]))
        weeklys = np.vstack((weeklys, day_diff_week[np.newaxis]))

        return weeklys, monthlys, constants

