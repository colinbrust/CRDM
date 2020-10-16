import numpy as np
import pickle
import glob
import os
from crdm.loaders.Aggregate import Aggregate
from crdm.utils.ImportantVars import VARIABLES
from crdm.utils.ParseFileNames import parse_fname


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

        match = '.dat' if self.memmap else '.tif'

        # Read one variable at a time so that tensors are all formatted the same for training.
        for v in VARIABLES:
            filt = sorted([x for x in self.monthlys if v+match in x])
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

    def remove_lat_lon(self):
        match = '.dat' if self.memmap else '.tif'
        self.constants = [x for x in self.constants if not ('lon'+match in x or 'lat'+match in x)]

