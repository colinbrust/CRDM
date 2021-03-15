import numpy as np
from crdm.loaders.AggregateConvLSTM import AggregateSpatial


# TODO: Make an 'AggregatePixel' class that both this class and the 'AggregateTrainingPixels' inherit from to
#  minimize code duplication.
class AggregateAllSpatial(AggregateSpatial):

    def premake_features(self) -> np.array:

        weeklys = sorted(self.weeklys)
        weeklys = np.array([np.memmap(x, 'float32', 'r') for x in weeklys])
        drought = np.array([np.memmap(x, 'int8', 'r') for x in self.initial_drought])
        # Scale between -1 and 1
        drought = 2 * drought / 5 - 1

        weeklys = np.concatenate((weeklys, drought))

        # dim = variable x location
        constants = [np.memmap(x, 'float32', 'c') for x in [*self.constants, *self.annuals]]
        constants = np.array(constants)
        # Add day of year for target image.
        target_doy = self.target_date.timetuple().tm_yday
        target_doy = (target_doy - 1) / (366 - 1)
        target_doy = np.ones_like(constants[0]) * target_doy

        # Add day of year for image guess date.
        guess_doy = self.guess_date.timetuple().tm_yday
        guess_doy = (guess_doy - 1) / (366 - 1)
        guess_doy = np.ones_like(constants[0]) * guess_doy

        day_diff = self._get_day_diff()
        day_diff = (day_diff - 7) / (84 - 7)
        day_diff = np.ones_like(constants[0]) * day_diff

        constants = np.concatenate((constants, target_doy[np.newaxis]))
        constants = np.concatenate((constants, guess_doy[np.newaxis]))
        constants = np.concatenate((constants, day_diff[np.newaxis]))

        weeklys = np.vstack((constants, weeklys))

        return weeklys
