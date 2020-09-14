import numpy as np
from crdm.loaders.Aggregate import Aggregate


class AggregatePixels(Aggregate):

    def make_feature_stack(self) -> np.array:
        assert 'size' in self.kwargs, "'size' must be included as a kwarg when instantiating the AggregatePixels class."
        assert self.kwargs['size'] <= 161040, "'size' must be smaller than 161040 (the number of 9km pixels in the CONUS domain)."
        
        [np.memmap(x, 'float32', 'c') for x in self.monthlys[:10]]
        
        indices = np.random.choice(161040, size=size, replace=False)
        train, test = indices[:size//2], indices[size//2:]


    def get_features(self):
        pass
    
    def get_target(self):
        pass