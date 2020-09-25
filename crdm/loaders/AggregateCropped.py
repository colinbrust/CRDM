from crdm.loaders.Aggregate import Aggregate
from crdm.utils.ImportantVars import DIMS


# test = AggregateCropped('/Users/colinbrust/projects/CRDM/data/drought/out_classes/out_memmap/20140225_USDM.dat', '/Users/colinbrust/projects/CRDM/data/drought/in_features', 2, 12)

class AggregateCropped(Aggregate):

    def premake_features(self):

        [np.memmap(x, dtype='float32', mode='r', shape=DIMS) for x in self.monthlys]
        pass
    pass