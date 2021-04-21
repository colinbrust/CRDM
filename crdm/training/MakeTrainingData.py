from crdm.loaders.AggregatePixels import PremakeTrainingPixels
from crdm.utils.MakeModelDir import make_model_dir
from crdm.utils.ImportantVars import LENGTH
import os
import numpy as np
from pathlib import Path
import pickle
from sklearn.model_selection import train_test_split


def make_training_data(in_features, out_classes, **kwargs):

    assert 'n_weeks' in kwargs
    assert 'mx_lead' in kwargs
    assert 'size' in kwargs

    targets = sorted([str(x) for x in Path(out_classes).glob('*.dat')])
    targets = [targets[i:i+kwargs['mx_lead']] for i in range(len(targets))]
    targets = list(filter(lambda x: len(x) == kwargs['mx_lead'], targets))
    targets = [x for x in targets if not ('/2015' in x[0] or '/2017' in x[0] or '/2007' in x[0])]

    mask = np.memmap(kwargs['pix_mask'], dtype='int8')
    locs = [idx for idx, x in enumerate(mask) if x == 1]

    train_locs, test_locs = train_test_split(locs, train_size=0.6)

    train_x, train_y = [], []
    test_x, test_y = [], []

    for target in targets:
        try:
            print(target[0])
            tmp = PremakeTrainingPixels(target, in_features, kwargs['n_weeks'])
            tmp_w, tmp_t = tmp.premake_features(indices=np.random.choice(train_locs, int(kwargs['size']*0.75)))
            train_x.append(tmp_w)
            train_y.append(tmp_t)

            tmp = PremakeTrainingPixels(target, in_features, kwargs['n_weeks'])
            tmp_w, tmp_t = tmp.premake_features(indices=np.random.choice(test_locs, int(kwargs['size']*0.25)))
            test_x.append(tmp_w)
            test_y.append(tmp_t)

        except AssertionError as e:
            print(e)

    train_x, train_y = np.concatenate(train_x, axis=-1), np.concatenate(train_y, axis=-1)
    test_x, test_y = np.concatenate(test_x, axis=-1), np.concatenate(test_y, axis=-1)

    train_x, train_y = np.swapaxes(train_x, 0, 2), np.swapaxes(train_y, 0, 1)
    test_x, test_y = np.swapaxes(test_x, 0, 2), np.swapaxes(test_y, 0, 1)
    train_y, test_y = train_y/5, test_y/5

    train_x, test_x = np.swapaxes(train_x, 1, 2), np.swapaxes(test_x, 1, 2)

    dirname = make_model_dir()

    locs = {'train': train_locs,
            'test': test_locs}

    shps = {'train_x.dat': train_x.shape,
            'train_y.dat': train_y.shape,
            'test_x.dat': test_x.shape,
            'test_y.dat': test_y.shape}

    with open(os.path.join(dirname, 'locs.p'), 'wb') as f:
        pickle.dump(locs, f)

    with open(os.path.join(dirname, 'shps.p'), 'wb') as f:
        pickle.dump(shps, f)

    for name, dat in zip(['train_x.dat', 'train_y.dat', 'test_x.dat', 'test_y.dat'], [train_x, train_y, test_x, test_y]):

        mm = np.memmap(os.path.join(dirname, name), dtype='float32', mode='w+', shape=dat.shape)
        mm[:] = dat[:]
        del mm

    return dirname
