import pandas as pd
from crdm.loaders.AggregatePixels import PremakeTrainingPixels
from crdm.utils.MakeModelDir import make_model_dir
from crdm.utils.ImportantVars import DIMS
import os
import numpy as np
from pathlib import Path
import pickle
import verde as vd


def make_training_data(in_features, out_classes, **kwargs):

    assert 'n_weeks' in kwargs
    assert 'mx_lead' in kwargs
    assert 'size' in kwargs

    targets = sorted([str(x) for x in Path(out_classes).glob('*.dat')])
    targets = [targets[i:i+kwargs['mx_lead']] for i in range(len(targets))]
    targets = list(filter(lambda x: len(x) == kwargs['mx_lead'], targets))
    targets = [x for x in targets if not ('/2014' in x[0] or '/2017' in x[0] or '/2007' in x[0])]

    df = []
    for y in range(DIMS[0]):
        for x in range(DIMS[1]):
            df.append({'x': x, 'y': y, 'val': 0})

    df = pd.DataFrame(df)
    train_locs, test_locs = vd.train_test_split(coordinates=(df.y, df.x), data=df.val, spacing=9, test_size=0.25)
    train, test = train_locs[0], test_locs[0]

    chk = np.zeros(DIMS)

    for i, _ in enumerate(train[0]):
        chk[train[0][i], train[1][i]] = 1

    train_locs = list(np.where(chk.ravel() == 1)[0])
    test_locs = list(np.where(chk.ravel() == 0)[0])

    dirname = make_model_dir()

    locs = {'train': train_locs,
            'test': test_locs}

    with open(os.path.join(dirname, 'locs.p'), 'wb') as f:
        pickle.dump(locs, f)

    train_x, train_y = [], []
    test_x, test_y = [], []

    for target in targets:
        try:
            print(target[0])
            tmp = PremakeTrainingPixels(target, in_features, kwargs['n_weeks'], kwargs['mx_lead'], kwargs['size'])
            indices = tmp.sample_evenly()
            tmp_w, tmp_t = tmp.premake_features(indices=list(set(indices).intersection(set(train_locs))))
            train_x.append(tmp_w)
            train_y.append(tmp_t)

            tmp_w, tmp_t = tmp.premake_features(indices=list(set(indices).intersection(set(test_locs))))
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

    shps = {'train_x.dat': train_x.shape,
            'train_y.dat': train_y.shape,
            'test_x.dat': test_x.shape,
            'test_y.dat': test_y.shape}

    with open(os.path.join(dirname, 'shps.p'), 'wb') as f:
        pickle.dump(shps, f)

    for name, dat in zip(['train_x.dat', 'train_y.dat', 'test_x.dat', 'test_y.dat'], [train_x, train_y, test_x, test_y]):

        mm = np.memmap(os.path.join(dirname, name), dtype='float32', mode='w+', shape=dat.shape)
        mm[:] = dat[:]
        del mm

    return dirname
