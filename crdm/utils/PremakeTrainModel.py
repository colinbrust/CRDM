from crdm.loaders.AggregatePixels import PremakeTrainingPixels
from crdm.utils.MakeModelDir import make_model_dir
from crdm.utils.ImportantVars import LENGTH
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split


def premake_train_model(in_features, out_classes, **kwargs):

    assert 'n_weeks' in kwargs
    assert 'mx_lead' in kwargs
    assert 'size' in kwargs

    targets = sorted([str(x) for x in Path(out_classes).glob('*.dat')])
    targets = [targets[i:i+kwargs['mx_lead']] for i in range(len(targets))]
    targets = list(filter(lambda x: len(x) == kwargs['mx_lead'], targets))

    train_locs, test_locs = train_test_split(list(range(LENGTH)), train_size=0.6)

    train_w, train_t = [], []
    test_w, test_t = [], []

    for target in targets:
        try:
            print(target[0])
            tmp = PremakeTrainingPixels(target, in_features, kwargs['n_weeks'],
                                        indices=np.random.choice(train_locs, int(kwargs['size']*0.75)))
            tmp_w, tmp_t = tmp.premake_features()
            train_w.append(tmp_w)
            train_t.append(tmp_t)

            tmp = PremakeTrainingPixels(target, in_features, kwargs['n_weeks'],
                                        indices=np.random.choice(test_locs, int(kwargs['size']*0.25)))
            tmp_w, tmp_t = tmp.premake_features()
            test_w.append(tmp_w)
            test_t.append(tmp_t)

        except AssertionError as e:
            print(e)

    train_w, train_t = np.concatenate(train_w, axis=-1), np.concatenate(train_t, axis=-1)
    test_w, test_t = np.concatenate(test_w, axis=-1), np.concatenate(test_t, axis=-1)

