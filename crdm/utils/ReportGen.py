import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def pretty_params(metadata):

    with open(metadata, 'rb') as f:
        metadata = pickle.load(f)

    out = []
    for k, v in metadata.items():
        k = k.replace('_', ' ').title()
        v = ', '.join(v) if type(v) == list else str(v)

        out.append('{}: {}'.format(k, v))

    return '\n'.join(out)

def calc_loss(err):

    with open(err, 'rb') as f:
        err = pickle.load(f)

    m_train_out, m_test_out = [], []
    s_train_out, s_test_out = [], []
    for epoch in err:
        train = err[epoch]['train']
        test = err[epoch]['test']

        m_train, m_test = np.mean(train), np.mean(test)
        s_train, s_test = np.std(train), np.std(test)

        m_train_out.append(m_train)
        m_test_out.append(m_test)
        s_train_out.append(s_train)
        s_test_out.append(s_test)

    out = pd.DataFrame({
        'm_train': m_train_out,
        'm_test': m_test_out,
        's_train': s_train_out,
        's_test': s_test_out
    })



def example_img(model, feature_dir):
    pass
