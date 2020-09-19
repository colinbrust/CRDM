# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import os
%matplotlib inline
# %%
# f = '/Users/colinbrust/projects/CRDM/data/drought/model_results/LSTM_epochs-20_batch-32_nMonths-13_hiddenSize-32_leadTime-2_err.p'
def plot_single_error(f):
    with open(f, 'rb') as x:
        dat = pickle.load(x)

    out = {}
    for key in dat.keys():
        train = np.mean(dat[key]['train'])
        test = np.mean(dat[key]['test'])
        out[key] = {'train': train,
                    'test': test}

    out = pd.DataFrame.from_dict(out).T
    out.plot(y=['train', 'test'], xlabel='Epoch', ylabel='Cross-Entropy Loss',
             xticks=[x for x in dat.keys()])
# %%
f_dir = '/Users/colinbrust/projects/CRDM/data/drought/model_results'
# def compare_batch_errors(f_dir):
f_list = [str(x) for x in Path(f_dir).glob('*_err.p')]
for f in f_list:
    epochs, batch, nMonths, hiddenSize, leadTime = [x.split('-')[-1] for x in os.path.basename(f).split('_')[1:-1]]
# %%
