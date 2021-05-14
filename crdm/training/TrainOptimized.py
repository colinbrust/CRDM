import argparse
from crdm.models.SeqAttn import Seq2Seq
from crdm.models.SeqVanilla import Seq2Seq as vanilla
from crdm.training.TrainModel import train_model
from crdm.loaders.LSTMLoader import LSTMLoader
import os
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import pickle


def train_lstm(setup):

    with open(os.path.join(setup['dirname'], 'shps.p'), 'rb') as f:
        shps = pickle.load(f)

    # Make train and test set data loaders and add them to model setup
    train_loader = LSTMLoader(dirname=setup['dirname'], train=True, categorical=setup['categorical'], n_weeks=setup['n_weeks'])
    test_loader = LSTMLoader(dirname=setup['dirname'], train=False, categorical=setup['categorical'], n_weeks=setup['n_weeks'])
    setup['train'] = DataLoader(dataset=train_loader, batch_size=setup['batch_size'], shuffle=True, drop_last=True)
    setup['test'] = DataLoader(dataset=test_loader, batch_size=setup['batch_size'], shuffle=True, drop_last=True)

    if setup['model'] == 'vanilla':
        print('Using vanilla model.')
        setup['batch_first'] = True
        model = vanilla(1, shps['train_x.dat'][1], setup['hidden_size'], setup['mx_lead'], setup['categorical'])
    else:
        print('Using simple attention.')
        setup['batch_first'] = True
        model = Seq2Seq(1, shps['train_x.dat'][1], shps['train_x.dat'][-1],
                        setup['hidden_size'], setup['mx_lead'], categorical=setup['categorical'])

    criterion = nn.MSELoss()
    lr = 0.002
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5, verbose=True)

    setup['model'] = model
    setup['criterion'] = criterion
    setup['optimizer'] = optimizer
    setup['scheduler'] = scheduler

    train_model(setup)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Drought Prediction Model')
    parser.add_argument('-if', '--in_features', type=str, help='Directory containing training features.')
    parser.add_argument('-c', '--out_classes', type=str, help='Directory containing target classes.')
    parser.add_argument('-e', '--epochs', type=int, default=25, help='Number of epochs.')
    parser.add_argument('-dn', '--dirname', type=str, default=None,
                        help='Directory with training data to use. If left blank, training data will be created.')

    args = parser.parse_args()

    # These are the best performing hyperparameters.
    setup = {
        'in_features': args.in_features,
        'out_classes': args.out_classes,
        'epochs': args.epochs,
        'batch_size': 128,
        'hidden_size': 128,
        'n_weeks': 30,
        'mx_lead': 12,
        'size': 1024,
        'lead_time': None,
        'early_stop': 10,
        'categorical': False,
        'pix_mask': '/mnt/e/PycharmProjects/DroughtCast/data/pix_mask.dat',
        'model_type': 'vanilla',
        'dirname': args.dirname
    }

    i = 0

    while os.path.exists('ensemble_{}'.format(i)):
        i += 1

    out_dir = 'ensemble_{}'.format(i)
    os.mkdir(out_dir)
    setup['out_dir'] = out_dir

    model = vanilla(1, 32, setup['hidden_size'], setup['mx_lead'], setup['categorical'])

    train_lstm(setup)

if not None:
    print('a')