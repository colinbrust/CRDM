import argparse
from dc.models.Seq2Seq import Seq2Seq
from dc.training.MakeTrainingData import make_training_data
from dc.training.TrainModel import train_model
from dc.loaders.LSTMLoader import LSTMLoader
import os
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import pickle


def train_seq(setup):

    with open(os.path.join(setup['dirname'], 'shps.p'), 'rb') as f:
        shps = pickle.load(f)

    # Make train and test set data loaders and add them to model setup
    train_loader = LSTMLoader(dirname=setup['dirname'], train=True, categorical=setup['categorical'], n_weeks=setup['n_weeks'])
    test_loader = LSTMLoader(dirname=setup['dirname'], train=False, categorical=setup['categorical'], n_weeks=setup['n_weeks'])
    setup['train'] = DataLoader(dataset=train_loader, batch_size=setup['batch_size'], shuffle=True, drop_last=True)
    setup['test'] = DataLoader(dataset=test_loader, batch_size=setup['batch_size'], shuffle=True, drop_last=True)

    setup['batch_first'] = True
    model = Seq2Seq(1, shps['train_x.dat'][1], setup['n_weeks'], setup['hidden_size'], setup['mx_lead'])

    criterion = nn.MSELoss()
    lr = 0.002
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, threshold=0.00001,  verbose=True)

    setup['model'] = model
    setup['criterion'] = criterion
    setup['optimizer'] = optimizer
    setup['scheduler'] = scheduler

    model = train_model(setup)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Drought Prediction Model')
    parser.add_argument('-if', '--in_features', type=str, help='Directory containing training features.')
    parser.add_argument('-c', '--out_classes', type=str, help='Directory containing target classes.')
    parser.add_argument('-e', '--epochs', type=int, default=25, help='Number of epochs.')
    parser.add_argument('-bs', '--batch_size', type=int, default=64, help='Batch size to train model with.')
    parser.add_argument('-hs', '--hidden_size', type=int, default=64, help='LSTM hidden dimension size.')
    parser.add_argument('-nw', '--n_weeks', type=int, default=25, help='Number of week history to use for prediction')
    parser.add_argument('-sz', '--size', type=int, default=1024, help='How many samples to take per image.')
    parser.add_argument('-lt', '--lead_time', type=int, default=None,
                        help='Lead time to predict. If None, a timeseries will be predicted')

    parser.add_argument('-mx', '--mx_lead', type=int, default=8,
                        help='How many weeks into the future to make predictions.')

    parser.add_argument('-dn', '--dirname', type=str, default=None,
                        help='Directory with training data to use. If left blank, training data will be created.')

    parser.set_defaults(search=False)

    args = parser.parse_args()

    setup = {
        'in_features': args.in_features,
        'out_classes': args.out_classes,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'hidden_size': args.hidden_size,
        'n_weeks': args.n_weeks,
        'mx_lead': args.mx_lead,
        'size': args.size,
        'lead_time': args.lead_time,
        'early_stop': 10,
        'categorical': args.categorical,
        'model_type': 'seq',
        'pix_mask': '/mnt/e/PycharmProjects/DroughtCast/data/pix_mask.dat',
        'model': 'vanilla',
        'iter_print': 100
    }

    if args.dirname is None:
        dirname = make_training_data(**setup)
        setup['dirname'] = dirname
    else:
        setup['dirname'] = args.dirname

    i = 0

    while os.path.exists(os.path.join(setup['dirname'], 'metadata_{}_{}.p'.format(i, setup['model_type']))):
        i += 1

    setup['index'] = i

    with open(os.path.join(setup['dirname'], 'metadata_{}_{}.p'.format(setup['index'], setup['model_type'])), 'wb') as f:
        pickle.dump(setup, f)

    train_seq(setup)

