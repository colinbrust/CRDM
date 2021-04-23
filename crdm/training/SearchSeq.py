import argparse
from crdm.models.LSTM import LSTM
from crdm.models.SeqToSeq import Seq2Seq
from crdm.training.MakeTrainingData import make_training_data
from crdm.training.TrainModel import train_model
from crdm.loaders.LSTMLoader import LSTMLoader
import os
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import pickle


def train_lstm(setup, dirname=None):

    # If model isn't being trained on premade data, create a model directory and make training data.
    if dirname is None:
        dirname = make_training_data(**setup)

    setup['dirname'] = dirname
    with open(os.path.join(dirname, 'shps.p'), 'rb') as f:
        shps = pickle.load(f)

    # Make train and test set data loaders and add them to model setup
    train_loader = LSTMLoader(dirname=dirname, train=True)
    test_loader = LSTMLoader(dirname=dirname, train=False)
    setup['train'] = DataLoader(dataset=train_loader, batch_size=setup['batch_size'], shuffle=True, drop_last=True)
    setup['test'] = DataLoader(dataset=test_loader, batch_size=setup['batch_size'], shuffle=True, drop_last=True)

    # Define model, loss and optimizer.
    if setup.seq:
        model = Seq2Seq(1, shps['train_x.dat'][1], shps['train_x.dat'][-1], setup['hidden_size'], setup['mx_lead'])
    else:
        model = LSTM(size=shps['train_x.dat'][1], hidden_size=setup['hidden_size'], batch_size=setup['batch_size'],
                     mx_lead=setup['mx_lead'], lead_time=setup['lead_time'])

    criterion = nn.MSELoss()
    lr = 0.002
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True)
    scheduler = StepLR(optimizer, step_size=3, gamma=0.5, verbose=True)

    setup['model'] = model
    setup['criterion'] = criterion
    setup['optimizer'] = optimizer
    setup['scheduler'] = scheduler

    train_model(setup)
    return dirname


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Drought Prediction Model')
    parser.add_argument('-if', '--in_features', type=str, help='Directory containing training features.')
    parser.add_argument('-c', '--out_classes', type=str, help='Directory containing target classes.')
    parser.add_argument('-dn', '--dirname', type=str, default=None,
                        help='Directory with training data to use. If left blank, training data will be created.')

    parser.set_defaults(search=False)

    args = parser.parse_args()

    setup = {
        'in_features': args.in_features,
        'out_classes': args.out_classes,
        'epochs': 15,
        'batch_size': 64,
        'hidden_size': 64,
        'n_weeks': 30,
        'mx_lead': 12,
        'size': 8192,
        'clip': -1,
        'lead_time': None,
        'early_stop': 5,
        'model_type': 'seq',
        'pix_mask': '/home/colin/data/in_feature/pix_mask.dat'
    }
    dirname = args.dirname
    i = 1
    # Hyperparameter grid search
    if args.search:
        
        hidden_list = [64, 128, 256]
        batch_list = [64, 128, 256]
        setup['index'] = i
        setup['n_weeks'] = 30

        # dirname = train_lstm(setup, dirname=None)
        for hidden in hidden_list:
            for batch in batch_list:
                setup['hidden_size'] = hidden
                setup['batch_size'] = batch
                setup['index'] = i    
                with open(os.path.join(dirname, 'metadata_{}_{}.p'.format(setup['index'], setup['model_type'])), 'wb') as f:
                    pickle.dump(setup, f)
                print('Grid search with hidden_size={}, batch_size={}'.format(hidden, batch))
                dirname = train_lstm(setup, dirname=dirname)
                i += 1

    else:
        setup['index'] = i
        dirname = train_lstm(setup, dirname=args.dirname)
        with open(os.path.join(dirname, 'metadata_{}_{}.p'.format(setup['index'], setup['model_type'])), 'wb') as f:
            pickle.dump(setup, f)
