import argparse
from crdm.models.LSTMUnbranched import LSTM
from crdm.training.MakeTrainingData import make_training_data
from crdm.training.TrainModel import train_model
from crdm.loaders.LSTMLoader import LSTMLoader
import os
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
    model = LSTM(size=shps['train_x.dat'][1], hidden_size=setup['hidden_size'], batch_size=setup['batch_size'],
                 mx_lead=setup['mx_lead'])

    criterion = nn.MSELoss()
    lr = 0.002
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=10, threshold=1e-5, verbose=True, factor=0.5)

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
    parser.add_argument('-e', '--epochs', type=int, default=25, help='Number of epochs.')
    parser.add_argument('-bs', '--batch_size', type=int, default=64, help='Batch size to train model with.')
    parser.add_argument('-hs', '--hidden_size', type=int, default=64, help='LSTM hidden dimension size.')
    parser.add_argument('-nw', '--n_weeks', type=int, default=25, help='Number of week history to use for prediction')
    parser.add_argument('-sz', '--size', type=int, default=1024, help='How many samples to take per image.')
    parser.add_argument('-clip', '--clip', type=float, default=-1, help='Gradient clip')

    parser.add_argument('-mx', '--mx_lead', type=int, default=8,
                        help='How many weeks into the future to make predictions.')
    parser.add_argument('-dn', '--dirname', type=str, default=None,
                        help='Directory with training data to use. If left blank, training data will be created.')

    parser.add_argument('--search', dest='search', action='store_true', help='Perform hyperparameter grid search.')
    parser.add_argument('--no-search', dest='search', action='store_false',
                        help="Don't perform hyperparameter grid search.")
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
        'clip': args.clip,
        'early_stop': 10,
        'model_type': 'lstm'
    }
    dirname = args.dirname
    i = 3
    # Hyperparameter grid search
    if args.search:

        week_list = [15, 25, 50]
        hidden_list = [512, 1024]

        setup['index'] = i
        setup['n_weeks'] = 30
        setup['hidden_size'] = 256
        print('Grid search with n_weeks={}, hidden_size={}'.format(15, 128))

        # dirname = train_lstm(setup, dirname=args.dirname)
        setup['n_weeks'] = args.n_weeks
        for hidden in hidden_list:
            for n_weeks in week_list:
                setup['hidden_size'] = hidden
                setup['n_weeks'] = n_weeks
                setup['index'] = i    
                with open(os.path.join(dirname, 'metadata_{}_{}.p'.format(setup['index'], setup['model_type'])), 'wb') as f:
                    pickle.dump(setup, f)
                print('Grid search with hidden_size={}, n_weeks={}'.format(hidden, n_weeks))
                dirname = train_lstm(setup, dirname=dirname)
                i += 1

    else:
        setup['index'] = i
        dirname = train_lstm(setup, dirname=args.dirname)
        with open(os.path.join(dirname, 'metadata_{}_{}.p'.format(setup['index'], setup['model_type'])), 'wb') as f:
            pickle.dump(setup, f)
