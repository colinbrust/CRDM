import argparse
from crdm.models.TCN import TCN
from crdm.training.MakeTrainingData import make_training_data
from crdm.training.TrainModel import train_model
from crdm.loaders.TCNLoader import TCNLoader
import os
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import pickle


def train_tcn(setup, dirname=None):

    # If model isn't being trained on premade data, create a model directory and make training data.
    if dirname is None:
        dirname = make_training_data(**setup)

    setup['dirname'] = dirname
    with open(os.path.join(dirname, 'shps.p'), 'rb') as f:
        shps = pickle.load(f)

    # Make train and test set data loaders and add them to model setup
    train_loader = TCNLoader(dirname=dirname, train=True)
    test_loader = TCNLoader(dirname=dirname, train=False)
    setup['train'] = DataLoader(dataset=train_loader, batch_size=setup['batch_size'], shuffle=True)
    setup['test'] = DataLoader(dataset=test_loader, batch_size=setup['batch_size'], shuffle=True)

    # Define model, loss and optimizer.
    model = TCN(input_size=shps['train_x.dat'][1], output_size=1,
                num_channels=[setup['hidden_size']] * setup['n_layers'],
                kernel_size=setup['kernel_size'], mx_lead=setup['mx_lead'], dropout=0.2)

    criterion = nn.MSELoss()
    lr = 0.002
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=10, threshold=1e-5, verbose=True, factor=0.5)

    setup['model'] = model
    setup['criterion'] = criterion
    setup['optimizer'] = optimizer
    setup['scheduler'] = scheduler
    setup['model_name'] = 'model-tcn_{}.p'.format(setup['index'])

    train_model(setup)
    return dirname


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Drought Prediction Model')
    parser.add_argument('-if', '--in_features', type=str, help='Directory containing training features.')
    parser.add_argument('-c', '--out_classes', type=str, help='Directory containing target classes.')
    parser.add_argument('-e', '--epochs', type=int, default=25, help='Number of epochs.')
    parser.add_argument('-bs', '--batch_size', type=int, default=64, help='Batch size to train model with.')
    parser.add_argument('-hs', '--hidden_size', type=int, default=64, help='LSTM hidden dimension size.')
    parser.add_argument('-ks', '--kernel_size', type=int, default=7, help='Filter kernel size.')
    parser.add_argument('-nl', '--n_layers', type=int, default=7, help='Number of residual layers in network.')
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
        'kernel_size': args.kernel_size,
        'n_layers': args.n_layers,
        'n_weeks': args.n_weeks,
        'mx_lead': args.mx_lead,
        'size': args.size,
        'clip': args.clip,
        'early_stop': 10,
        'model_type': 'tcn'
    }
    dirname = args.dirname
    i = 1
    # Hyperparameter grid search
    if args.search:
        setup['index'] = i
        week_list = [30, 50, 75, 100]
        # setup['n_weeks'] = 15
        print('Grid search with n_weeks={}'.format(15))
        # dirname = train_tcn(setup, dirname=args.dirname)
        for week in week_list:
            setup['n_weeks'] = week
            setup['index'] = i
            print('Grid search with n_weeks={}'.format(week))
            print(setup)
            with open(os.path.join(dirname, 'metadata_{}_{}.p'.format(setup['index'], setup['model_type'])), 'wb') as f:
                pickle.dump(setup, f)
            dirname = train_tcn(setup, dirname=dirname)
            i += 1

        hidden_list = [20, 25, 30, 35]
        layer_list = [5, 10, 15, 20]
        kern_list = [4, 5, 6, 7, 8]

        setup['n_weeks'] = args.n_weeks
        for hidden in hidden_list:
            for layer in layer_list:
                for kern in kern_list:
                    setup['hidden_size'] = hidden
                    setup['n_layers'] = layer
                    setup['kernel_size'] = kern
                    setup['index'] = i
                    with open(os.path.join(dirname, 'metadata.p'), 'wb') as f:
                        pickle.dump(setup, f)
                    print('Grid search with hidden_size={}, layers={}, kernel_size={}'.format(hidden, layer, kern))
                    dirname = train_tcn(setup, dirname=dirname)
                    i += 1

    else:
        setup['index'] = i
        dirname = train_tcn(setup, dirname=args.dirname)
        with open(os.path.join(dirname, 'metadata_{}_{}.p'.format(setup['index'], setup['model_type'])), 'wb') as f:
            pickle.dump(setup, f)
