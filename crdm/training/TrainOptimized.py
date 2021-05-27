import argparse
from crdm.models.SeqTest import Seq2Seq
from crdm.training.TrainModel import train_model
from crdm.loaders.LSTMLoader import LSTMLoader
from crdm.utils.ModelToMap import Mapper
from crdm.utils.ImportantVars import ConvergenceError
import os
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import pickle


def train_lstm(setup):

    with open(os.path.join(setup['dirname'], 'shps.p'), 'rb') as f:
        shps = pickle.load(f)

    # Make train and test set data loaders and add them to model setup
    train_loader = LSTMLoader(dirname=setup['dirname'], train=True, categorical=setup['categorical'],
                              n_weeks=setup['n_weeks'], sample=None, even_sample=True)

    test_loader = LSTMLoader(dirname=setup['dirname'], train=False, categorical=setup['categorical'],
                             n_weeks=setup['n_weeks'], sample=None, even_sample=False)

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
        'early_stop': 20,
        'categorical': False,
        'pix_mask': '/mnt/e/PycharmProjects/DroughtCast/data/pix_mask.dat',
        'model_type': 'vanilla',
        'dirname': args.dirname
    }
    
    with open(os.path.join(setup['dirname'], 'shps.p'), 'rb') as f:
        shps = pickle.load(f)
    
    for ensemble in range(5):
    
        i = 0

        while os.path.exists('ensemble_{}'.format(i)):
            i += 1

        out_dir = 'ensemble_{}'.format(i)
        os.mkdir(out_dir)
        setup['out_dir'] = out_dir

        flag = True
        while flag:
            try:
                model = train_lstm(setup)
                flag = False
            except ConvergenceError as e:
                print(e)

        mpr = Mapper(model, setup, args.in_features, args.out_classes, out_dir, shps, True, None, setup['categorical'])
        mpr.get_preds()