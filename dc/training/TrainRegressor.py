import argparse
from dc.loaders.RegressorLoader import RegressorLoader
from dc.models.regression import Regressor
import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau


def train_regressor(setup):

    train_loader = RegressorLoader(ens_path=setup['ensemble_path'], target_path=setup['target_path'], train=True)
    test_loader = RegressorLoader(ens_path=setup['ensemble_path'], target_path=setup['target_path'], train=False)

    model = Regressor(num_ensemble=10, mx_lead=12)

    criterion = nn.MSELoss()
    lr = 0.01

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, threshold=0.00001,  verbose=True)

    setup['batch_first'] = True
    setup['model'] = model
    setup['criterion'] = criterion
    setup['optimizer'] = optimizer
    setup['scheduler'] = scheduler

    for t in range(100):
        x, y, lead_time = test[np.random.randint(len(test))]
        optimizer.zero_grad()
        y_pred = model(torch.Tensor(x), lead_time)
        loss = criterion(y_pred, torch.Tensor(y))
        loss.backward()
        optimizer.step()
        print(loss.item())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Drought Prediction Model')
    parser.add_argument('-ep', '--ensemble_path', type=str, help='Directory containing training features.')
    parser.add_argument('-tp', '--target_path', type=str, help='Directory containing target classes.')
    parser.add_argument('-e', '--epochs', type=int, default=25, help='Number of epochs.')
    parser.add_argument('-bs', '--batch_size', type=int, default=8, help='Batch size to train model with.')

    args = parser.parse_args()

    setup = {
        'ensemble_path': args.ensemble_path,
        'target_path': args.target_path,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'early_stop': 10,
        'categorical': False,
        'model_type': 'regressor',
        'iter_print': 10,
        'out_dir': '.'
    }

    train_regressor(setup)
