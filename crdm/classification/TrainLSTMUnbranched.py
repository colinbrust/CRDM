import argparse
from crdm.classification.LSTMUnbranched import LSTM
from crdm.loaders.TemporalLoader import DroughtLoader
from crdm.utils.MakeModelDir import make_model_dir
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pickle

FEATS = ['pr', 'vpd', 'tmmx', 'sm-rootzone', 'vapor', 'ET', 'gpp', 'vs', 'vod', 'USDM']
NUM_CONST = 17
NUM_FEATS = 17


def train_lstm(feature_dir, const_dir, n_weeks=25, epochs=50, batch_size=64, hidden_size=64, mx_lead=12, feats=FEATS):
    test_loader = DroughtLoader(feature_dir, const_dir, False, mx_lead, n_weeks, batch_size, feats)
    train_loader = DroughtLoader(feature_dir, const_dir, True, mx_lead, n_weeks, batch_size, feats)

    test_sampler = torch.utils.data.sampler.BatchSampler(
        torch.utils.data.sampler.RandomSampler(test_loader),
        batch_size=batch_size,
        drop_last=False)

    train_sampler = torch.utils.data.sampler.BatchSampler(
        torch.utils.data.sampler.RandomSampler(train_loader),
        batch_size=batch_size,
        drop_last=False)

    test_loader = DataLoader(dataset=test_loader, sampler=test_sampler)
    train_loader = DataLoader(dataset=train_loader, sampler=train_sampler)

    size = NUM_FEATS if feats[0] == '*' else len(feats)
    # Define model, loss and optimizer.
    model = LSTM(size=size, hidden_size=hidden_size, batch_size=batch_size, mx_lead=mx_lead, const_size=NUM_CONST)

    if torch.cuda.is_available():
        print('Using GPU')
        model.cuda()

    # Provide relative frequency weights to use in loss function.
    criterion = nn.MSELoss()
    lr = 3e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, threshold=1e-5, verbose=True, factor=0.5)

    prev_best_loss = 1e6
    err_out = {}

    pth = make_model_dir()

    metadata = {'epochs': epochs,
                'batch_size': batch_size,
                'hidden_size': hidden_size,
                'n_weeks': n_weeks,
                'mx_lead': mx_lead,
                'feats': feats,
                'num_const': NUM_CONST}

    with open(os.path.join(pth, 'metadata.p'), 'wb') as f:
        pickle.dump(metadata, f)

    week_h, week_c = model.init_state()

    for epoch in range(epochs):
        total_loss = 0
        train_loss = []
        test_loss = []

        model.train()

        # Loop over each subset of data
        for i, item in enumerate(train_loader, 1):
            features, consts, target = item[0].squeeze(), item[1].squeeze(), item[2].squeeze()
            # Zero out the optimizer's gradient buffer
            optimizer.zero_grad()

            # Make prediction with model
            outputs, _, _ = model(features, (week_h, week_c))
            outputs = outputs.squeeze()
            # Compute the loss and step the optimizer
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            print('Epoch: {}, Train Loss: {}'.format(epoch, loss.item()))

            # Store loss info
            train_loss.append(loss.item())

        # Switch to evaluation mode
        model.eval()

        for i, item in enumerate(test_loader, 1):
            features, consts, target = item[0].squeeze(), item[1].squeeze(), item[2].squeeze()

            # Make prediction with model
            outputs, _, _ = model(features, (week_h, week_c))
            outputs = outputs.squeeze()

            # Compute the loss and step the optimizer
            loss = criterion(outputs, target)

            print('Epoch: {}, Test Loss: {}'.format(epoch, loss.item()))

            # Save loss info
            total_loss += loss.item()
            test_loss.append(loss.item())

        # If our new loss is better than old loss, save the model
        if prev_best_loss > total_loss:
            torch.save(model.state_dict(), os.path.join(pth, 'model.p'))
            prev_best_loss = total_loss

        scheduler.step(total_loss)

        # Save out train and test set loss.
        err_out[epoch] = {'train': train_loss,
                          'test': test_loss}

        with open(os.path.join(pth, 'err.p'), 'wb') as f:
            pickle.dump(err_out, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Drought Prediction Model')
    parser.add_argument('-f', '--feature_dir', type=str, help='Directory containing training features.')
    parser.add_argument('-c', '--const_dir', type=str, help='Directory containing constant features.')
    parser.add_argument('-e', '--epochs', type=int, default=25, help='Number of epochs.')
    parser.add_argument('-bs', '--batch_size', type=int, default=64, help='Batch size to train model with.')
    parser.add_argument('-hs', '--hidden_size', type=int, default=64, help='LSTM hidden dimension size.')
    parser.add_argument('-nw', '--n_weeks', type=int, default=25, help='Number of week history to use for prediction')
    parser.add_argument('-mx', '--max_lead', type=int, default=8,
                        help='How many weeks into the future to make predictions.')

    parser.add_argument('--search', dest='search', action='store_true', help='Perform hyperparameter grid search.')
    parser.add_argument('--no-search', dest='search', action='store_false',
                        help="Don't perform hyperparameter grid search.")
    parser.set_defaults(search=False)

    args = parser.parse_args()
    if args.search:

        feat_list = [['*'], ['pr', 'USDM'], ['pr', 'vpd', 'tmmx', 'sm-rootzone', 'USDM']]
        for feats in feat_list:
            print('Grid search with feat_list={}'.format(feats))
            train_lstm(feature_dir=args.feature_dir, const_dir=args.const_dir, epochs=15,
                       batch_size=args.batch_size, hidden_size=args.hidden_size,
                       n_weeks=25, mx_lead=args.max_lead, feats=feats)

        week_list = [10, 20, 30, 40, 50]
        for week in week_list:
            print('Grid search with n_weeks={}'.format(week))
            train_lstm(feature_dir=args.feature_dir, const_dir=args.const_dir, epochs=20,
                       batch_size=args.batch_size, hidden_size=args.hidden_size,
                       n_weeks=week, mx_lead=args.max_lead, feats=['*'])

        hidden_list = [32, 64, 128, 16]
        for hidden in hidden_list:
            print('Grid search with hidden_size={}'.format(hidden))
            train_lstm(feature_dir=args.feature_dir, const_dir=args.const_dir, epochs=20,
                       batch_size=args.batch_size, hidden_size=hidden,
                       n_weeks=25, mx_lead=args.max_lead, feats=['*'])

    else:
        train_lstm(feature_dir=args.feature_dir, const_dir=args.const_dir, epochs=args.epochs,
                   batch_size=args.batch_size, hidden_size=args.hidden_size,
                   n_weeks=args.n_weeks, mx_lead=args.max_lead, feats=['*'])
