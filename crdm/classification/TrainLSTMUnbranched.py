import argparse
from crdm.classification.LSTMUnbranched import LSTM
from crdm.loaders.LoaderLSTMUnbranched import PixelLoader
from crdm.utils.ImportantVars import MONTHLY_VARS, WEEKLY_VARS
from crdm.utils.ParseFileNames import parse_fname
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import pickle


def train_lstm(target_dir, in_features, n_weeks, epochs=50, batch_size=64, hidden_size=64):

    test_loader = PixelLoader(target_dir=target_dir, in_features=in_features,
                              n_weeks=n_weeks, batch_size=batch_size, test=True, cuda=torch.cuda.is_available())
    train_loader = PixelLoader(target_dir=target_dir, in_features=in_features,
                               n_weeks=n_weeks, batch_size=batch_size, test=False, cuda=torch.cuda.is_available())

    dims = test_loader[0][0].shape

    test_sampler = torch.utils.data.sampler.BatchSampler(
        torch.utils.data.sampler.RandomSampler(test_loader),
        batch_size=batch_size,
        drop_last=False)

    train_sampler = torch.utils.data.sampler.BatchSampler(
        torch.utils.data.sampler.RandomSampler(train_loader),
        batch_size=batch_size,
        drop_last=False)

    train_loader = DataLoader(dataset=train_loader, sampler=train_sampler)
    test_loader = DataLoader(dataset=test_loader, sampler=test_sampler)

    # Define model, loss and optimizer.
    model = LSTM(size=dims[-1], hidden_size=hidden_size,  batch_size=batch_size,
                 output_size=1, cuda=torch.cuda.is_available())

    if torch.cuda.is_available():
        print('Using GPU')
        model.cuda()

    # Provide relative frequency weights to use in loss function.
    criterion = nn.MSELoss()
    lr = 3e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, threshold=1e-5, verbose=True)

    prev_best_loss = 1e6
    err_out = {}

    out_name_mod = 'epochs-{}_batch-{}nWeeks-{}_hiddenSize-{}_fType-model.p'.format(
        epochs, batch_size, n_weeks, hidden_size)
    out_name_err = 'epochs-{}_batch-{}nWeeks-{}_hiddenSize-{}_fType-err.p'.format(
        epochs, batch_size, n_weeks, hidden_size)

    week_h, week_c = model.init_state()

    for epoch in range(epochs):
        total_loss = 0
        train_loss = []
        test_loss = []

        model.train()

        # Loop over each subset of data
        for i, item in enumerate(train_loader, 1):
            features, target = item[0].squeeze(), item[1].squeeze()
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
            features, target = item[0].squeeze(), item[1].squeeze()

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
            torch.save(model.state_dict(), out_name_mod)
            prev_best_loss = total_loss

        scheduler.step(total_loss)

        # Save out train and test set loss.
        err_out[epoch] = {'train': train_loss,
                          'test': test_loss}

        with open(out_name_err, 'wb') as f:
            pickle.dump(err_out, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Drought Prediction Model')
    parser.add_argument('-td', '--target_dir', type=str, help='Directory containing output classes.')
    parser.add_argument('-if', '--in_features', type=str, help='Directory containing training features.')
    parser.add_argument('-e', '--epochs', type=int, default=25, help='Number of epochs.')
    parser.add_argument('-bs', '--batch_size', type=int, help='Batch size to train model with.')
    parser.add_argument('-hs', '--hidden_size', type=int, help='LSTM hidden dimension size.')

    parser.set_defaults(batch_size=1024)
    parser.set_defaults(hidden_size=1024)

    args = parser.parse_args()

    train_lstm(target_dir=args.target_dir, in_features=args.in_features, epochs=args.epochs,
               batch_size=args.batch_size, hidden_size=args.hidden_size, n_weeks=25)
