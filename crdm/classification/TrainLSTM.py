import argparse
from collections import Counter
from crdm.loaders.ContinuousPixelLoader import PixelLoader
from crdm.utils.ImportantVars import MONTHLY_VARS, WEEKLY_VARS
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from sklearn.model_selection import train_test_split
import pickle


class LSTM(nn.Module):
    def __init__(self, monthly_size=1, weekly_size=1, hidden_size=64, output_size=6,
                 batch_size=64, const_size=8, cuda=False):
        super().__init__()

        self.hidden_size = hidden_size
        self.device = 'cuda:0' if torch.cuda.is_available() and cuda else 'cpu'
        self.batch_size = batch_size
        self.weekly_size = weekly_size

        self.weekly_lstm = nn.LSTM(weekly_size, self.hidden_size)
        self.monthly_lstm = nn.LSTM(monthly_size, self.hidden_size)

        # Downscale to output size
        self.classifier = nn.Sequential(
            nn.Linear(2*hidden_size + const_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(16, output_size),
            nn.ReLU()
        )

    def init_state(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(1, self.batch_size, self.hidden_size, device=self.device),
                torch.zeros(1, self.batch_size, self.hidden_size, device=self.device))

    def forward(self, weekly_seq, monthly_seq, constants, prev_week_state, prev_month_state):
        # Run the LSTM forward

        week_out, week_state = self.weekly_lstm(weekly_seq, prev_week_state)
        month_out, month_state = self.monthly_lstm(monthly_seq, prev_month_state)

        lstm_and_const = torch.cat((week_out[-1], month_out[-1], constants), dim=1)
        preds = self.classifier(lstm_and_const)
        return preds, week_state, month_state


def train_lstm(feature_dir, target_dir, pixel_per_img, lead_time,
               n_weeks, epochs=50, batch_size=64, hidden_size=64, cuda=False):

    # feature_dir = '/mnt/e/PycharmProjects/CRDM/data/in_features'
    # target_dir = '/mnt/e/PycharmProjects/CRDM/data/out_classes/out_memmap'
    # n_weeks = 16
    # lead_time = 4
    # pixel_per_img = 10000
    # epochs = 10
    # batch_size = 512
    # hidden_size = 512

    # Make data loader
    loader = PixelLoader(target_dir=target_dir, feature_dir=feature_dir, pixel_per_img=pixel_per_img,
                         lead_time=lead_time, n_weeks=n_weeks)

    # Split into training and test sets
    train, test = train_test_split([x for x in range(len(loader))], test_size=0.25)
    train_sampler = SubsetRandomSampler(train)
    test_sampler = SubsetRandomSampler(test)

    train_loader = DataLoader(dataset=loader, batch_size=batch_size, sampler=train_sampler)
    test_loader = DataLoader(dataset=loader, batch_size=batch_size, sampler=test_sampler)

    # Define model, loss and optimizer.
    model = LSTM(weekly_size=len(WEEKLY_VARS), monthly_size=len(MONTHLY_VARS), hidden_size=hidden_size, output_size=6,
                 batch_size=batch_size, const_size=8)

    device = 'cuda:0' if torch.cuda.is_available() and cuda else 'cpu'
    model.to(device)

    if torch.cuda.is_available() and cuda:
        print('Using GPU')
        model.cuda()

    # Provide relative frequency weights to use in loss function. These weights correspond to approximately the
    # inverse of how frequent each drough class is.

    weights = torch.Tensor([0.2488, 0.9486, 0.9312, 0.9744, 0.9936, 0.9035]).type(
         torch.cuda.FloatTensor if (cuda and torch.cuda.is_available()) else torch.FloatTensor
    )
    criterion = nn.CrossEntropyLoss(weight=weights)
    lr = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, threshold=1e-4, verbose=True)

    prev_best_loss = 1e6
    err_out = {}

    out_name_mod = 'modelType-LSTM_epochs-{}_batch-{}_nWeeks-{}_hiddenSize-{}_leadTime-{}_numPixels-{}_rmFeatures-{}_fType-model.p'.format(epochs, batch_size, n_weeks, hidden_size, lead_time, pixel_per_img, False)
    out_name_err = 'modelType-LSTM_epochs-{}_batch-{}_nWeeks-{}_hiddenSize-{}_leadTime-{}_numPixels-{}_rmFeatures-{}_fType-err.p'.format(epochs, batch_size, n_weeks, hidden_size, lead_time, pixel_per_img, False)

    for epoch in range(epochs):
        total_loss = 0
        train_loss = []
        test_loss = []

        # Use this for stateful LSTM
        # week_h, week_c = model.init_state()
        # month_h, month_c = model.init_state()

        model.train()

        # Loop over each subset of data
        for i, item in enumerate(train_loader, 1):

            try:
                # Use this for stateless LSTM
                week_h, week_c = model.init_state()
                month_h, month_c = model.init_state()

                mon = item['mon'].permute(2, 0, 1)
                week = item['week'].permute(2, 0, 1)
                const = item['const'].permute(0, 1)

                mon = mon.type(torch.cuda.FloatTensor if (torch.cuda.is_available() and cuda) else torch.FloatTensor)
                week = week.type(torch.cuda.FloatTensor if (torch.cuda.is_available() and cuda) else torch.FloatTensor)
                const = const.type(torch.cuda.FloatTensor if (torch.cuda.is_available() and cuda) else torch.FloatTensor)

                # Zero out the optimizer's gradient buffer
                optimizer.zero_grad()

                # Make prediction with model
                outputs, (week_h, week_c), (month_h, month_c) = model(week, mon, const, (week_h, week_c), (month_h, month_c))

                week_h, month_h = week_h.detach(), month_h.detach()
                week_c, month_c = week_c.detach(), month_c.detach()

                # Compute the loss and step the optimizer
                loss = criterion(
                    outputs.type(torch.cuda.FloatTensor if (torch.cuda.is_available() and cuda) else torch.FloatTensor),
                    item['target'].type(torch.cuda.LongTensor if (torch.cuda.is_available() and cuda) else torch.LongTensor)
                )
                loss.backward()
                optimizer.step()

                # if i % 500 == 0:
                print('Epoch: {}, Train Loss: {}'.format(epoch, loss.item()))

                # Store loss info
                train_loss.append(loss.item())

            except RuntimeError as e:
                # For some reason the SubsetRandomSampler makes uneven batch sizes at the end of the batch,
                # so this is done as a workaound.
                print(e, '\nSkipping this mini-batch.')

        # Switch to evaluation mode
        model.eval()

        for i, item in enumerate(test_loader, 1):

            try:

                week_h, week_c = model.init_state()
                month_h, month_c = model.init_state()

                mon = item['mon'].permute(1, 0, 2)
                week = item['week'].permute(1, 0, 2)
                const = item['const'].permute(0, 1)

                mon = mon.type(torch.cuda.FloatTensor if (torch.cuda.is_available() and cuda) else torch.FloatTensor)
                week = week.type(torch.cuda.FloatTensor if (torch.cuda.is_available() and cuda) else torch.FloatTensor)
                const = const.type(torch.cuda.FloatTensor if (torch.cuda.is_available() and cuda) else torch.FloatTensor)

                # Run model on test set
                outputs, (week_h, week_c), (month_h, month_c) = model(week, mon, const, (week_h, week_c), (month_h, month_c))
                loss = criterion(
                    outputs.type(torch.cuda.FloatTensor if (torch.cuda.is_available() and cuda) else torch.FloatTensor),
                    item['target'].type(torch.cuda.LongTensor if (torch.cuda.is_available() and cuda) else torch.LongTensor)
                )

                # if i % 500 == 0:
                print('Epoch: {}, Test Loss: {}\n'.format(epoch, loss.item()))

                # Save loss info
                total_loss += loss.item()
                test_loss.append(loss.item())

            except RuntimeError as e:
                # For some reason the SubsetRandomSampler makes uneven batch sizes at the end of the batch,
                # so this is done as a workaound.
                print(e, '\nSkipping this mini-batch.')

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
    parser.add_argument('-fd', '--feature_dir', type=str, help='Directory containing features')
    parser.add_argument('-td', '--target_dir', type=str, help='Directory containing targets')
    parser.add_argument('-np', '--num_pixels', type=int, default=10000, help='Number of pixels to sample per image.')
    parser.add_argument('-lt', '--lead_time', type=int, default=4, help='Lead time for making predictions')
    parser.add_argument('-nw', '--n_weeks', type=int, default=20, help='Number of weeks to use for predictions.')
    parser.add_argument('-e', '--epochs', type=int, default=25, help='Number of epochs.')
    parser.add_argument('-bs', '--batch_size', type=int, default=512, help='Batch size to train model with.')
    parser.add_argument('-hs', '--hidden_size', type=int, default=512, help='LSTM hidden dimension size.')
    parser.add_argument('--search', dest='search', action='store_true',
                        help='Perform gridsearch for hyperparameter selection.')
    parser.add_argument('--no-search', dest='search', action='store_false',
                        help='Do not perform gridsearch for hyperparameter selection.')
    parser.add_argument('--cuda', dest='cuda', action='store_true',
                        help='Train model on GPU.')
    parser.add_argument('--no-cuda', dest='cuda', action='store_false',
                        help='Train model on CPU. ')
    parser.set_defaults(search=False)
    parser.set_defaults(cuda=False)

    args = parser.parse_args()

    if args.search:
        for hidden in [32, 64, 128, 256, 512, 1024]:
            for batch in [32, 64, 128, 256, 512, 1024]:
                train_lstm(feature_dir=args.feature_dir, target_dir=args.target_dir,
                           pixel_per_img=args.num_pixels, lead_time=args.lead_time, n_weeks=args.n_weeks,
                           epochs=args.epochs, batch_size=batch, hidden_size=hidden, cuda=args.cuda)

    else:
        try:
            assert 'batch_size' in args and 'hidden_size' in args
            train_lstm(feature_dir=args.feature_dir, target_dir=args.target_dir,
                       pixel_per_img=args.num_pixels, lead_time=args.lead_time, n_weeks=args.n_weeks,
                       epochs=args.epochs, batch_size=args.batch_size, hidden_size=args.hidden_size, cuda=args.cuda)
        except AssertionError as e:
            print('-bs and -hs flags must be used when you are not using the search option.')
