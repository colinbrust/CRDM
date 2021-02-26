import argparse
from collections import Counter
from crdm.loaders.DataStack import DataStack
from crdm.utils.ImportantVars import MONTHLY_VARS, WEEKLY_VARS
from crdm.utils.ParseFileNames import parse_fname
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

        print(self.hidden_size)
        self.device = 'cuda:0' if torch.cuda.is_available() and cuda else 'cpu'
        self.batch_size = batch_size
        self.weekly_size = weekly_size

        self.weekly_lstm = nn.LSTM(weekly_size, self.hidden_size)
        self.monthly_lstm = nn.LSTM(monthly_size, self.hidden_size)

        # Downscale to output size
        self.classifier = nn.Sequential(
            nn.Linear(2*hidden_size + const_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.25),
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


def train_lstm(data_dir, epochs=50, batch_size=64, hidden_size=64, cuda=False, init=True, n_weeks=25, lead_time=None):

    device = 'cuda:0' if torch.cuda.is_available() and cuda else 'cpu'

    # Make data loader
    train_stack = DataStack(
        data_dir=data_dir, n_weeks=n_weeks, train='train', cuda=cuda, lead_time=lead_time, num_samples=batch_size*1000
    )

    test_stack = DataStack(
        data_dir=data_dir, n_weeks=n_weeks, train='test', cuda=cuda, lead_time=lead_time, num_samples=batch_size*500
    )

    # Split into training and test sets
    train_sampler = torch.utils.data.sampler.BatchSampler(
        torch.utils.data.sampler.RandomSampler(train_stack),
        batch_size=batch_size,
        drop_last=False)

    test_sampler = torch.utils.data.sampler.BatchSampler(
        torch.utils.data.sampler.RandomSampler(test_stack),
        batch_size=batch_size,
        drop_last=False)

    train_loader = DataLoader(dataset=train_stack, sampler=train_sampler)
    test_loader = DataLoader(dataset=test_stack, sampler=test_sampler)

    test = train_stack[[0]]
    const_size = test_stack[[0]]['const'].shape[0]

    weekly_size = len(WEEKLY_VARS)
    # Define model, loss and optimizer.
    model = LSTM(weekly_size=weekly_size, monthly_size=len(MONTHLY_VARS), hidden_size=hidden_size, output_size=1,
                 batch_size=batch_size, const_size=const_size, cuda=cuda)

    model.to(device)

    if torch.cuda.is_available() and cuda:
        print('Using GPU')
        model.cuda()

    # Provide relative frequency weights to use in loss function.
    criterion = nn.MSELoss()
    lr = 3e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, threshold=1e-5, verbose=True)

    prev_best_loss = 1e6
    err_out = {}

    out_name_mod = 'epochs-{}_batch-{}_nMonths-{}_hiddenSize-{}_leadTime-{}_fType-model.p'.format(
        epochs, batch_size, n_weeks, hidden_size, lead_time)
    out_name_err = 'epochs-{}_batch-{}_nMonths-{}_hiddenSize-{}_leadTime-{}_fType-err.p'.format(
        epochs, batch_size, n_weeks, hidden_size, lead_time)

    dt = torch.cuda.FloatTensor if (torch.cuda.is_available() and cuda) else torch.FloatTensor

    for epoch in range(epochs):
        total_loss = 0
        train_loss = []
        test_loss = []

        model.train()

        # Loop over each subset of data
        for i, item in enumerate(train_loader, 1):

            try:
                week_h, week_c = model.init_state()
                month_h, month_c = model.init_state()

                # time, batch, features
                mon = item['mon'].squeeze().permute(0, 2, 1)
                # time, batch, features
                week = item['week'].squeeze().permute(0, 2, 1)
                # batch, features
                const = item['const'].squeeze().permute(1, 0)

                # Zero out the optimizer's gradient buffer
                optimizer.zero_grad()

                # model.hidden = model.init_hidden()

                # Make prediction with model
                outputs, (week_h, week_c), (month_h, month_c) = model(week, mon, const, (week_h, week_c), (month_h, month_c))

                week_h, month_h = week_h.detach(), month_h.detach()
                week_c, month_c = week_c.detach(), week_c.detach()
                
                # Compute the loss and step the optimizer
                target = item['target'].squeeze()/5 
                outputs = outputs.squeeze()

                loss = criterion(outputs, target)
                loss.backward()
                optimizer.step()

                if i % 250 == 0:
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


                # time, batch, features
                mon = item['mon'].squeeze().permute(0, 2, 1)
                # time, batch, features
                week = item['week'].squeeze().permute(0, 2, 1)
                # batch, features
                const = item['const'].squeeze().permute(1, 0)
                outputs, (week_h, week_c), (month_h, month_c) = model(week, mon, const, (week_h, week_c), (month_h, month_c))

                target = item['target'].squeeze()/5 
                outputs = outputs.squeeze()

                loss = criterion(outputs, target)
                    
                if i % 250 == 0:
                    print('Epoch: {}, Test Loss: {}'.format(epoch, loss.item()))

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
    parser.add_argument('-d', '--data_dir', type=str, help='Directory containing premade model data.')
    parser.add_argument('-e', '--epochs', type=int, default=25, help='Number of epochs.')
    parser.add_argument('-bs', '--batch_size', type=int, help='Batch size to train model with.')
    parser.add_argument('-hs', '--hidden_size', type=int, help='LSTM hidden dimension size.')
    parser.add_argument('-lt', '--lead_time', type=int, help='Lead time to use. Use 9999 to train model with all lead times.')
    parser.add_argument('--cuda', dest='cuda', action='store_true',
                        help='Train model on GPU.')
    parser.add_argument('--no-cuda', dest='cuda', action='store_false',
                        help='Train model on CPU. ')
    parser.set_defaults(search=False)
    parser.set_defaults(cuda=False)
    parser.set_defaults(batch_size=1024)
    parser.set_defaults(hidden_size=1024)
    parser.set_defaults(lead_time=9999)

    args = parser.parse_args()

    train_lstm(data_dir=args.data_dir, epochs=args.epochs, batch_size=args.batch_size, hidden_size=args.hidden_size, cuda=args.cuda, init=True,
               lead_time=args.lead_time, n_weeks=25)
