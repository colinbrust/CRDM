import argparse
from collections import Counter
from crdm.loaders.PixelLoader import PixelLoader
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
        self.device = 'cuda:0' if torch.cuda.is_available() and cuda else 'cpu'
        self.batch_size = batch_size
        self.weekly_size = weekly_size

        self.weekly_lstm = nn.LSTM(weekly_size, self.hidden_size)
        self.monthly_lstm = nn.LSTM(monthly_size, self.hidden_size)

        # Downscale to output size
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_size + const_size, 1024),
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


def train_lstm(const_f, week_f, mon_f, target_f, epochs=50, batch_size=64, hidden_size=64, cuda=False, mod_f=None,
               retrain_run=0):
    device = 'cuda:0' if torch.cuda.is_available() and cuda else 'cpu'
    info = parse_fname(const_f)
    lead_time = info['leadTime']

    # Make data loader
    loader = PixelLoader(const_f, week_f, mon_f, target_f, True)

    # Split into training and test sets
    train, test = train_test_split([x for x in range(len(loader))], test_size=0.25)
    train_sampler = SubsetRandomSampler(train)
    test_sampler = SubsetRandomSampler(test)

    train_loader = DataLoader(dataset=loader, batch_size=batch_size, sampler=train_sampler)
    test_loader = DataLoader(dataset=loader, batch_size=batch_size, sampler=test_sampler)

    const_size = loader[0]['const'].shape[-1]

    weekly_size = len(WEEKLY_VARS) + 1
    # Define model, loss and optimizer.

    if mod_f is not None:
        model = LSTM(weekly_size=weekly_size, monthly_size=len(MONTHLY_VARS), hidden_size=hidden_size, output_size=1,
                     batch_size=batch_size, const_size=const_size, cuda=cuda)

        model.load_state_dict(torch.load(mod_f)) if cuda and torch.cuda.is_available() else model.load_state_dict(
            torch.load(mod_f, map_location=torch.device('cpu')))

    else:
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

    if mod_f is not None:
        assert retrain_run != 0, 'If you are retraining the model, you must specify which number rerun this is ' \
                                  'using the -rr flag. '

    out_name_mod = 'epochs-{}_batch-{}_nMonths-{}_hiddenSize-{}_leadTime-{}_remove-{}_init-{}_rr-{}_fType-model.p'.format(
        epochs, batch_size, info['nWeeks'], hidden_size, lead_time, info['rmYears'], info['init'], retrain_run)
    out_name_err = 'epochs-{}_batch-{}_nMonths-{}_hiddenSize-{}_leadTime-{}_remove-{}_init-{}_rr-{}_fType-err.p'.format(
        epochs, batch_size, info['nWeeks'], hidden_size, lead_time, info['rmYears'], info['init'], retrain_run)

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

                mon = item['mon'].permute(1, 0, 2)
                week = item['week'].permute(1, 0, 2)
                const = item['const'].permute(0, 1)

                mon = mon.type(torch.cuda.FloatTensor if (torch.cuda.is_available() and cuda) else torch.FloatTensor)
                week = week.type(torch.cuda.FloatTensor if (torch.cuda.is_available() and cuda) else torch.FloatTensor)
                const = const.type(
                    torch.cuda.FloatTensor if (torch.cuda.is_available() and cuda) else torch.FloatTensor)

                # Zero out the optimizer's gradient buffer
                optimizer.zero_grad()

                # model.hidden = model.init_hidden()

                # Make prediction with model
                outputs, (week_h, week_c), (month_h, month_c) = model(week, mon, const, (week_h, week_c),
                                                                      (month_h, month_c))
                outputs = outputs.squeeze()

                week_h, month_h = week_h.detach(), month_h.detach()
                week_c, month_c = week_c.detach(), week_c.detach()

                # Compute the loss and step the optimizer
                target = item['target'] / 5
                outputs = outputs.squeeze()

                loss = criterion(outputs.type(dt), target.type(dt))

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

                mon = item['mon'].permute(1, 0, 2)
                week = item['week'].permute(1, 0, 2)
                const = item['const'].permute(0, 1)

                mon = mon.type(torch.cuda.FloatTensor if (torch.cuda.is_available() and cuda) else torch.FloatTensor)
                week = week.type(torch.cuda.FloatTensor if (torch.cuda.is_available() and cuda) else torch.FloatTensor)
                const = const.type(
                    torch.cuda.FloatTensor if (torch.cuda.is_available() and cuda) else torch.FloatTensor)

                # Run model on test set
                outputs, (week_h, week_c), (month_h, month_c) = model(week, mon, const, (week_h, week_c),
                                                                      (month_h, month_c))

                target = item['target'] / 5
                outputs = outputs.squeeze()

                loss = criterion(outputs.type(dt), target.type(dt))

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
    parser.add_argument('-p', '--pickle_f', type=str, help='File of memmap of constants.')
    parser.add_argument('-e', '--epochs', type=int, default=25, help='Number of epochs.')
    parser.add_argument('-bs', '--batch_size', type=int, help='Batch size to train model with.')
    parser.add_argument('-hs', '--hidden_size', type=int, help='LSTM hidden dimension size.')
    parser.add_argument('-mf', '--model_file', type=str, default=None, help='Location of pretrained model pickle.')
    parser.add_argument('-rr', '--retrain_run', type=int, default=0, help='Number of times model has been retrained')

    parser.add_argument('--cuda', dest='cuda', action='store_true',
                        help='Train model on GPU.')
    parser.add_argument('--no-cuda', dest='cuda', action='store_false',
                        help='Train model on CPU. ')
    parser.set_defaults(cuda=False)

    args = parser.parse_args()
    infile = open(args.pickle_f, 'rb')
    pick = pickle.load(infile)
    infile.close()

    week_f = os.path.join(os.path.dirname(args.pickle_f), pick['featType-weekly'])
    mon_f = os.path.join(os.path.dirname(args.pickle_f), pick['featType-monthly'])
    const_f = os.path.join(os.path.dirname(args.pickle_f), pick['featType-constant'])
    target_f = os.path.join(os.path.dirname(args.pickle_f), pick['featType-target'])

    train_lstm(const_f=const_f, mon_f=mon_f, week_f=week_f, target_f=target_f,
               epochs=args.epochs, batch_size=args.batch_size, hidden_size=args.hidden_size,
               cuda=args.cuda, mod_f=args.model_file, retrain_run=args.retrain_run)