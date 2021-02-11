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
                 batch_size=64, const_size=8, cuda=False, num_layers=1):
        super().__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = 'cuda:0' if torch.cuda.is_available() and cuda else 'cpu'
        self.batch_size = batch_size
        self.weekly_size = weekly_size
        self.output_size = output_size

        self.weekly_lstm = nn.LSTM(weekly_size, self.hidden_size, num_layers=num_layers, dropout=0.5)
        self.monthly_lstm = nn.LSTM(monthly_size, self.hidden_size, num_layers=num_layers, dropout=0.5)

        # Downscale to output size
        self.classifier = nn.Sequential(
            nn.Linear(2*hidden_size + const_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.preds2 = nn.Linear(16, self.output_size)
        self.preds4 = nn.Linear(16, self.output_size)
        self.preds6 = nn.Linear(16, self.output_size)
        self.preds8 = nn.Linear(16, self.output_size)

    def init_state(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_size, device=self.device),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_size, device=self.device))

    def forward(self, weekly_seq, monthly_seq, constants, prev_week_state, prev_month_state):
        # Run the LSTM forward

        week_out, week_state = self.weekly_lstm(weekly_seq, prev_week_state)
        month_out, month_state = self.monthly_lstm(monthly_seq, prev_month_state)

        lstm_and_const = torch.cat((week_out[-1], month_out[-1], constants), dim=1)
        preds = self.classifier(lstm_and_const)

        preds2, preds4, preds6, preds8 = self.preds2(preds), self.preds4(preds), self.preds6(preds), self.preds8(preds)

        return [preds2, preds4, preds6, preds8], week_state, month_state


def train_lstm(const_f, week_f, mon_f, target_f, epochs=50, batch_size=64,
               hidden_size=64, cuda=False, init=True, num_layers=1, stateful=False):

    device = 'cuda:0' if torch.cuda.is_available() and cuda else 'cpu'
    info = parse_fname(const_f)

    # Make data loader
    loader = PixelLoader(const_f, week_f, mon_f, target_f, info['init'])

    # Split into training and test sets
    train, test = train_test_split([x for x in range(len(loader))], test_size=0.25)
    train_sampler = SubsetRandomSampler(train)
    test_sampler = SubsetRandomSampler(test)

    train_loader = DataLoader(dataset=loader, batch_size=batch_size, sampler=train_sampler)
    test_loader = DataLoader(dataset=loader, batch_size=batch_size, sampler=test_sampler)

    const_size = loader[0]['const'].shape[-1]

    weekly_size = len(WEEKLY_VARS) + 1 if init else len(WEEKLY_VARS)
    # Define model, loss and optimizer.
    model = LSTM(weekly_size=weekly_size, monthly_size=len(MONTHLY_VARS), hidden_size=hidden_size, output_size=1,
                 batch_size=batch_size, const_size=const_size, cuda=cuda, num_layers=num_layers)

    model.to(device)

    if torch.cuda.is_available() and cuda:
        print('Using GPU')
        model.cuda()

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, threshold=1e-4, verbose=True)

    prev_best_loss = np.inf
    err_out = {}

    f_name_info = {'epochs': str(epochs), 'batch': str(batch_size), 'nWeeks': str(info['nWeeks']),
                   'hiddenSize': str(hidden_size), 'remove': str(info['rmYears']),
                   'init': str(info['init']), 'numLayers': str(num_layers), 'stateful': str(stateful)}
    f_name_info = '_'.join('{}-{}'.format(key, value) for key, value in f_name_info.items())

    out_name_mod = f_name_info + '_fType-model.p'
    out_name_err = f_name_info + '_fType-err.p'

    for epoch in range(epochs):
        total_loss = 0
        train_loss = []
        test_loss = []

        model.train()

        if stateful:
            week_h, week_c = model.init_state()
            month_h, month_c = model.init_state()

        # Loop over each subset of data
        for i, item in enumerate(train_loader, 1):

            try:

                if not stateful:
                    week_h, week_c = model.init_state()
                    month_h, month_c = model.init_state()

                mon = item['mon'].permute(1, 0, 2)
                week = item['week'].permute(1, 0, 2)
                const = item['const'].permute(0, 1)

                mon = mon.type(torch.cuda.FloatTensor if (torch.cuda.is_available() and cuda) else torch.FloatTensor)
                week = week.type(torch.cuda.FloatTensor if (torch.cuda.is_available() and cuda) else torch.FloatTensor)
                const = const.type(torch.cuda.FloatTensor if (torch.cuda.is_available() and cuda) else torch.FloatTensor)

                # Zero out the optimizer's gradient buffer
                optimizer.zero_grad()

                # Make prediction with model
                outputs, (week_h, week_c), (month_h, month_c) = model(week, mon, const, (week_h, week_c), (month_h, month_c))
                targets = (item['target']*5).type(torch.cuda.FloatTensor if cuda else torch.FloatTensor)

                week_h, month_h = week_h.detach(), month_h.detach()
                week_c, month_c = week_c.detach(), week_c.detach()

                print(outputs[0][:5])
                print(targets[:, 1][:5])

                loss2 = criterion(outputs[0], targets[:, 1])
                loss4 = criterion(outputs[1], targets[:, 3])
                loss6 = criterion(outputs[2], targets[:, 5])
                loss8 = criterion(outputs[3], targets[:, 7])

                loss = loss2+loss4+loss6+loss8
                loss.requires_grad = True

                # Compute the loss and step the optimizer
                loss.backward()
                optimizer.step()

                if i % 500 == 0:
                    print('Epoch: {}, Train Loss: {}'.format(epoch, loss))

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

                if not stateful:
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
                # outputs = outputs.type(torch.cuda.FloatTensor if cuda else torch.FloatTensor)
                targets = (item['target']*5).type(torch.cuda.FloatTensor if cuda else torch.FloatTensor)

                week_h, month_h = week_h.detach(), month_h.detach()
                week_c, month_c = week_c.detach(), month_c.detach()

                loss2 = criterion(torch.argmax(outputs[0], dim=-1), targets[:, 1])
                loss4 = criterion(torch.argmax(outputs[1], dim=-1), targets[:, 3])
                loss6 = criterion(torch.argmax(outputs[2], dim=-1), targets[:, 5])
                loss8 = criterion(torch.argmax(outputs[3], dim=-1), targets[:, 7])

                loss = loss2+loss4+loss6+loss8

                if i % 500 == 0:
                    print('Epoch: {}, Train Loss: {}'.format(epoch, loss))

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
    parser.add_argument('-nl', '--num_layers', type=int, default=1, help='Number of stacked LSTM layers to use.')
    parser.add_argument('--search', dest='search', action='store_true',
                        help='Perform gridsearch for hyperparameter selection.')
    parser.add_argument('--no-search', dest='search', action='store_false',
                        help='Do not perform gridsearch for hyperparameter selection.')
    parser.add_argument('--cuda', dest='cuda', action='store_true',
                        help='Train model on GPU.')
    parser.add_argument('--no-cuda', dest='cuda', action='store_false',
                        help='Train model on CPU. ')
    parser.add_argument('--state', dest='state', action='store_true', help='Maintain state across batches.')
    parser.add_argument('--no-state', dest='state', action='store_false', help='Reset state after each mini-batch.')
    parser.set_defaults(search=False)
    parser.set_defaults(cuda=False)
    parser.set_defaults(state=False)

    args = parser.parse_args()
    infile = open(args.pickle_f, 'rb')
    pick = pickle.load(infile)
    infile.close()

    week_f = os.path.join(os.path.dirname(args.pickle_f), pick['featType-weekly'])
    mon_f = os.path.join(os.path.dirname(args.pickle_f), pick['featType-monthly'])
    const_f = os.path.join(os.path.dirname(args.pickle_f), pick['featType-constant'])
    target_f = os.path.join(os.path.dirname(args.pickle_f), pick['featType-target'])

    info = parse_fname(week_f)
    init = info['init']

    if args.search:
        for layers in [1, 2, 3, 4]:
            train_lstm(const_f=const_f, mon_f=mon_f, week_f=week_f, target_f=target_f, epochs=args.epochs,
                       batch_size=args.batch_size, hidden_size=args.hidden_size, cuda=args.cuda, init=init, num_layers=layers)

    else:
        try:
            assert 'batch_size' in args and 'hidden_size' in args
            train_lstm(const_f=const_f, mon_f=mon_f, week_f=week_f, target_f=target_f, epochs=args.epochs,
                       batch_size=args.batch_size, hidden_size=args.hidden_size, cuda=args.cuda, init=init, num_layers=args.num_layers)
        except AssertionError as e:
            print('-bs and -hs flags must be used when you are not using the search option.')
