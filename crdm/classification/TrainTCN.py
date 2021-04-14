from crdm.classification.TCN import TCN
import argparse
from crdm.loaders.TemporalLoader import DroughtLoader
from crdm.utils.MakeModelDir import make_model_dir
import os
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pickle


FEATS = ('pr', 'vpd', 'tmmx', 'sm-rootzone', 'vapor', 'ET', 'gpp', 'vs', 'vod', 'USDM')
NUM_CONST = 17
NUM_FEATS = 17


def train_lstm(feature_dir, const_dir, n_weeks=25, epochs=50, batch_size=64, mx_lead=12,
               hidden_size=32, kernel_size=7, n_layers=8, feats=FEATS):

    test_loader = DroughtLoader(feature_dir, const_dir, False, mx_lead, n_weeks, batch_size, feats)
    train_loader = DroughtLoader(feature_dir, const_dir, True, mx_lead, n_weeks, batch_size, feats)

    size = NUM_FEATS if feats[0] == '*' else len(feats)

    # Define model, loss and optimizer.
    model = TCN(input_size=size, output_size=1, num_channels=[hidden_size] * 8,
                kernel_size=kernel_size, mx_lead=mx_lead, dropout=0.2)

    if torch.cuda.is_available():
        print('Using GPU')
        model.cuda()

    # Provide relative frequency weights to use in loss function.
    criterion = nn.MSELoss()
    lr = 0.002
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=10, threshold=1e-5, verbose=True, factor=0.5)

    prev_best_loss = 1e6
    err_out = {}

    pth = make_model_dir()

    metadata = {'epochs': epochs,
                'batch_size': batch_size,
                'hidden_size': hidden_size,
                'n_weeks': n_weeks,
                'mx_lead': mx_lead,
                'kernel_size': kernel_size,
                'n_layers': n_layers,
                'feats': feats,
                'num_const': NUM_CONST}

    with open(os.path.join(pth, 'metadata.p'), 'wb') as f:
        pickle.dump(metadata, f)

    for epoch in range(epochs):
        total_loss = 0
        train_loss = []
        test_loss = []

        model.train()

        # Loop over each subset of data
        for i in range(len(train_loader)):
            x, _, y = train_loader[i]
            # Zero out the optimizer's gradient buffer
            optimizer.zero_grad()

            # Make prediction with model
            outputs = model(x.permute(0, 2, 1))
            outputs = outputs.squeeze()
            # Compute the loss and step the optimizer
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            print('Epoch: {}, Train Loss: {}'.format(epoch, loss.item()))

            # Store loss info
            train_loss.append(loss.item())

        # Switch to evaluation mode
        model.eval()

        for i, item in enumerate(test_loader, 1):
            x, _, y = item
            # Zero out the optimizer's gradient buffer
            optimizer.zero_grad()

            # Make prediction with model
            outputs = model(x.permute(0, 2, 1))
            outputs = outputs.squeeze()
            # Compute the loss and step the optimizer
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

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
    parser.add_argument('-ks', '--kernel_size', type=int, default=7, help='Filter kernel size.')
    parser.add_argument('-nl', '--n_layers', type=int, default=7, help='Number of residual layers in network.')
    parser.add_argument('-nw', '--n_weeks', type=int, default=25, help='Number of week history to use for prediction')
    parser.add_argument('-mx', '--max_lead', type=int, default=8,
                        help='How many weeks into the future to make predictions.')

    parser.add_argument('--search', dest='search', action='store_true', help='Perform hyperparameter grid search.')
    parser.add_argument('--no-search', dest='search', action='store_false',
                        help="Don't perform hyperparameter grid search.")
    parser.set_defaults(search=False)

    args = parser.parse_args()
    if args.search:

        week_list = [15, 30, 50, 75, 100]
        for week in week_list:
            print('Grid search with n_weeks={}'.format(week))
            train_lstm(feature_dir=args.feature_dir, const_dir=args.const_dir, epochs=10,
                       batch_size=args.batch_size, hidden_size=args.hidden_size,
                       n_weeks=week, mx_lead=12, feats=['*'])

        hidden_list = [20, 25, 30, 35]
        layer_list = [5, 10, 15, 20]
        kern_list = [4, 5, 6, 7, 8]

        for hidden in hidden_list:
            for layer in layer_list:
                for kern in kern_list:
                    print('Grid search with hidden_size={}'.format(hidden))
                    train_lstm(feature_dir=args.feature_dir, const_dir=args.const_dir, epochs=10,
                               batch_size=512, n_weeks=25, mx_lead=12, feats=['*'],
                               kernel_size=kern, n_layers=layer, hidden_size=hidden)

    else:
        train_lstm(feature_dir=args.feature_dir, const_dir=args.const_dir, epochs=args.epochs,
                   batch_size=args.batch_size, hidden_size=args.hidden_size,
                   n_weeks=args.n_weeks, mx_lead=args.max_lead, feats=['*'])



# feat_list = [['*'], ['pr', 'USDM'], ['pr', 'vpd', 'tmmx', 'sm-rootzone', 'USDM']]
# for feats in feat_list:
#     print('Grid search with feat_list={}'.format(feats))
#     train_lstm(feature_dir=args.feature_dir, const_dir=args.const_dir, epochs=15,
#                batch_size=args.batch_size, hidden_size=args.hidden_size,
#                n_weeks=25, mx_lead=args.max_lead, feats=feats)