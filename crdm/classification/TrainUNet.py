import argparse
from crdm.loaders.LoaderUNet import CroppedLoader
from crdm.classification.UNet import UNet
import pickle
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import nn


def train_model(target_dir, in_features, epochs=50, batch_size=64, n_weeks=25):

    test_loader = CroppedLoader(target_dir=target_dir, in_features=in_features,
                                batch_size=batch_size, n_weeks=n_weeks,
                                cuda=torch.cuda.is_available(), test=True, crop_size=32)

    train_loader = CroppedLoader(target_dir=target_dir, in_features=in_features,
                                 batch_size=batch_size, n_weeks=n_weeks,
                                 cuda=torch.cuda.is_available(), test=False, crop_size=32)

    sample_dims = test_loader[0][0].shape

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
    model = UNet(n_channels=sample_dims[1], n_classes=1)

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

    out_name_mod = 'epochs-{}_batch-{}_nMonths-{}_fType-model.p'.format(
        epochs, batch_size, n_weeks)
    out_name_err = 'epochs-{}_batch-{}_nMonths-{}_fType-err.p'.format(
        epochs, batch_size, n_weeks)

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
            outputs = model(features)
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
            outputs = model(features)
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

    parser.set_defaults(batch_size=1024)


    args = parser.parse_args()

    train_model(target_dir=args.target_dir, in_features=args.in_features, epochs=args.epochs,
                batch_size=args.batch_size, n_weeks=25)

