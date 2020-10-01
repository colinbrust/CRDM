import sys
import numpy as np
from collections import Counter
import torch
import torch.nn as nn
import pickle
import os
from pathlib import Path
import argparse
import copy
from sklearn.model_selection import train_test_split
from crdm.classification.UNetParts import DoubleConv, Down, Up, OutConv
from crdm.loaders.CroppedLoader import CroppedLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        factor = 2 if bilinear else 1

        # Do additional downscaling if we have more than 128 input features
        mid_channels = 128 if n_channels > 128 else n_channels
        self.inc = DoubleConv(n_channels, 64, mid_channels=mid_channels)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

target_dir = '/Users/colinbrust/projects/CRDM/data/drought/out_classes/out_memmap'
in_features = '/Users/colinbrust/projects/CRDM/data/drought/in_features'
out_dir = '/Users/colinbrust/projects/CRDM/data/drought/premade/unet'
lead_time = 1
n_months = 5
crop_size = 64
crops_per_img = 75
rm_features=  True

def train_model(target_dir, in_features, lead_time=1, n_months=5, crop_size=64, 
                crops_per_img=50, rm_features=True, batch_size=64, epochs=50):

    # Make data loader
    loader = CroppedLoader(in_features=in_features, target_dir=target_dir, lead_time=lead_time, n_months=n_months, 
                           crop_size=crop_size, crops_per_img=crops_per_img, rm_features=rm_features)

    # Split into training and test sets
    train, test = train_test_split([x for x in range(len(loader))], test_size=0.25)
    train_sampler = SubsetRandomSampler(train)
    test_sampler = SubsetRandomSampler(test)

    train_loader = DataLoader(dataset=loader, batch_size=batch_size, sampler=train_sampler)
    test_loader = DataLoader(dataset=loader, batch_size=batch_size, sampler=test_sampler)

    # Define model, loss and optimizer.
    model = UNet(n_channels=loader[0]['feats'].shape[0], n_classes=6)

    model.to(device)

    # if torch.cuda.is_available():
    #     print('Using GPU')
    #     model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    prev_best_loss = 1e6
    err_out = {}

    out_name_mod = 'modelType-UNet_epochs-{}_batch-{}_nMonths-{}_leadTime-{}_cropSize-{}_rmFeatures-{}_fType-model.p'.format(epochs, batch_size, n_months, lead_time, crop_size, rm_features)
    out_name_err = 'modelType-UNet_epochs-{}_batch-{}_nMonths-{}_leadTime-{}_cropSize-{}_rmFeatures-{}_fType-err.p'.format(epochs, batch_size, n_months, lead_time, crop_size, rm_features)

    for epoch in range(epochs):
        total_loss = 0
        train_loss = []
        test_loss = []
        model.train()

        # Loop over each subset of data
        for i, item in enumerate(train_loader, 1):

            try:

                feats = item['feats'].type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor)
                target = item['target'].type(torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor)

                # Zero out the optimizer's gradient buffer
                optimizer.zero_grad()

                # Make prediction with model
                outputs = model(feats)

                # Compute the loss and step the optimizer
                loss = criterion(outputs.type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), 
                                 target)
                loss.backward()
                optimizer.step() 

                if i % 500 == 0:
                    print('Epoch: {}, Train Loss: {}'.format(epoch, loss.item()))
            
                # Store loss info
                train_loss.append(loss.item())

            except RuntimeError as e:
                # For some reason the SubsetRandomSampler makes uneven batch sizes at the end of the batch, so this is done as a workaound.
                print(e, '\nSkipping this mini-batch.')

        # Switch to evaluation mode
        model.eval()
        for i, item in enumerate(test_loader, 1):

            try:

                feats = item['feats'].type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor)
                target = item['target'].type(torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor)

                # Run model on test set
                outputs = model(feats)
                loss = criterion(outputs.type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), 
                                 target)

                if i % 500 == 0:
                    print('Epoch: {}, Test Loss: {}\n'.format(epoch, loss.item()))
                
                # Save loss info
                total_loss += loss.item()
                test_loss.append(loss.item())

            except RuntimeError as e:
                # For some reason the SubsetRandomSampler makes uneven batch sizes at the end of the batch, so this is done as a workaound.
                print(e, '\nSkipping this mini-batch.')
        
        # If our new loss is better than old loss, save the model
        if prev_best_loss > total_loss:
            torch.save(model.state_dict(), out_name_mod)
            prev_best_loss = total_loss

        # Save out train and test set loss. 
        err_out[epoch] = {'train': train_loss,
                            'test': test_loss}

        with open(out_name_err, 'wb') as f:
            pickle.dump(err_out, f)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train Drought Prediction Model')
    
    parser.add_argument('-td', '--target_dir', type=str, help='Path to dir of memmap targets.')
    parser.add_argument('-if', '--in_features', type=str, help='Path to dir of input features.')
    parser.add_argument('-lt', '--lead_time', type=int, default=1, help='Number of months in advance to make predictions.')
    parser.add_argument('-e', '--epochs', type=int, default=25, help='Number of epochs.')
    parser.add_argument('-nm', '--n_months', type=int, default=5, help='Number of months to use for predictions.')
    parser.add_argument('-cs', '--crop_size', type=int, default=64, help='Size of crop to sample from input features/targets.')
    parser.add_argument('-cpi', '-crops_per_img', type=int, default=64, help='Number of crops to take per image.')
    parser.add_argument('-bs', '--batch_size', type=int, default=64, help='Batch size to train model with.')
    parser.add_argument('--search', dest='search', action='store_true', help='Perform gridsearch for hyperparameter selection.')
    parser.add_argument('--no-search', dest='search', action='store_false', help='Do not perform gridsearch for hyperparameter selection.')
    parser.set_defaults(search=False)

    args = parser.parse_args()

    if args.search:
        for batch in [32, 64, 128, 256, 512]:
            train_model(args.target_dir, args.in_features, args.lead_time, args.n_months,
                        args.crop_size, args.crops_per_img, True, batch, args.epochs)

    else:  
            train_model(args.target_dir, args.in_features, args.lead_time, args.n_months,
                    args.crop_size, args.crops_per_img, True, args.batch, args.epochs)
