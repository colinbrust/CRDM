import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split


class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, constant_size=18, output_size=1, batch_size=64):
        super().__init__()

        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, hidden_size)

        self.upscale = nn.Sequential(
          nn.Linear(constant_size, 32),
          nn.ReLU(),
          nn.Linear(32, 64),
          nn.ReLU()
        )

        self.classifier = nn.Sequential(
          nn.Linear(hidden_size, 128),
          nn.ReLU(),
          nn.Linear(128, 256),
          nn.ReLU(),
          nn.Linear(256, 128),
          nn.ReLU(),
          nn.Linear(128, 64),
          nn.ReLU(),
          nn.Linear(64, 32),
          nn.ReLU(),
          nn.Linear(32, output_size)
        )

        self.hidden_cell = (torch.zeros(1,batch_size,self.hidden_size),
                            torch.zeros(1,batch_size,self.hidden_size))

    def forward(self, input_seq, constants):
        print('made it nowhere')
        lstm_out, self.hidden_cell = self.lstm(input_seq, self.hidden_cell)
        print('made it here')
        constants = torch.unsqueeze(constants, 0).expand(lstm_out.shape[0], -1, -1)

        constants = self.upscale(constants)
        print(constants.shape)
        print(lstm_out.shape)
        lstm_and_const = torch.cat((lstm_out, constants))
        print('made it here2')
        preds = self.classifier(lstm_and_const)
        print('asdf')
        return preds


class PixelLoader(Dataset):
    
    def __init__(self, constant, monthly, target):
        self.constant = np.memmap(constant, dtype='float32', mode='c', shape=(461000, 18))
        self.constant = np.nan_to_num(self.constant, nan=-0.5)
        self.monthly = np.memmap(monthly, dtype='float32', mode='c', shape=(461000, 13, 12))
        self.monthly = np.nan_to_num(self.monthly, nan=-0.5)
        self.target = np.memmap(target, dtype='int8', mode='c')

    def __len__(self):
        return len(self.target)
    
    def __getitem__(self, idx):
        return {'const': torch.tensor(self.constant[idx]), 
                'mon': torch.tensor(self.monthly[idx]),
                'target': self.target[idx]}


def train_lstm(const_f, mon_f, target_f, epochs=1000, batch_size=64):

    # const_f ='/Users/colinbrust/projects/CRDM/data/drought/premade/constant_pixelPremade_nMonths-13_leadTime-2_size-1000.dat'
    # mon_f = '/Users/colinbrust/projects/CRDM/data/drought/premade/monthly_pixelPremade_nMonths-13_leadTime-2_size-1000.dat'
    # target_f = '/Users/colinbrust/projects/CRDM/data/drought/premade/target_pixelPremade_nMonths-13_leadTime-2_size-1000.dat'
    # epochs = 1000
    # batch_size = 64
     
    loader = PixelLoader(const_f, mon_f, target_f)
    train, test = train_test_split([x for x in range(len(loader))], test_size=0.25)
    
    train_loader = DataLoader(dataset=loader, batch_size=batch_size, shuffle=True)

    model = LSTM(input_size=13, hidden_size=64, constant_size=loader.constant.shape[1], output_size=6)
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):

        model.train()
        # Loop over each subset of data
        for item in train_loader:

            mon = item['mon'].permute(2, 0, 1)
            const = item['const'].permute(0, 1)

            print(mon.shape)
            # Zero out the optimizer's gradient buffer
            optimizer.zero_grad()
            # Make a prediction based on the model
            outputs = model(mon, const)
            # Compute the loss
            print(outputs.shape)
            print(item['target'].shape)
            loss = criterion(outputs.permute(1, 0, 2), item['target'])
            # Use backpropagation to compute the derivative of the loss with respect to the parameters
            loss.backward()
            # Use the derivative information to update the parameters
            optimizer.step()
            # Print the epoch, the training loss, and the test set accuracy.
            print('Epoch: {}, Train Loss: {}\n'.format(epoch, loss.item()))
