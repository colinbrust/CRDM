from torch import nn
import torch


class LSTM(nn.Module):

    def __init__(self, size=23, hidden_size=64, batch_size=64, mx_lead=12, const_size=15):
        super().__init__()

        self.hidden_size = hidden_size
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.batch_size = batch_size
        self.size = size
        self.mx_lead = mx_lead
        self.const_size = const_size

        self.lstm = nn.LSTM(size, self.hidden_size, num_layers=2, batch_first=True)

        classifier = []
        sz = self.hidden_size
        while sz > 32:
            classifier.append(nn.Linear(sz+self.const_size, sz//2))
            classifier.append(nn.BatchNorm1d(sz//2))
            classifier.append(nn.ReLU())
            classifier.append(nn.Dropout(0.25))
            sz /= 2

        classifier.append(nn.Linear(sz, 1))
        classifier.append(nn.ReLU())

    def init_state(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(1, self.batch_size, self.hidden_size, device=self.device),
                torch.zeros(1, self.batch_size, self.hidden_size, device=self.device))

    def forward(self, lstm_seq, const, prev_state):
        # Run the LSTM forward
        lstm_out, lstm_state = self.lstm(lstm_seq, prev_state)
        lstm_out = lstm_out[:, -self.mx_lead:, :]
        lstm_and_const = torch.cat([lstm_out, const], dim=-1)
        preds = self.classifier(lstm_and_const)
        preds = preds.squeeze()
        return preds, lstm_state
