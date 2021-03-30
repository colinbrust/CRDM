from torch import nn
import torch


class LSTM(nn.Module):

    def __init__(self, size=23, hidden_size=64, output_size=6,
                 batch_size=64, cuda=False):
        super().__init__()

        self.hidden_size = hidden_size
        self.device = 'cuda:0' if torch.cuda.is_available() and cuda else 'cpu'
        self.batch_size = batch_size
        self.size = size

        self.encoder_lstm = nn.LSTM(size, self.hidden_size, num_layers=2)
        self.decoder_lstm = nn.LSTM(self.hidden_size, self.hidden_size, num_layers=2)

        classifier = []

        sz = self.hidden_size
        while sz > 32:
            classifier.append(nn.Linear(sz, sz/2))
            classifier.append(nn.BatchNorm1d(sz/2))
            classifier.append(nn.ReLU())
            classifier.append(nn.Dropout(0.25))
            sz /= 2

        classifier.append(nn.Linear(sz, 1))
        classifier.append(nn.ReLU())

        self.classifier = nn.Sequential(*classifier)



    def init_state(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(1, self.batch_size, self.hidden_size, device=self.device),
                torch.zeros(1, self.batch_size, self.hidden_size, device=self.device))

    def forward(self, lstm_seq, prev_state):
        # Run the LSTM forward
        lstm_out, lstm_state = self.lstm(lstm_seq, prev_state)
        preds = self.classifier(lstm_out)
        return preds, lstm_state
