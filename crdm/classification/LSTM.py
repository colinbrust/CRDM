from torch import nn
import torch


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
