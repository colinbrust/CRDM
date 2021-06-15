import torch
from torch import nn
import torch.nn.functional as F

# Some methods borrowed from
# https://discuss.pytorch.org/t/seq2seq-model-with-attention-for-time-series-forecasting/80463/10

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Encoder that 
class RNNEncoder(nn.Module):
    
    def __init__(self, num_layers=1, input_size=1, hidden_size=100, bidirectional=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.rnn_directions = 2 if bidirectional else 1
        self.gru = nn.GRU(
            num_layers=num_layers,
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=bidirectional
        )

    def forward(self, input_seq):

        ht = torch.zeros(self.num_layers * self.rnn_directions, input_seq.size(0), self.hidden_size, device=device)

        gru_out, hidden = self.gru(input_seq, ht)

        if self.rnn_directions > 1:
            gru_out = gru_out.view(input_seq.size(0), input_seq.shape[1], self.rnn_directions, self.hidden_size)
            gru_out = torch.sum(gru_out, dim=2)

        return gru_out, hidden.squeeze(0)


class DecoderCell(nn.Module):

    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.decoder_rnn_cell = nn.GRUCell(
            input_size=input_size,
            hidden_size=hidden_size,
        )

        self.out = nn.Linear(hidden_size, input_size)

    def forward(self, prev_y, prev_hidden):

        rnn_hidden = self.decoder_rnn_cell(prev_y, prev_hidden)
        output = self.out(rnn_hidden)
        return output, rnn_hidden


class Seq2Seq(nn.Module):
    def __init__(self, num_layers=1, input_size=1, n_weeks=168, hidden_size=100, lead_time=12):
        super().__init__()

        # Add some linear layers before data are passed into encoder/decoder framework
        enc_linear = []
        sz = input_size
        sz_2 = 64

        while sz < hidden_size:
            enc_linear.append(nn.Linear(int(sz), int(sz_2)))
            enc_linear.append(nn.BatchNorm1d(n_weeks))
            enc_linear.append(nn.ReLU())
            enc_linear.append(nn.Dropout(0.5))
            sz = sz_2
            sz_2 *= 2

        self.enc_linear = nn.Sequential(*enc_linear)

        self.encoder = RNNEncoder(num_layers, input_size=hidden_size, hidden_size=hidden_size)
        self.decoder_cell = DecoderCell(input_size=hidden_size, hidden_size=hidden_size)
        self.lead_time = lead_time

        # Add some linear layers before making the prediction.
        classifier = []
        sz = hidden_size
        while sz > 32:
            classifier.append(nn.Linear(int(sz), int(sz//2)))
            classifier.append(nn.BatchNorm1d(self.lead_time))
            classifier.append(nn.ReLU())
            classifier.append(nn.Dropout(0.5))
            sz /= 2

        classifier.append(nn.Linear(int(sz), 1))
        classifier.append(nn.ReLU())

        self.classifier = nn.Sequential(*classifier)

    def forward(self, xb):

        input_seq = self.enc_linear(xb)
        encoder_output, encoder_hidden = self.encoder(input_seq)
        prev_hidden = encoder_hidden

        outputs = []
        y_prev = input_seq[:, -1, :]
        for i in range(self.lead_time):
            rnn_output, prev_hidden = self.decoder_cell(y_prev, prev_hidden)

            y_prev = rnn_output

            outputs.append(rnn_output)

        outputs = torch.stack(outputs, 1)

        outputs = F.relu(outputs)
        outputs = self.classifier(outputs)

        return outputs.squeeze(-1)
