import torch
from torch import nn
import torch.nn.functional as F

# credit to https://discuss.pytorch.org/t/seq2seq-model-with-attention-for-time-series-forecasting/80463/10
# and https://stackoverflow.com/questions/50571991/implementing-luong-attention-in-pytorch

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bidirectional=False, batch_size=1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.batch_size = batch_size

        self.gru = nn.GRU(input_size, hidden_size, num_layers)

    def initHidden(self):
        directions = 2 if self.bidirectional else 1
        return torch.zeros(
            self.num_layers * directions,
            self.batch_size,
            self.hidden_size,
            device=DEVICE
        )

    def forward(self, input):
        hidden = self.initHidden()
        output, hidden = self.gru(input, hidden)
        return output, hidden


class AttnDecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.gru = nn.GRUCell(input_size, hidden_size)
        self.attn = nn.Linear(hidden_size, hidden_size)
        # hc: [hidden, context]
        self.Whc = nn.Linear(hidden_size * 2, hidden_size)
        # s: softmax
        self.Ws = nn.Linear(hidden_size, output_size)

    def forward(self, y, hidden, encoder_outputs):

        hidden = self.gru(y, hidden)

        attn_prod = torch.bmm(self.attn(hidden).unsqueeze(1), encoder_outputs.permute(1, 2, 0)).squeeze(1)
        attn_weights = F.softmax(attn_prod, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs.transpose(0, 1)).squeeze(1)

        # hc: [hidden: context]
        hc = torch.cat([hidden, context], dim=1)
        out_hc = F.tanh(self.Whc(hc))
        output = F.log_softmax(self.Ws(out_hc), dim=1)

        return output, hidden, attn_weights


class Seq2Seq(nn.Module):
    def __init__(self, input_size=1, hidden_size=100, forecast_len=12, categorical=False, batch_size=64):
        super().__init__()
        self.encoder = EncoderRNN(input_size, hidden_size=hidden_size, batch_size=batch_size)
        self.decoder_cell = AttnDecoderRNN(input_size, hidden_size, input_size)
        self.forecast_len = forecast_len

        self.n_classes = 6 if categorical else 1
        self.out = nn.Linear(input_size, self.n_classes)
        self.categorical = categorical

    def forward(self, xb):
        input_seq = xb
        encoder_output, encoder_hidden = self.encoder(input_seq)
        prev_hidden = encoder_hidden

        outputs = []
        y_prev = input_seq[-1, :, :]
        prev_hidden = prev_hidden.squeeze()
        for i in range(self.forecast_len):
            rnn_output, prev_hidden, _ = self.decoder_cell(y_prev, prev_hidden, encoder_output)
            y_prev = rnn_output

            outputs.append(rnn_output)

        outputs = torch.stack(outputs, 1)
        outputs = self.out(outputs)

        outputs = outputs.permute(0, 2, 1) if self.categorical else F.relu(outputs)
        return outputs.squeeze(-1)

