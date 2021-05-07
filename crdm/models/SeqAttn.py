import torch
from torch import nn
import torch.nn.functional as F

# credit to https://discuss.pytorch.org/t/seq2seq-model-with-attention-for-time-series-forecasting/80463/10

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class RNNEncoder(nn.Module):
    def __init__(self, rnn_num_layers=1, input_feature_len=1, sequence_len=168, hidden_size=100, bidirectional=False):
        super().__init__()
        self.sequence_len = sequence_len
        self.hidden_size = hidden_size
        self.input_feature_len = input_feature_len
        self.num_layers = rnn_num_layers
        self.rnn_directions = 2 if bidirectional else 1
        self.gru = nn.GRU(
            num_layers=rnn_num_layers,
            input_size=input_feature_len,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=bidirectional
        )

    def forward(self, input_seq):
        ht = torch.zeros(self.num_layers * self.rnn_directions, input_seq.size(0), self.hidden_size, device=device)
        if input_seq.ndim < 3:
            input_seq.unsqueeze_(2)
        gru_out, hidden = self.gru(input_seq, ht)
        if self.rnn_directions > 1:
            gru_out = gru_out.view(input_seq.size(0), self.sequence_len, self.rnn_directions, self.hidden_size)
            gru_out = torch.sum(gru_out, dim=2)
        return gru_out, hidden.squeeze(0)


class AttentionDecoderCell(nn.Module):
    def __init__(self, input_feature_len, hidden_size, sequence_len):
        super().__init__()
        # attention - inputs - (decoder_inputs, prev_hidden)
        self.attention_linear = nn.Linear(hidden_size + input_feature_len, sequence_len)
        # attention_combine - inputs - (decoder_inputs, attention * encoder_outputs)
        self.decoder_rnn_cell = nn.GRUCell(
            input_size=hidden_size,
            hidden_size=hidden_size,
        )
        self.out = nn.Linear(hidden_size, input_feature_len)

    def forward(self, encoder_output, prev_hidden, y):
        attention_input = torch.cat([prev_hidden, y], dim=1)
        attention_weights = F.softmax(self.attention_linear(attention_input), dim=1).unsqueeze(1)
        attention_combine = torch.bmm(attention_weights, encoder_output).squeeze(1)
        rnn_hidden = self.decoder_rnn_cell(attention_combine, prev_hidden)
        output = self.out(rnn_hidden)
        return output, rnn_hidden


class Seq2Seq(nn.Module):
    def __init__(self, rnn_num_layers=1, input_feature_len=1, sequence_len=168, hidden_size=100, output_size=12, categorical=False):
        super().__init__()
        self.encoder = RNNEncoder(rnn_num_layers, input_feature_len, sequence_len, hidden_size)
        self.decoder_cell = AttentionDecoderCell(input_feature_len, hidden_size, sequence_len)
        self.output_size = output_size

        self.n_classes = 6 if categorical else 1
        self.out = nn.Linear(input_feature_len, self.n_classes)
        self.categorical = categorical

    def forward(self, xb):
        input_seq = xb
        encoder_output, encoder_hidden = self.encoder(input_seq)
        prev_hidden = encoder_hidden

        outputs = []
        y_prev = input_seq[:, -1, :]
        for i in range(self.output_size):
            rnn_output, prev_hidden = self.decoder_cell(encoder_output, prev_hidden, y_prev)
            y_prev = rnn_output

            outputs.append(rnn_output)

        outputs = torch.stack(outputs, 1)
        outputs = F.relu(outputs)
        outputs = self.out(outputs)

        outputs = outputs.permute(0, 2, 1) if self.categorical else F.relu(outputs)
        return outputs.squeeze(-1)