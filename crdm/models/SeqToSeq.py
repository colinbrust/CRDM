from torch import nn
import torch


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(Encoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # intialize the GRU to take the input dimetion of embbed, and output dimention of hidden and
        # set the number of gru layers
        self.gru = nn.GRU(self.input_dim, self.hidden_dim, num_layers=self.num_layers)

    def forward(self, seq):

        outputs, hidden = self.gru(seq)
        return outputs, hidden


class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, num_layers):
        super(Decoder, self).__init__()

        # set the encoder output dimension, embed dimension, hidden dimension, and number of layers
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(self.output_dim, self.hidden_dim, num_layers=self.num_layers)
        self.out = nn.Linear(self.hidden_dim, self.output_dim)
        self.relu = nn.ReLU()

    def forward(self, input, hidden):
        # reshape the input to (1, batch_size)
        embedded = self.relu(input)
        output, hidden = self.gru(embedded, hidden)
        prediction = self.relu(self.out(output[0]))

        return prediction, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        # initialize the encoder and decoder
        self.encoder = encoder
        self.decoder = decoder
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    def forward(self, source, target, teacher_forcing_ratio=0.5):

        input_length = source.size(0)  # get the input length (number of words in sentence)
        batch_size = target.shape[1]
        target_length = target.shape[0]
        vocab_size = self.decoder.output_dim

        # initialize a variable to hold the predicted outputs
        outputs = torch.zeros(target_length, batch_size, vocab_size).to(self.device)

        # encode every word in a sentence
        for i in range(input_length):
            encoder_output, encoder_hidden = self.encoder(source[i])

        # use the encoder’s hidden layer as the decoder hidden
        decoder_hidden = encoder_hidden.to(device)

        # add a token before the first predicted word
        decoder_input = torch.tensor([SOS_token], device=device)  # SOS

        # topk is used to get the top K value over a list
        # predict the output word from the current target word. If we enable the teaching force,  then the #next decoder input is the next word, else, use the decoder output highest value.

        for t in range(target_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            outputs[t] = decoder_output
            teacher_force = random.random() < teacher_forcing_ratio
            topv, topi = decoder_output.topk(1)
            input = (target[t] if teacher_force else topi)
            if (teacher_force == False and input.item() == EOS_token):
                break

        return outputs