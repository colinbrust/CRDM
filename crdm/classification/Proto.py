from torch import nn
import torch


class Proto(nn.Module):

    def __init__(self, in_channels, crop_size, future_steps):
        super().__init__()

        self.in_channels = in_channels
        self.future_steps = future_steps
        self.crop_size = crop_size

        self.encoder_cnn = nn.Sequential(
            nn.Conv3d(in_channels=self.in_channels,
                      out_channels=16,
                      kernel_size=(3, 3, 3),
                      padding=(1, 1, 1)),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=16,
                      out_channels=8,
                      kernel_size=(3, 3, 3),
                      padding=(1, 1, 1)),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=8,
                      out_channels=1,
                      kernel_size=(3, 3, 3),
                      padding=(1, 1, 1)),
            nn.BatchNorm3d(1),
            nn.ReLU(inplace=True)
        )

        self.encoder_lstm = nn.LSTM(input_size=self.crop_size**2, hidden_size=self.crop_size**2,
                                    batch_first=True, num_layers=2, dropout=0.25)
        self.decoder_lstm = nn.LSTM(input_size=self.crop_size**2, hidden_size=self.crop_size**2,
                                    batch_first=True, num_layers=2, dropout=0.25)

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        x = self.encoder_cnn(x)
        x = x.squeeze().flatten(-2)
        x, _ = self.encoder_lstm(x)
        x = x[:, -1, :].unsqueeze(1)
        outputs = []
        for t in range(self.future_steps):
            if t == 0:
                x, state = self.decoder_lstm(x)
            else:
                x, state = self.decoder_lstm(x, state)
            outputs += [x]

        outputs = torch.stack(outputs, 1)
        outputs = outputs.squeeze()
        outputs = outputs.unflatten(-1, (self.crop_size, self.crop_size))

        return outputs
