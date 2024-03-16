class HartleyEncoderRNN(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
        self.dht = DHT(m)
        self.idht = IDHT(m)
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.LSTM(64, 32),
        )
        self.decoder = nn.Sequential(
            nn.LSTM(32, 64),
            nn.ConvTranspose1d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.dht(x)
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.idht(x)
        return x