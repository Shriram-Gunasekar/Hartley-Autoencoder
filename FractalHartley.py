"""
FractalBlock layers to be generated from FractalNets ICLR 2017
"""

class HartleyFractalNet(nn.Module):
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
            FractalBlock(32, 64),
            FractalBlock(64, 128),
        )
        self.decoder = nn.Sequential(
            FractalBlock(128, 64),
            FractalBlock(64, 32),
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
