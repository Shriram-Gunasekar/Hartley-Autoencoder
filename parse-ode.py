class HartleyEncoderODESolver(nn.Module):
    def __init__(self, m, n):
        super().__init__()
        self.m = m
        self.n = n
        self.dht = DHT(m)
        self.idht = IDHT(m)
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            ODESolverBlock(32, 64, n),
        )
        self.decoder = nn.Sequential(
            ODESolverBlock(64, 32, n),
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

class ODESolverBlock(nn.Module):
    """
    To Be Input with a Neural ODE Solver (NIPS 2019) or such
    """