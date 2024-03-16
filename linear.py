class Autoencoder(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
        self.dht = DHT(m)
        self.idht = IDHT(m)
        self.encoder = nn.Sequential(
            nn.Linear(m, m//2),
            nn.ReLU(),
            nn.Linear(m//2, m//4),
            nn.ReLU(),
            nn.Linear(m//4, m//8),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(m//8, m//4),
            nn.ReLU(),
            nn.Linear(m//4, m//2),
            nn.ReLU(),
            nn.Linear(m//2, m),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.dht(x)
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.idht(x)
        return x