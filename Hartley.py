class DHT(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        X = np.fft.fftn(x)
        X = np.real(X)-np.imag(X)
        return torch.tensor(X)

class IDHT(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        n = len(x)
        X = np.fft.fftn(x)
        X = np.real(X)-np.imag(X)
        x = 1.0/n*X
        return torch.tensor(x)