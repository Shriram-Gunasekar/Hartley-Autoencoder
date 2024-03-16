class TabularDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# Load the tabular dataset
x = pd.read_csv()
y = np.load('y.npy')

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

# Scale the data
# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# Create the autoencoder
autoencoder = Autoencoder(8)

# Define the loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.01)

# Train the autoencoder
epochs = 100
for epoch in range(epochs):
    # Train the autoencoder on the training set
    autoencoder.train()
    train_loader = DataLoader(TabularDataset(x_train, x_train), batch_size=128, shuffle=True)
    for x, y in train_loader:
        optimizer.zero_grad()
        output = autoencoder(x)
        loss = loss_fn(output, x)
        loss.backward()
        optimizer.step()

    # Evaluate the autoencoder on the testing set
autoencoder.eval()
test_loader = DataLoader(TabularDataset(x_test, x_test), batch_size=128, shuffle=False)
with torch.no_grad():
    y_pred = autoencoder(x_test)

# Save the trained autoencoder
torch.save(autoencoder.state_dict(), 'autoencoder.pt')

# Load the trained autoencoder
autoencoder.load_state_dict(torch.load('autoencoder.pt'))

# Generate new data using the autoencoder
x_new = autoencoder(x_test)

# Print the generated data
print(x_new)