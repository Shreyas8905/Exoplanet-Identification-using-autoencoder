import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from data_preprocessing import ALMADataset, transform
from model import AutoEncoder
import glob

def train_autoencoder():
    device = torch.device("cpu")
    model = AutoEncoder().to(device)

    data_dir = "./data/*.fits"
    file_paths = glob.glob(data_dir)

    dataset = ALMADataset(file_paths, transforms=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = 1e-3, weight_decay = 1e-5)

    num_epochs = 50

    print("Starting training...")
    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for data, _ in dataloader:
            data = data.to(device)
            optimizer.zero_grad()

            latent, reconstruction = model(data)
            loss = criterion(reconstruction, data)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "autoencoder.pth")
    print("Model saved!")