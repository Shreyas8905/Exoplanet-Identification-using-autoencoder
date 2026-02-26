import torch
import glob
import numpy as np
from torch.utils.data import DataLoader
from data_preprocessing import ALMADataset, transform
from model import AutoEncoder

def feature_extraction():
    print("Extracting structural fingerprints...")
    device = torch.device("cpu")
    model = AutoEncoder(latent_dim=512).to(device)
    model.load_state_dict(torch.load('autoencoder.pth'))
    device = torch.device("cpu")
    data_dir = "./data/*.fits"
    file_paths = glob.glob(data_dir)
    dataset = ALMADataset(file_paths, transforms=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    model.eval()
    latent_features = []
    image_paths = []

    with torch.no_grad():
        for data, paths in dataloader:
            data = data.to(device)
            latent, _ = model(data)
            latent_features.extend(latent.cpu().numpy())
            image_paths.extend(paths)
    latent_features = np.array(latent_features)

    return latent_features, image_paths