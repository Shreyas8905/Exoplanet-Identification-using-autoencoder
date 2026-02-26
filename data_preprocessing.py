import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from astropy.io import fits
from sklearn.preprocessing import minmax_scale
import numpy as np
import matplotlib.pyplot as plt

class ALMADataset(Dataset):
    def __init__(self, file_paths, transforms=None):
        self.file_paths = file_paths
        self.transforms = transforms

    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        path = self.file_paths[idx]
        with fits.open(path) as image_data:
            data = image_data[0].data
            if data.ndim > 2:
                data = data.flatten()[:360000].reshape(600, 600)
            data = np.nan_to_num(data)
            data = np.log1p(data)
            data = data / (np.max(data) + 1e-8)
            data = data.astype(np.float32)
            data = np.expand_dims(data, axis=0)
        data = torch.tensor(data)
        if self.transforms:
            data = self.transforms(data)
        return data, path

transform = transforms.Compose([
    transforms.CenterCrop(512), 
    transforms.RandomRotation(180, fill=0) 
])   
