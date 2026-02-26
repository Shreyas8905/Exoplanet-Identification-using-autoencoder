"""Extract latent-space features from a trained autoencoder."""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.exoplanet_id.config import Config
from src.exoplanet_id.data.dataset import get_dataloader
from src.exoplanet_id.models.autoencoder import AutoEncoder

logger = logging.getLogger(__name__)


def extract_features(config: Config) -> Tuple[np.ndarray, List[str]]:
    """Load a trained model and extract latent vectors for every FITS image.

    Args:
        config: Pipeline configuration object.

    Returns:
        A tuple ``(latent_features, image_paths)`` where
        ``latent_features`` is an ``(N, latent_dim)`` numpy array and
        ``image_paths`` is a list of source file paths.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    model = AutoEncoder(latent_dim=config.latent_dim).to(device)
    model.load_state_dict(
        torch.load(config.model_path, map_location=device, weights_only=True)
    )
    model.eval()
    logger.info("Loaded model weights from %s", config.model_path)

    dataloader: DataLoader = get_dataloader(config, shuffle=False)

    latent_features: list = []
    image_paths: List[str] = []

    logger.info("Extracting latent features …")
    with torch.no_grad():
        for data, paths in dataloader:
            data = data.to(device)
            latent, _ = model(data)
            latent_features.extend(latent.cpu().numpy())
            image_paths.extend(paths)

    latent_features_array = np.array(latent_features)
    logger.info("Extracted features: shape %s", latent_features_array.shape)
    return latent_features_array, image_paths
