"""Training loop for the convolutional autoencoder."""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.optim as optim

from src.exoplanet_id.config import Config
from src.exoplanet_id.data.dataset import get_dataloader
from src.exoplanet_id.models.autoencoder import AutoEncoder

logger = logging.getLogger(__name__)


def train_autoencoder(config: Config) -> None:
    """Train the autoencoder on FITS images and save the weights.

    The training loop uses MSE reconstruction loss with Adam optimiser.
    Model weights are saved to ``config.model_path`` after training.

    Args:
        config: Pipeline configuration object.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    model = AutoEncoder(latent_dim=config.latent_dim).to(device)
    dataloader = get_dataloader(config, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    logger.info(
        "Starting training — %d epochs, batch_size=%d, lr=%.1e",
        config.epochs, config.batch_size, config.learning_rate,
    )
    model.train()

    for epoch in range(config.epochs):
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
        logger.info("Epoch [%d/%d]  Loss: %.6f", epoch + 1, config.epochs, avg_loss)

    config.model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), config.model_path)
    logger.info("Model saved to %s", config.model_path)
