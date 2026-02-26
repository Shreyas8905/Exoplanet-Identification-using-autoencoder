"""Convolutional Autoencoder for ALMA observation images.

The encoder maps a single-channel (1, 512, 512) image to a compact
latent vector of configurable dimensionality.  The decoder reconstructs
the original image from that latent representation.  Training uses
reconstruction loss (MSE); the latent vectors are later used for
unsupervised clustering to identify exoplanet signatures.
"""

from __future__ import annotations

import torch.nn as nn


class AutoEncoder(nn.Module):
    """Symmetric convolutional autoencoder.

    Architecture (encoder):
        5 × [Conv2d → BatchNorm2d → ReLU]  (stride-2 downsampling)
        Flatten → Linear → latent vector

    Architecture (decoder):
        Linear → Reshape
        5 × [ConvTranspose2d → BatchNorm2d → ReLU]  (stride-2 upsampling)
        Final activation: Sigmoid (output in [0, 1])

    Args:
        latent_dim: Size of the latent feature vector.  Default ``512``.
    """

    def __init__(self, latent_dim: int = 512) -> None:
        super().__init__()

        # ── Encoder (convolutional) ───────────────────────────────────
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )

        self.flatten = nn.Flatten()
        self.encoder_fc = nn.Linear(256 * 16 * 16, latent_dim)

        # ── Decoder (transposed convolutional) ────────────────────────
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 256 * 16 * 16),
            nn.ReLU(True),
        )

        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),

            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """Forward pass returning both the latent vector and reconstruction.

        Args:
            x: Input tensor of shape ``(B, 1, 512, 512)``.

        Returns:
            Tuple of ``(latent, reconstruction)`` where ``latent`` has shape
            ``(B, latent_dim)`` and ``reconstruction`` has the same shape as
            the input.
        """
        x = self.encoder(x)
        x = self.flatten(x)
        latent = self.encoder_fc(x)

        x = self.decoder_fc(latent)
        x = x.view(-1, 256, 16, 16)
        reconstruction = self.decoder_cnn(x)

        return latent, reconstruction
