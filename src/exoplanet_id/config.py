"""Centralized configuration for the Exoplanet Identification pipeline."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    """All configuration values for the pipeline.

    Attributes:
        project_root:       Absolute path to the project root directory.
        data_dir:           Directory containing raw FITS files.
        model_path:         Path where the trained model weights are saved/loaded.
        output_dir:         Directory for saving plots and analysis outputs.

        latent_dim:         Dimensionality of the autoencoder's latent space.
        image_size:         Spatial size (H=W) images are cropped to before the model.

        epochs:             Number of training epochs.
        batch_size:         Mini-batch size for the DataLoader.
        learning_rate:      Adam optimizer learning rate.
        weight_decay:       Adam optimizer weight decay (L2 regularization).

        umap_neighbors:     UMAP ``n_neighbors`` parameter.
        umap_min_dist:      UMAP ``min_dist`` parameter.
        umap_components:    Number of UMAP output dimensions.
        umap_random_state:  Random seed for UMAP reproducibility.

        hdbscan_min_cluster_size:  HDBSCAN ``min_cluster_size`` parameter.
    """

    # ── Paths ──────────────────────────────────────────────────────────
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2])
    data_dir: Path = field(init=False)
    model_path: Path = field(init=False)
    output_dir: Path = field(init=False)

    # ── Model ──────────────────────────────────────────────────────────
    latent_dim: int = 512
    image_size: int = 512

    # ── Training ───────────────────────────────────────────────────────
    epochs: int = 50
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5

    # ── Clustering — UMAP ─────────────────────────────────────────────
    umap_neighbors: int = 15
    umap_min_dist: float = 0.1
    umap_components: int = 2
    umap_random_state: int = 42

    # ── Clustering — HDBSCAN ──────────────────────────────────────────
    hdbscan_min_cluster_size: int = 5

    def __post_init__(self) -> None:
        self.data_dir = self.project_root / "data"
        self.model_path = self.project_root / "outputs" / "autoencoder.pth"
        self.output_dir = self.project_root / "outputs"
        self.output_dir.mkdir(parents=True, exist_ok=True)
