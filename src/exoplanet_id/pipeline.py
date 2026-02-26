"""End-to-end pipeline orchestration.

Provides a single ``run_pipeline`` function that chains every stage:
train → extract → cluster → visualize.
"""

from __future__ import annotations

import logging

from src.exoplanet_id.analysis.clustering import cluster_features
from src.exoplanet_id.analysis.visualize import visualize_clusters
from src.exoplanet_id.config import Config
from src.exoplanet_id.inference.feature_extractor import extract_features
from src.exoplanet_id.training.trainer import train_autoencoder

logger = logging.getLogger(__name__)


def run_pipeline(config: Config) -> None:
    """Execute the full exoplanet identification pipeline.

    Stages:
        1. **Train** the autoencoder on FITS images.
        2. **Extract** latent features from the trained model.
        3. **Cluster** the latent features with UMAP + HDBSCAN.
        4. **Visualize** the clustering results and sample images.

    Args:
        config: Pipeline configuration object.
    """
    logger.info("=" * 60)
    logger.info("EXOPLANET IDENTIFICATION PIPELINE")
    logger.info("=" * 60)

    # Stage 1 — Training
    logger.info("Stage 1/4: Training autoencoder …")
    train_autoencoder(config)

    # Stage 2 — Feature extraction
    logger.info("Stage 2/4: Extracting latent features …")
    latent_features, image_paths = extract_features(config)

    # Stage 3 — Clustering
    logger.info("Stage 3/4: Clustering …")
    embedding, cluster_labels = cluster_features(latent_features, config)

    # Stage 4 — Visualization
    logger.info("Stage 4/4: Generating visualizations …")
    visualize_clusters(embedding, cluster_labels, image_paths, config)

    logger.info("Pipeline complete. Outputs saved to %s", config.output_dir)
