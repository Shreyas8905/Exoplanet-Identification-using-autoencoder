"""Visualization helpers for cluster analysis results."""

from __future__ import annotations

import logging
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from src.exoplanet_id.config import Config

logger = logging.getLogger(__name__)


def visualize_clusters(
    embedding: np.ndarray,
    cluster_labels: np.ndarray,
    image_paths: List[str],
    config: Config,
) -> None:
    """Generate and save cluster scatter-plot and per-cluster sample grids.

    Two figures are produced:

    1. **UMAP scatter plot** — all observations coloured by cluster label.
    2. **Sample grid** — up to 5 representative FITS images per cluster.

    Both are saved as PNG files to ``config.output_dir`` *and* displayed
    with ``plt.show()``.

    Args:
        embedding:      2-D UMAP embedding of shape ``(N, 2)``.
        cluster_labels: Integer cluster label for each observation.
        image_paths:    Corresponding FITS file paths.
        config:         Pipeline configuration object.
    """
    # ── 1. UMAP Scatter Plot ──────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    scatter = ax1.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=cluster_labels,
        cmap="Spectral",
        s=60,
        alpha=0.8,
        edgecolors="w",
        linewidths=0.5,
    )
    fig1.colorbar(scatter, ax=ax1, label="HDBSCAN Cluster Label")
    ax1.set_title("Unsupervised Clustering of ALMA Data", fontsize=16, fontweight="bold")
    ax1.set_xlabel("UMAP Dimension 1")
    ax1.set_ylabel("UMAP Dimension 2")
    ax1.grid(True, linestyle="--", alpha=0.3)

    scatter_path = config.output_dir / "cluster_scatter.png"
    fig1.savefig(scatter_path, dpi=150, bbox_inches="tight")
    logger.info("Saved scatter plot to %s", scatter_path)
    plt.show()

    # ── 2. Per-Cluster Sample Grid ────────────────────────────────────
    unique_clusters = np.unique(cluster_labels)
    num_clusters = len(unique_clusters)

    fig2, axes = plt.subplots(num_clusters, 5, figsize=(15, 3 * num_clusters))
    fig2.suptitle("Sample Observations per Cluster", fontsize=18, y=1.02)

    for i, cluster_id in enumerate(unique_clusters):
        indices = np.where(cluster_labels == cluster_id)[0]
        sample_indices = indices[:5]

        for j in range(5):
            ax = axes[j] if num_clusters == 1 else axes[i, j]

            if j < len(sample_indices):
                idx = sample_indices[j]
                with fits.open(image_paths[idx]) as hdul:
                    img_data = hdul[0].data
                    if img_data.ndim > 2:
                        img_data = img_data.flatten()[:360_000].reshape(600, 600)

                ax.imshow(img_data, cmap="inferno", origin="lower")
                ax.set_title(f"Cluster {cluster_id}")
            ax.axis("off")

    plt.tight_layout()
    grid_path = config.output_dir / "cluster_samples.png"
    fig2.savefig(grid_path, dpi=150, bbox_inches="tight")
    logger.info("Saved sample grid to %s", grid_path)
    plt.show()
