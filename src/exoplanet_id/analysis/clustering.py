"""Dimensionality reduction (UMAP) and density-based clustering (HDBSCAN)."""

from __future__ import annotations

import logging
from typing import Tuple

import hdbscan
import numpy as np
import umap

from src.exoplanet_id.config import Config

logger = logging.getLogger(__name__)


def cluster_features(
    latent_features: np.ndarray,
    config: Config,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project latent features to 2-D with UMAP and cluster with HDBSCAN.

    Args:
        latent_features: Array of shape ``(N, latent_dim)``.
        config:          Pipeline configuration object.

    Returns:
        A tuple ``(embedding, cluster_labels)`` where ``embedding`` has
        shape ``(N, 2)`` and ``cluster_labels`` is an integer array of
        length *N* (noise points are labelled ``-1``).
    """
    logger.info("Reducing dimensionality with UMAP (n_neighbors=%d) …", config.umap_neighbors)
    reducer = umap.UMAP(
        n_neighbors=config.umap_neighbors,
        min_dist=config.umap_min_dist,
        n_components=config.umap_components,
        random_state=config.umap_random_state,
    )
    embedding = reducer.fit_transform(latent_features)

    logger.info("Clustering with HDBSCAN (min_cluster_size=%d) …", config.hdbscan_min_cluster_size)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=config.hdbscan_min_cluster_size,
        gen_min_span_tree=True,
    )
    cluster_labels = clusterer.fit_predict(embedding)

    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = int(np.sum(cluster_labels == -1))
    logger.info("Found %d clusters and %d noise points", n_clusters, n_noise)

    return embedding, cluster_labels
