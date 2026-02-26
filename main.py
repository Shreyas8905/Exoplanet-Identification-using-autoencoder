"""CLI entry point for the Exoplanet Identification pipeline.

Usage examples::

    python main.py run               # Full pipeline (train → analyse)
    python main.py train             # Train the autoencoder only
    python main.py extract           # Extract latent features
    python main.py cluster           # UMAP + HDBSCAN clustering
    python main.py visualize         # Generate cluster visualizations
    python main.py run --epochs 100  # Override default hyperparameters
"""

from __future__ import annotations

import argparse
import logging
import sys


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="exoplanet-id",
        description="Exoplanet Identification using Convolutional Autoencoder",
    )
    sub = parser.add_subparsers(dest="command", help="Pipeline stage to run")

    # ── Shared arguments ──────────────────────────────────────────────
    for name, help_text in [
        ("run",       "Run the full pipeline (train → extract → cluster → visualize)"),
        ("train",     "Train the autoencoder"),
        ("extract",   "Extract latent features from trained model"),
        ("cluster",   "Run UMAP + HDBSCAN clustering on extracted features"),
        ("visualize", "Generate cluster visualizations"),
    ]:
        sp = sub.add_parser(name, help=help_text)
        sp.add_argument("--epochs",     type=int,   default=None, help="Training epochs")
        sp.add_argument("--batch-size", type=int,   default=None, help="Batch size")
        sp.add_argument("--lr",         type=float, default=None, help="Learning rate")
        sp.add_argument("--latent-dim", type=int,   default=None, help="Latent vector size")

    return parser


def _config_from_args(args: argparse.Namespace):
    """Create a Config, overriding defaults with any CLI arguments."""
    from src.exoplanet_id.config import Config

    overrides = {}
    if getattr(args, "epochs", None) is not None:
        overrides["epochs"] = args.epochs
    if getattr(args, "batch_size", None) is not None:
        overrides["batch_size"] = args.batch_size
    if getattr(args, "lr", None) is not None:
        overrides["learning_rate"] = args.lr
    if getattr(args, "latent_dim", None) is not None:
        overrides["latent_dim"] = args.latent_dim
    return Config(**overrides)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = _build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    config = _config_from_args(args)

    if args.command == "run":
        from src.exoplanet_id.pipeline import run_pipeline
        run_pipeline(config)

    elif args.command == "train":
        from src.exoplanet_id.training.trainer import train_autoencoder
        train_autoencoder(config)

    elif args.command == "extract":
        from src.exoplanet_id.inference.feature_extractor import extract_features
        extract_features(config)

    elif args.command == "cluster":
        from src.exoplanet_id.inference.feature_extractor import extract_features
        from src.exoplanet_id.analysis.clustering import cluster_features
        latent_features, image_paths = extract_features(config)
        embedding, cluster_labels = cluster_features(latent_features, config)

    elif args.command == "visualize":
        from src.exoplanet_id.inference.feature_extractor import extract_features
        from src.exoplanet_id.analysis.clustering import cluster_features
        from src.exoplanet_id.analysis.visualize import visualize_clusters
        latent_features, image_paths = extract_features(config)
        embedding, cluster_labels = cluster_features(latent_features, config)
        visualize_clusters(embedding, cluster_labels, image_paths, config)


if __name__ == "__main__":
    main()
