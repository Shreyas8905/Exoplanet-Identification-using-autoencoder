# Exoplanet Identification using Convolutional Autoencoder

A fully **unsupervised** deep-learning pipeline for detecting signatures of hidden exoplanets in synthetic [ALMA](https://www.almaobservatory.org/) continuum observations of protoplanetary disks. Embedded planets induce localized deviations from Keplerian velocity fields — manifesting as gaps, rings, and spiral wakes in the dust continuum. This pipeline extracts those subtle morphological anomalies without any ground-truth labels by compressing each observation into a dense latent "fingerprint" and clustering the resulting manifold.

> Because the task is strictly unsupervised, supervised architectures (EfficientNetV2, RegNet, etc.) used in prior kinematic-detection work are ruled out. The model is also explicitly designed to be **invariant to disk inclination and position angle**, preventing trivial clustering by viewing geometry.

---

## Table of Contents

- [Pipeline Architecture](#-pipeline-architecture)
- [Data Ingestion & Preprocessing](#-data-ingestion--preprocessing)
- [Geometric Invariance Strategy](#-geometric-invariance-strategy)
- [Autoencoder Architecture](#-autoencoder-architecture)
- [Dimensionality Reduction — UMAP](#-dimensionality-reduction--umap)
- [Density-Based Clustering — HDBSCAN](#-density-based-clustering--hdbscan)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Usage](#-usage)
- [Configuration](#%EF%B8%8F-configuration)
- [Outputs](#-outputs)
- [Technologies Used](#-technologies-used)

---

## Pipeline Architecture

The pipeline is structured in four sequential stages, each addressing a specific challenge inherent to radio-interferometric protoplanetary disk data:

```
 FITS Cubes ──► Preprocessing ──► Autoencoder ──► UMAP ──► HDBSCAN ──► Cluster Map
   (4D)          (Stokes I,        (512-D          (2-D       (density-    (discrete
                  log-scale,        latent          topo-      based        categories
                  crop, rotate)     vector)         logy)      labels)      + noise)
```

1. **Data Preprocessing** — Isolate Stokes I continuum, compress dynamic range, enforce geometric invariance.
2. **Feature Extraction** — Train a convolutional autoencoder (CAE) to learn a 512-D morphological fingerprint per observation.
3. **Dimensionality Reduction** — Project the high-dimensional latent manifold to 2-D with UMAP, preserving both local and global topology.
4. **Clustering** — Apply HDBSCAN to discover density-connected groupings of structurally similar disks without specifying the number of clusters *a priori*.

---

## Data Ingestion & Preprocessing

The raw data consists of synthetic `.fits` observation cubes — 4-layer data cubes at 600 × 600 spatial resolution, where the four layers correspond to the **Stokes parameters** (I, Q, U, V) as defined in each FITS header.

### Stokes Parameter Isolation

The pipeline extracts only the **Stokes I** (total intensity / continuum flux) layer at index 0. This isolates the thermal dust emission that encodes the physical morphology of the disk — gaps carved by planetary torques, ring structures at pressure maxima, and spiral density wakes. The polarization states (Q, U, V) carry information about magnetic field alignment and grain properties but are irrelevant for morphological clustering and are therefore discarded.

### Logarithmic Dynamic-Range Compression

Protoplanetary disks exhibit an extreme dynamic range: the central stellar emission is orders of magnitude brighter than the faint spiral wakes induced by embedded planetary bodies at large orbital radii. Applying standard min-max normalization produces a **brightness bias** — the autoencoder dedicates its entire learning capacity to reconstructing the overwhelmingly bright central peak while ignoring the scientifically critical faint outer structures.

To counter this, `np.log1p(x)` (i.e., log(1 + x)) is applied before normalization:

- **Compresses** the bright central flux into a narrow dynamic range.
- **Boosts** the signal-to-noise ratio of faint planetary gaps and ring features.
- **Forces the autoencoder** to weight subtle planetary structures equally with macroscopic disk geometry.

---

## Geometric Invariance Strategy

A naïve autoencoder will trivially separate face-on (circular) disks from highly inclined, edge-on (elliptical) disks because the raw pixel variance between these viewing geometries is massive — far exceeding the variance from planetary signatures. Two transforms address this:

### CenterCrop(512)

A 600 × 600 tensor passed through five stride-2 convolutional layers creates fractional spatial dimensions (e.g., 600 → 300 → 150 → 75 → 37.5), causing catastrophic tensor-shape mismatches during the symmetric upsampling phase. Cropping to **512 × 512** (a power of two) ensures clean integer division at every downsampling step (512 → 256 → 128 → 64 → 32 → 16) while removing only empty deep-space pixels at the image periphery.

### RandomRotation(180°, fill=0)

Every image is subjected to a **continuous random rotation** sampled uniformly from [−180°, +180°] at load time. This explicitly forces the autoencoder to map identical gap structures to the **same latent-space coordinate** regardless of their azimuthal orientation or disk position angle. The `fill=0` parameter ensures that the empty triangular corners created by the affine rotation matrix are padded with true background (zero flux) rather than introducing geometric edge artifacts that the encoder could latch onto.

---

## Autoencoder Architecture

A deep **Convolutional Autoencoder (CAE)** learns compressed structural representations by training against reconstruction loss (MSE). The strict information bottleneck forces the network to discard pixel-level noise and retain only the morphologically significant features — precisely the gap widths, ring spacings, and wake geometries induced by embedded planets.

```
 Input (1 × 512 × 512)
   │
   ▼  ENCODER — 5 × [Conv2d(3×3, stride=2, pad=1) → BatchNorm2d → ReLU]
   │  Channel expansion:  1 → 16 → 32 → 64 → 128 → 256
   │  Spatial reduction:  512 → 256 → 128 → 64 → 32 → 16
   │
   ▼  Flatten (256 × 16 × 16 = 65 536 features)
   │
   ▼  Linear → 512-D LATENT VECTOR  (the morphological "fingerprint")
   │
   ▼  Linear → 65 536 → Reshape (256 × 16 × 16)
   │
   ▼  DECODER — 5 × [ConvTranspose2d(3×3, stride=2, pad=1, out_pad=1) → BatchNorm → ReLU]
   │  Channel contraction:  256 → 128 → 64 → 32 → 16 → 1
   │  Spatial expansion:    16 → 32 → 64 → 128 → 256 → 512
   │
   ▼  Sigmoid → Reconstructed Image (1 × 512 × 512), bounded [0, 1]
```

- **BatchNorm2d** after every convolution stabilizes gradient flow and accelerates convergence.
- **Adam optimizer** (lr = 10⁻³, weight decay = 10⁻⁵) provides adaptive per-parameter learning rates with mild L2 regularization.
- **MSE loss** drives the bottleneck to faithfully encode the true physical morphology of each synthetic observation. The 512-D latent vector becomes a continuous, rotation-invariant structural descriptor.

---

## Dimensionality Reduction — UMAP

The 512-D latent space encodes planetary morphology effectively, but suffers from the **curse of dimensionality**: Euclidean distance metrics lose discriminative power in high-dimensional spaces, causing downstream clustering algorithms to fail.

**Uniform Manifold Approximation and Projection (UMAP)** resolves this by projecting the 512-D latent manifold onto a 2-D topology that preserves meaningful structural relationships.

| Hyperparameter  | Value | Rationale                                                                                      |
| --------------- | ----- | ---------------------------------------------------------------------------------------------- |
| `n_neighbors`   | 15    | Balances local vs. global structure; tighter values (e.g., 5) collapse smooth disks into a single unreadable singularity |
| `min_dist`      | 0.1   | Relaxes the projection to preserve both local relationships (similar gap structures) and global separation (smooth vs. multi-ring) |
| `n_components`  | 2     | Produces a human-interpretable 2-D scatter plot                                                |
| `random_state`  | 42    | Ensures reproducibility of the non-deterministic optimization                                  |

---

## Density-Based Clustering — HDBSCAN

The final stage assigns discrete structural categories to the continuous 2-D UMAP topology. Algorithms like K-Means are **inappropriate** for this task because they:

1. Require a human prior (guessing *K*, the number of planetary configurations).
2. Enforce spherical Voronoi partitions that cannot capture the organic, irregularly shaped groupings typical of astrophysical data.

**Hierarchical Density-Based Spatial Clustering of Applications with Noise (HDBSCAN)** overcomes both limitations:

- **Density over Geometry** — HDBSCAN scans the 2-D UMAP space for continuous regions of high data density, allowing it to adapt to arbitrarily shaped clusters. It autonomously determines the number of distinct planetary configurations present in the dataset.
- **Anomaly Filtering** — Observations with highly chaotic, corrupted, or unique morphologies that do not align with any dominant manifold are formally isolated and labelled as **noise (Cluster −1)**. This prevents anomalous data from contaminating the clean clusters of scientifically viable planetary systems.

| Hyperparameter       | Value | Rationale                                                  |
| -------------------- | ----- | ---------------------------------------------------------- |
| `min_cluster_size`   | 5     | Minimum membership for a density-connected group           |
| `gen_min_span_tree`  | True  | Enables visualization of the cluster hierarchy             |

---

## Project Structure

```
Exoplanet-Identification-using-autoencoder/
│
├── main.py                              # CLI entry point (argparse)
├── requirements.txt                     # Python dependencies
├── README.md
├── .gitignore
│
├── src/
│   └── exoplanet_id/                    # Main Python package
│       ├── __init__.py
│       ├── config.py                    # Centralized configuration (dataclass)
│       ├── pipeline.py                  # End-to-end pipeline orchestration
│       │
│       ├── data/
│       │   ├── __init__.py
│       │   └── dataset.py              # ALMADataset (PyTorch) + DataLoader factory
│       │
│       ├── models/
│       │   ├── __init__.py
│       │   └── autoencoder.py          # Convolutional Autoencoder architecture
│       │
│       ├── training/
│       │   ├── __init__.py
│       │   └── trainer.py              # Training loop with logging
│       │
│       ├── inference/
│       │   ├── __init__.py
│       │   └── feature_extractor.py    # Latent feature extraction
│       │
│       └── analysis/
│           ├── __init__.py
│           ├── clustering.py           # UMAP + HDBSCAN clustering
│           └── visualize.py            # Matplotlib visualizations
│
├── data/                                # (gitignored) FITS observation files
│   └── *.fits
│
└── outputs/                             # (auto-created) Model weights & plots
    ├── autoencoder.pth
    ├── cluster_scatter.png
    └── cluster_samples.png
```

---

## Getting Started

### Prerequisites

- **Python** ≥ 3.9
- **pip** or **conda**
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) (optional — enables GPU-accelerated training)

### Installation

```bash
# Clone the repository
git clone https://github.com/Shreyas8905/Exoplanet-Identification-using-autoencoder.git
cd Exoplanet-Identification-using-autoencoder

# Create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Download the Data

The synthetic FITS observation cubes are hosted separately. Download and place them in the `data/` directory:

**[Download FITS Data (Google Drive)](https://drive.google.com/drive/folders/1VkS3RHkAjiKjJ6DnZmEKZ_nUv4w6pz7P)**

---

## Usage

```bash
# Full pipeline: train → extract → cluster → visualize
python main.py run

# Individual stages
python main.py train             # Train the autoencoder
python main.py extract           # Extract 512-D latent features
python main.py cluster           # UMAP projection + HDBSCAN clustering
python main.py visualize         # Generate scatter plot & sample grids

# Override hyperparameters
python main.py run --epochs 100 --batch-size 32 --lr 0.0005
python main.py train --latent-dim 256 --epochs 80

# Help
python main.py --help
```

| Command       | Requires                             |
| ------------- | ------------------------------------ |
| `train`       | FITS files in `data/`                |
| `extract`     | Trained model in `outputs/`          |
| `cluster`     | Runs `extract` automatically         |
| `visualize`   | Runs `extract` + `cluster` automatically |
| `run`         | FITS files in `data/` (runs everything) |

---

## Configuration

All defaults are defined in [`src/exoplanet_id/config.py`](src/exoplanet_id/config.py):

| Parameter                 | Default    | Description                                              |
| ------------------------- | ---------- | -------------------------------------------------------- |
| `latent_dim`              | `512`      | Autoencoder latent vector dimensionality                  |
| `image_size`              | `512`      | CenterCrop target (must be power of 2)                    |
| `epochs`                  | `50`       | Training epochs                                           |
| `batch_size`              | `16`       | Mini-batch size                                           |
| `learning_rate`           | `1e-3`     | Adam optimizer learning rate                              |
| `weight_decay`            | `1e-5`     | L2 regularization strength                                |
| `umap_neighbors`          | `15`       | UMAP `n_neighbors` — local vs. global balance             |
| `umap_min_dist`           | `0.1`      | UMAP `min_dist` — projection relaxation                   |
| `hdbscan_min_cluster_size`| `5`        | Minimum density-connected group membership                |

---

## Outputs

| File                    | Description                                                                             |
| ----------------------- | --------------------------------------------------------------------------------------- |
| `autoencoder.pth`       | Trained encoder–decoder weights                                                          |
| `cluster_scatter.png`   | 2-D UMAP projection coloured by HDBSCAN cluster label (noise points labelled −1)         |
| `cluster_samples.png`   | Grid of representative FITS observations from each discovered morphological cluster      |

---

## Technologies Used

| Technology     | Role                                                                 |
| -------------- | -------------------------------------------------------------------- |
| **PyTorch**    | Convolutional autoencoder training and inference                      |
| **Astropy**    | FITS I/O and Stokes parameter header parsing                          |
| **NumPy**      | Log-scaling, normalization, and array manipulation                    |
| **Matplotlib** | Cluster scatter plots and observation sample grids                    |
| **UMAP**       | Non-linear manifold projection (512-D → 2-D)                         |
| **HDBSCAN**    | Hierarchical density-based clustering with noise isolation            |
| **scikit-learn** | Preprocessing utilities                                            |

---

## License

This project is open-source. Feel free to use, modify, and distribute.
