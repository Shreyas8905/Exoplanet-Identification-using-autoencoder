import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

def analyze(embedding, cluster_labels, image_paths):
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        embedding[:, 0], 
        embedding[:, 1], 
        c=cluster_labels, 
        cmap='Spectral', 
        s=60, 
        alpha=0.8, 
        edgecolors='w', 
        linewidths=0.5
    )
    plt.colorbar(scatter, label="HDBSCAN Cluster Label")
    plt.title("Unsupervised Clustering of ALMA Data", fontsize=16, fontweight='bold')
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.show()

    unique_clusters = np.unique(cluster_labels)
    num_clusters = len(unique_clusters)

    fig, axes = plt.subplots(num_clusters, 5, figsize=(15, 3 * num_clusters))
    fig.suptitle("Sample Observations per cluster", fontsize=18, y=1.02)

    for i, cluster_id in enumerate(unique_clusters):
        indices = np.where(cluster_labels == cluster_id)[0]
        sample_indices = indices[:5]

        for j in range(5):
            if num_clusters == 1:
                ax = axes[j]
            else:
                ax = axes[i, j]

            if j < len(sample_indices):
                idx = sample_indices[j]

                with fits.open(image_paths[idx]) as hdul:
                    img_data = hdul[0].data
                    if img_data.ndim > 2:
                        img_data = img_data.flatten()[:360000].reshape(600, 600)

                ax.imshow(img_data, cmap='inferno', origin='lower')
                ax.set_title(f"Cluster {cluster_id}")
                ax.axis('off')
            else:
                ax.axis('off')
    plt.tight_layout()
    plt.show()