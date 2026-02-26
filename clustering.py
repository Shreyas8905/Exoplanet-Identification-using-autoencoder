import umap
import hdbscan

def cluster(latent_features):
    print("Mapping Latent space to 2d with UMAP...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    embedding = reducer.fit_transform(latent_features)
    print("Identifying clusters with HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True)
    cluster_labels = clusterer.fit_predict(embedding)

    return embedding, cluster_labels
