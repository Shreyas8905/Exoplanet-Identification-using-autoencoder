from train import train_autoencoder
from feature_extraction import feature_extraction
from analyze import analyze
from clustering import cluster

def main():
    train_autoencoder()
    latent_features, image_paths = feature_extraction()
    embedding, cluster_labels = cluster(latent_features)
    analyze(embedding, cluster_labels, image_paths)

if __name__ == "__main__":
    main()
