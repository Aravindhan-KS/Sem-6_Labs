# K-Means Clustering on Iris Dataset
# Full working code with no errors

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def main():
    # Load Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply K-Means clustering
    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )
    kmeans.fit(X_scaled)

    # Get cluster labels
    labels = kmeans.labels_

    # Reduce dimensions for visualization using PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Plot the clusters
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=labels,
        cmap='viridis',
        s=50
    )

    plt.title("K-Means Clustering on Iris Dataset")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.colorbar(scatter, label="Cluster Label")
    plt.grid(True)
    plt.show()

    # Print cluster centers (in scaled feature space)
    print("Cluster Centers (scaled features):")
    print(kmeans.cluster_centers_)

if __name__ == "__main__":
    main()









