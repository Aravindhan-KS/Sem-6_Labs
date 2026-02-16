# K-Means Clustering on Iris Dataset
# Simple, full, error-free code

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def main():
    # Load the Iris dataset
    iris = load_iris()
    X = iris.data

    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(X_scaled)

    # Get cluster labels
    labels = kmeans.labels_

    # Print results
    print("Cluster labels for each data point:")
    print(labels)

    print("\nCluster centers:")
    print(kmeans.cluster_centers_)

if __name__ == "__main__":
    main()







