import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# 1. Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# 2. Standardize the features (important for LDA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Apply LDA for dimensionality reduction to 2 components
# For a 3-class dataset, LDA can project to at most 2 discriminant dimensions (n_components = n_classes - 1)
lda = LinearDiscriminantAnalysis(n_components=2)
X_projected = lda.fit_transform(X_scaled, y)

# Explained variance ratio
print(f"Explained variance ratio (first two components): {lda.explained_variance_ratio_}")
print(f"Original shape: {X.shape}, Projected shape: {X_projected.shape}")

# 4. Visualize the projected data
plt.figure()
colors = ["navy", "turquoise", "darkorange"]
lw = 2

for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(
        X_projected[y == i, 0], X_projected[y == i, 1],
        color=color, alpha=0.8, lw=lw, label=target_name
    )
plt.legend(loc="best", shadow=False, scatterpoints=1)
plt.title("LDA of IRIS dataset projection onto the first two components")
plt.xlabel("Linear Discriminant 1")
plt.ylabel("Linear Discriminant 2")
plt.show()
