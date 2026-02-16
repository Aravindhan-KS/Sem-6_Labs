import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import pandas as pd
import seaborn as sns

# 1. Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# 2. Apply Principal Component Analysis (PCA) for a 2D projection
# We specify n_components=2 to project the data onto the first two principal components
pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X) # X_r contains the projected data

# 3. Create a DataFrame for easier plotting with Seaborn
pca_df = pd.DataFrame(data=X_r, columns=['principal component 1', 'principal component 2'])
pca_df['species'] = [target_names[i] for i in y]

# 4. Plot the projected data
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=pca_df,
    x='principal component 1',
    y='principal component 2',
    hue='species',
    palette='viridis',
    s=100, # size of points
    alpha=0.8
)
plt.title('PCA 2D Projection of Iris Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()


