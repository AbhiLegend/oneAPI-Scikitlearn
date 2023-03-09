import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearnex import patch_sklearn

patch_sklearn()

# Load the iris dataset
iris = load_iris()
X = iris.data

# Use the Intel optimized version of KMeans
kmeans = KMeans(algorithm='full', n_clusters=3, init='k-means++', n_init=10,
                max_iter=300, tol=0.0001, precompute_distances='auto', verbose=0,
                random_state=None, copy_x=True, n_jobs=None)

# Fit the KMeans model to the iris data
kmeans.fit(X)

# Predict the clusters for the iris data
y_pred = kmeans.predict(X)

# Add the predicted cluster labels to the iris dataset
iris_df = pd.DataFrame(X, columns=iris.feature_names)
iris_df['Cluster'] = y_pred

# Print the first 5 rows of the dataset with cluster labels
print(iris_df.head())
