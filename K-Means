import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import os

# Set environment variable to avoid memory leak on Windows with MKL
os.environ['OMP_NUM_THREADS'] = '1'

# Step 1: Data Preparation
# Load the data
file_path = r"C:\Users\saipr\Downloads\customers_data.csv"
data = pd.read_csv(file_path)

# Verify the column names
print("Columns in the dataset:", data.columns)

# Define numeric and categorical features
numeric_features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
categorical_features = ['Gender']

# Drop the 'CustomerID' as it's not needed for clustering
X = data.drop(columns=['CustomerID'])

# Create a column transformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Apply the transformations
X_processed = preprocessor.fit_transform(X)

# Step 2: Determine the optimal number of clusters using the Elbow Method
inertia = []
silhouette_scores = []

# Test different numbers of clusters
K = range(2, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_processed)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_processed, kmeans.labels_))

# Plot the Elbow Method
plt.figure(figsize=(14, 6))

# Elbow Method plot
plt.subplot(1, 2, 1)
plt.plot(K, inertia, 'bo-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.grid(True)

# Silhouette Scores plot
plt.subplot(1, 2, 2)
plt.plot(K, silhouette_scores, 'bo-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Scores for Optimal k')
plt.grid(True)

plt.tight_layout()
plt.show()

# Print inertia and silhouette scores
print("Inertia values for different k:", inertia)
print("Silhouette scores for different k:", silhouette_scores)

# Step 3: Apply K-means with the optimal number of clusters (assuming optimal k is 4 for this example)
optimal_k = 4  # You can determine this from the Elbow Method and Silhouette Score plots
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_processed)

# Add the cluster labels to the original data
data['Cluster'] = clusters

# Step 4: Evaluation and Visualization
# Visualize the clusters in 2D space (assuming we use PCA for dimensionality reduction)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_processed)

plt.figure(figsize=(8, 6))
for cluster in range(optimal_k):
    plt.scatter(X_pca[clusters == cluster, 0], X_pca[clusters == cluster, 1], label=f'Cluster {cluster}')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Customer Clusters')
plt.legend()
plt.grid(True)
plt.show()

# Print cluster centers
print("\nCluster Centers (in original feature space):")
inverse_transformed_centers = preprocessor.named_transformers_['num'].inverse_transform(
    kmeans.cluster_centers_[:, :len(numeric_features)]
)
print("Cluster Centers (Age, Annual Income, Spending Score):")
print(inverse_transformed_centers)

# Adding explanations
print("\nExplanation:")
print("1. The Elbow Method plot shows the inertia for different numbers of clusters. The 'elbow' point, where the inertia starts decreasing more slowly, indicates the optimal number of clusters.")
print("2. The Silhouette Scores plot shows the average silhouette score for different numbers of clusters. A higher silhouette score indicates better-defined clusters.")
print("3. Based on the plots, we assume the optimal number of clusters is 4 for this example.")
print("4. The PCA plot visualizes the clusters in a 2D space, showing how the customers are grouped into different clusters.")
print("5. The cluster centers output shows the average values of Age, Annual Income, and Spending Score for each cluster.")
