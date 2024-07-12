import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the data from CSV file
file_path = 'Mall_Customers.csv'
df = pd.read_csv(file_path)

# Display the first few rows of the dataframe to verify the data loaded correctly
print(df.head())

# Selecting numerical features for clustering
X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine the optimal number of clusters using the elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plotting the elbow curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.xticks(range(1, 11))
plt.grid(True)
plt.show()

# Based on the elbow curve, let's choose the number of clusters
k = 5  # Change this number based on the elbow method plot

# Applying K-means clustering
kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Add cluster labels to the original dataframe
df['Cluster'] = clusters

# Display cluster centers
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
cluster_centers_df = pd.DataFrame(cluster_centers, columns=X.columns)
print("\nCluster Centers:")
print(cluster_centers_df)

# Visualizing the clusters
plt.figure(figsize=(12, 8))

for i in range(k):
    plt.scatter(X.iloc[clusters == i, 0], X.iloc[clusters == i, 1], label=f'Cluster {i}')

plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='X', color='k', s=300, label='Centroids')
plt.title('Clusters of Customers')
plt.xlabel('Age')
plt.ylabel('Annual Income (k$)')
plt.legend()
plt.grid(True)
plt.show()

# Optionally, you can save the updated dataframe with cluster labels
# df.to_csv('Mall_Customers_with_clusters.csv', index=False)
