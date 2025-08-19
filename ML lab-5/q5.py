import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
# Load the dataset
df = pd.read_excel(r"\Users\iamre\OneDrive\Desktop\ML\ml_dataset.xlsx")
# Drop non-numeric or irrelevant columns
df_clustering = df.drop(columns=["Timestamp", "Patient_ID", "Gender", "Anaesthesia_Level", "BIS"])

# Drop rows with missing values (if any)
df_clustering = df_clustering.dropna()

# Split data for clustering (optional, you can also cluster on the full dataset)
X_train, X_test = train_test_split(df_clustering, test_size=0.2, random_state=42)

# Apply K-Means Clustering with k=2
kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto")
kmeans.fit(X_train)

# Get cluster labels and cluster centers
labels = kmeans.labels_
centers = kmeans.cluster_centers_
sil_score = silhouette_score(X_train, labels)
ch_score = calinski_harabasz_score(X_train, labels)
db_index = davies_bouldin_score(X_train, labels)

# Print results
print("=== K-Means Clustering Results ===")
print("Silhouette Score:", sil_score)
print("Calinski-Harabasz Score:", ch_score)
print("Davies-Bouldin Index:", db_index)
# Print results
print("Cluster Labels:")
print(labels)

print("\nCluster Centers:")
print(centers)
