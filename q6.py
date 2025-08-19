import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from kneed import KneeLocator
# Load the dataset
df = pd.read_excel(r"\Users\iamre\OneDrive\Desktop\ML\ml_dataset.xlsx")

# Drop non-numeric or irrelevant columns
df_clustering = df.drop(columns=["Timestamp", "Patient_ID", "Gender", "Anaesthesia_Level", "BIS"]).dropna()

# Lists to store metrics
k_values = range(2, 11)  # Try k from 2 to 10
silhouette_scores = []

ch_scores = []
db_indexes = []

# Loop through different k values
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto")
    kmeans.fit(df_clustering)
    labels = kmeans.labels_

    # Calculate metrics
    silhouette_scores.append(silhouette_score(df_clustering, labels))
    ch_scores.append(calinski_harabasz_score(df_clustering, labels))
    db_indexes.append(davies_bouldin_score(df_clustering, labels))

# === Plotting ===
plt.figure(figsize=(14, 5))

# Silhouette Score plot
plt.subplot(1, 3, 1)
plt.plot(k_values, silhouette_scores, marker='o')
plt.title("Silhouette Score vs K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")

# Calinski-Harabasz Score plot
plt.subplot(1, 3, 2)
plt.plot(k_values, ch_scores, marker='o', color='green')
plt.title("Calinski-Harabasz Score vs K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("CH Score")

# Davies-Bouldin Index plot
plt.subplot(1, 3, 3)
plt.plot(k_values, db_indexes, marker='o', color='red')
plt.title("Davies-Bouldin Index vs K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("DB Index")

plt.tight_layout()
plt.show()
