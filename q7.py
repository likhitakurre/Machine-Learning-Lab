import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from kneed import KneeLocator
# Load the dataset
df = pd.read_excel(r"\Users\iamre\OneDrive\Desktop\ML\ml_dataset.xlsx")
# Drop non-numeric or irrelevant columns
df_clustering = df.drop(columns=["Timestamp", "Patient_ID", "Gender", "Anaesthesia_Level", "BIS"]).dropna()
# Prepare distortion list
distortions = []
K_range=range(2,20)
# Run KMeans for k = 2 to 19
for k in range(2, 20):
    kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(df_clustering)
    distortions.append(kmeans.inertia_)  # Inertia = sum of squared distances to cluster centers
kneedle = KneeLocator(K_range, distortions, curve="convex", direction="decreasing")

optimal_k = kneedle.knee
# Plot the elbow graph
plt.figure(figsize=(8, 5))
plt.plot(range(2, 20), distortions, marker='o')
plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia (Distortion)")
plt.grid(True)
plt.show()
print(optimal_k)