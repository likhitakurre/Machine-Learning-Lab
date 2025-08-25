'''
A7. Heatmap Plot: Consider the first 20 observation vectors. Calculate the JC, SMC and COS between 
the pairs of vectors for these 20 vectors. Employ similar strategies for coefficient calculation as in A4 
& A5. Employ a heatmap plot to visualize the similarities.  
Suggestion to Python users → 
import seaborn as sns 
sns.heatmap(data, annot = True)
'''
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity

def load_binary_data(file_path, sheet_name="thyroid0387_UCI", n=20):
    data = pd.read_excel(file_path, sheet_name=sheet_name)
    data.columns = data.columns.str.strip()
    binary_data = data.select_dtypes(include=["int64", "bool"])
    return binary_data.head(n).astype(int).values

def compute_jaccard(X):
    return 1 - pairwise_distances(X, metric="jaccard")

def compute_smc(X):
    n = X.shape[0]
    smc = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            agree = np.sum(X[i] == X[j])
            smc[i, j] = agree / len(X[i])
    return smc

def compute_cosine(X):
    return cosine_similarity(X)

def plot_heatmaps(jaccard_sim, smc_sim, cosine_sim):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    sns.heatmap(jaccard_sim, annot=True, cmap="YlGnBu", ax=axes[0])
    axes[0].set_title("Jaccard Similarity (JC)")

    sns.heatmap(smc_sim, annot=True, cmap="YlGnBu", ax=axes[1])
    axes[1].set_title("Simple Matching Coefficient (SMC)")

    sns.heatmap(cosine_sim, annot=True, cmap="YlGnBu", ax=axes[2])
    axes[2].set_title("Cosine Similarity (COS)")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    file_path = r"C:\Users\iamre\OneDrive\Desktop\ML\Lab Session Data.xlsx"
    data_20 = load_binary_data(file_path, n=20)

    jaccard_sim = compute_jaccard(data_20)
    smc_sim = compute_smc(data_20)
    cosine_sim = compute_cosine(data_20)

    plot_heatmaps(jaccard_sim, smc_sim, cosine_sim)
