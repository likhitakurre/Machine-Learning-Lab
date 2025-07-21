import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances


file_path = r"C:\Users\Likhita\OneDrive\Desktop\PFL\Lab Session Data.xlsx"
data = pd.read_excel(file_path, sheet_name="thyroid0387_UCI")

data.columns = data.columns.str.strip()

binarydata = data.select_dtypes(include=['int64', 'bool'])

data_20 = binarydata.head(20).astype(int).values

jaccardsimilarity = 1 - pairwise_distances(data_20, metric="jaccard")

def smc_similarity(X):
    n = X.shape[0]
    smc = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            agree = np.sum(X[i] == X[j])
            smc[i][j] = agree / len(X[i])
    return smc

smc_sim = smc_similarity(data_20)


cosine_sim = cosine_similarity(data_20)


fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.heatmap(jaccardsimilarity, annot=True, cmap="YlGnBu", ax=axes[0])
axes[0].set_title("Jaccard Similarity (JC)")

sns.heatmap(smc_sim, annot=True, cmap="YlGnBu", ax=axes[1])
axes[1].set_title("Simple Matching Coefficient (SMC)")

sns.heatmap(cosine_sim, annot=True, cmap="YlGnBu", ax=axes[2])
axes[2].set_title("Cosine Similarity (COS)")

plt.tight_layout()
plt.show()