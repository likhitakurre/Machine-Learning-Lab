import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Corrected file path (raw string or escaped backslashes)
file_path = r"C:\Users\Likhita\OneDrive\Desktop\PFL\Lab Session Data.xlsx"

# Load data
df = pd.read_excel(file_path, sheet_name='thyroid0387_UCI')

# Replace '?' with NaN and drop rows with missing data
df.replace('?', np.nan, inplace=True)
df.dropna(inplace=True)

# Convert all columns to numeric (if possible)
df = df.apply(pd.to_numeric, errors='ignore')

# One-hot encode the entire cleaned dataframe
df_encoded = pd.get_dummies(df)

# Now take the first two encoded vectors
vector1_enc = df_encoded.iloc[0].values.reshape(1, -1)
vector2_enc = df_encoded.iloc[1].values.reshape(1, -1)

# Compute cosine similarity
cos_sim = cosine_similarity(vector1_enc, vector2_enc)[0][0]

# Output
print(f"Cosine Similarity between vector 1 and vector 2: {cos_sim:.4f}")
