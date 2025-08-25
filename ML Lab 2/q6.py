'''
A6. Cosine Similarity Measure: Now take the complete vectors for these two observations (including 
all the attributes). Calculate the Cosine similarity between the documents by using the second 
feature vector for each document.
'''
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def load_data(file_path, sheet_name="thyroid0387_UCI"):
    return pd.read_excel(file_path, sheet_name=sheet_name)

def encode_data(df):
    return pd.get_dummies(df)

def get_vectors(df_encoded, idx1=0, idx2=1):
    v1 = df_encoded.iloc[idx1].values.reshape(1, -1)
    v2 = df_encoded.iloc[idx2].values.reshape(1, -1)
    return v1, v2

def compute_cosine_similarity(v1, v2):
    return cosine_similarity(v1, v2)[0][0]

if __name__ == "__main__":
    file_path = r"C:\Users\iamre\OneDrive\Desktop\ML\Lab Session Data.xlsx"
    
    df = load_data(file_path)
    df_encoded = encode_data(df)
    v1, v2 = get_vectors(df_encoded, idx1=0, idx2=1)
    cos_sim = compute_cosine_similarity(v1, v2)
    
    print(f"Cosine Similarity between vector 1 and vector 2: {cos_sim:.4f}")
