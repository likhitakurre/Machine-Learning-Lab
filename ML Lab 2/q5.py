'''
A5. Similarity Measure: Take the first 2 observation vectors from the dataset. Consider only the 
attributes (direct or derived) with binary values for these vectors (ignore other attributes). Calculate 
the Jaccard Coefficient (JC) and Simple Matching Coefficient (SMC) between the document vectors. 
Use first vector for each document for this. Compare the values for JC and SMC and judge the 
appropriateness of each of them. 
JC = (f11) / (f01+ f10+ f11) 
SMC = (f11 + f00) / (f00 + f01 + f10 + f11) 
f11= number of attributes where the attribute carries value of 1 in both 
the vectors.
'''
import pandas as pd

def load_data(file_path, sheet_name="thyroid0387_UCI"):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    df = df.replace({'t': 1, 'f': 0})
    return df

def get_binary_columns(df):
    binary_columns = [col for col in df.columns if set(df[col].dropna().unique()).issubset({0, 1})]
    return binary_columns

def get_vectors(df, binary_columns, idx1=0, idx2=1):
    v1 = df.iloc[idx1][binary_columns].astype(int)
    v2 = df.iloc[idx2][binary_columns].astype(int)
    return v1, v2

def calculate_similarity(v1, v2):
    f11 = ((v1 == 1) & (v2 == 1)).sum()
    f00 = ((v1 == 0) & (v2 == 0)).sum()
    f10 = ((v1 == 1) & (v2 == 0)).sum()
    f01 = ((v1 == 0) & (v2 == 1)).sum()

    jc = f11 / (f11 + f10 + f01) if (f11 + f10 + f01) > 0 else 0
    smc = (f11 + f00) / (f11 + f00 + f10 + f01)

    return {"f11": f11, "f00": f00, "f10": f10, "f01": f01, "JC": jc, "SMC": smc}

def interpret_similarity(jc, smc):
    if jc < smc:
        return (
            "SMC is higher because it considers both agreements: 0s and 1s. "
            "JC is stricter, useful when 1s represent meaningful presence (e.g., symptom ON)."
        )
    else:
        return "JC and SMC are close; dataset has more agreement on 1s or few disagreements overall."

if __name__ == "__main__":
    file_path = r"C:\Users\iamre\OneDrive\Desktop\ML\Lab Session Data.xlsx"
    
    df = load_data(file_path)
    binary_columns = get_binary_columns(df)
    v1, v2 = get_vectors(df, binary_columns)
    similarity_results = calculate_similarity(v1, v2)
    interpretation = interpret_similarity(similarity_results["JC"], similarity_results["SMC"])
    
    print("\nBinary Columns:", binary_columns)
    print("\n--- Similarity Measures ---")
    print(f"f11 (1 in both): {similarity_results['f11']}")
    print(f"f00 (0 in both): {similarity_results['f00']}")
    print(f"f10 (1 in vector1, 0 in vector2): {similarity_results['f10']}")
    print(f"f01 (0 in vector1, 1 in vector2): {similarity_results['f01']}")
    print(f"Jaccard Coefficient: {similarity_results['JC']:.4f}")
    print(f"Simple Matching Coefficient: {similarity_results['SMC']:.4f}")
    print("\n--- Interpretation ---")
    print(interpretation)
