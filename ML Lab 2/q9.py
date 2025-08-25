'''
A9. Data Normalization / Scaling: from the data study, identify the attributes which may need 
normalization. Employ appropriate normalization techniques to create normalized set of data. 
'''
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def load_data(file_path, sheet_name):
    return pd.read_excel(file_path, sheet_name=sheet_name)

def identify_columns_to_normalize(df):
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    binary_cols = [col for col in numerical_cols if sorted(df[col].dropna().unique()) == [0, 1]]
    columns_to_normalize = [col for col in numerical_cols if col not in binary_cols]
    return columns_to_normalize

def apply_minmax_scaling(df, columns):
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[columns] = scaler.fit_transform(df[columns])
    return df_scaled

def apply_standard_scaling(df, columns):
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[columns] = scaler.fit_transform(df[columns])
    return df_scaled


if __name__ == "__main__":
    file_path = r"C:\Users\iamre\OneDrive\Desktop\ML\Lab Session Data.xlsx"
    sheet_name = "thyroid0387_UCI"

    df = load_data(file_path, sheet_name)
    cols_to_normalize = identify_columns_to_normalize(df)

    df_minmax_scaled = apply_minmax_scaling(df, cols_to_normalize)
    df_standard_scaled = apply_standard_scaling(df, cols_to_normalize)

    print("\nOriginal value range for normalization columns:")
    print(df[cols_to_normalize].describe())

    print("\nAfter Min-Max Normalization:")
    print(df_minmax_scaled[cols_to_normalize].describe())

    print("\nAfter Standard Scaling (mean ~0, std ~1):")
    print(df_standard_scaled[cols_to_normalize].describe())
