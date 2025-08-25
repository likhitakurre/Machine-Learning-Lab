'''
A4. Data Exploration: Load the data available in “thyroid0387_UCI” worksheet. Perform the 
following tasks: 
• Study each attribute and associated values present. Identify the datatype (nominal etc.) 
for the attribute. 
• For categorical attributes, identify the encoding scheme to be employed. (Guidance: 
employ label encoding for ordinal variables while One-Hot encoding may be employed 
for nominal variables). 
• Study the data range for numeric variables. 
• Study the presence of missing values in each attribute. 
• Study presence of outliers in data.  
• For numeric variables, calculate the mean and variance (or standard deviation).
'''
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

def load_data(file_path, sheet_name="thyroid0387_UCI"):
    return pd.read_excel(file_path, sheet_name=sheet_name)

def get_attribute_types(df):
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    return categorical_columns, numeric_columns

def check_missing_values(df):
    return df.isnull().sum()

def get_unique_categorical_values(df, categorical_columns):
    unique_values = {}
    for col in categorical_columns:
        unique_values[col] = df[col].unique()
    return unique_values

def get_numeric_statistics(df, numeric_columns):
    stats = {}
    for col in numeric_columns:
        stats[col] = {
            "mean": df[col].mean(),
            "variance": df[col].var(),
            "std_dev": df[col].std(),
            "min": df[col].min(),
            "max": df[col].max()
        }
    return stats

def detect_outliers(df, numeric_columns, plot=False):
    outlier_info = {}
    for col in numeric_columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = df[(df[col] < lower) | (df[col] > upper)][col]
        outlier_info[col] = outliers
        if plot:
            sns.boxplot(x=df[col])
            plt.title(f"Boxplot for {col}")
            plt.show()
    return outlier_info

def suggest_encoding(unique_values):
    encoding_suggestions = {}
    for col, values in unique_values.items():
        try:
            sorted_vals = sorted(values.tolist())
            encoding_suggestions[col] = "Label Encoding (Ordinal)" if sorted_vals == values.tolist() else "One-Hot Encoding (Nominal)"
        except Exception:
            encoding_suggestions[col] = "One-Hot Encoding (Nominal)"
    return encoding_suggestions

def encode_categorical(df, categorical_columns):
    df_encoded = df.copy()
    le = LabelEncoder()
    for col in categorical_columns:
        df_encoded[col] = le.fit_transform(df[col].astype(str))
    return df_encoded

if __name__ == "__main__":
    file_path = r"C:\Users\iamre\OneDrive\Desktop\ML\Lab Session Data.xlsx"
    df = load_data(file_path)
    
    categorical_columns, numeric_columns = get_attribute_types(df)
    missing_values = check_missing_values(df)
    unique_values = get_unique_categorical_values(df, categorical_columns)
    numeric_stats = get_numeric_statistics(df, numeric_columns)
    outliers = detect_outliers(df, numeric_columns, plot=True)
    encoding_suggestions = suggest_encoding(unique_values)
    df_encoded = encode_categorical(df, categorical_columns)
    
    print("First 5 rows of dataset:")
    print(df.head())
    print("\nCategorical Columns:", categorical_columns)
    print("Numeric Columns:", numeric_columns)
    print("\nMissing Values:\n", missing_values)
    
    print("\nUnique Values in Categorical Columns:")
    for col, vals in unique_values.items():
        print(f"{col}: {vals}")
    
    print("\nNumeric Statistics:")
    for col, stats in numeric_stats.items():
        print(f"{col}: Mean={stats['mean']:.2f}, Variance={stats['variance']:.2f}, StdDev={stats['std_dev']:.2f}, Range=({stats['min']}, {stats['max']})")
    
    print("\nEncoding Suggestions:")
    for col, suggestion in encoding_suggestions.items():
        print(f"{col}: {suggestion}")
    
    print("\nEncoded Data Sample:")
    print(df_encoded.head())
