'''
A8. Data Imputation: employ appropriate central tendencies to fill the missing values in the data 
variables. Employ following guidance. 
• Mean may be used when the attribute is numeric with no outliers 
• Median may be employed for attributes which are numeric and contain outliers 
• Mode may be employed for categorical attributes
'''
import pandas as pd
import numpy as np

def load_data(file_path, sheet_name):
    """Load dataset and replace '?' with NaN."""
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    df = df.replace('?', np.nan)
    return df

def detect_outliers(series):
    """Check if a numeric series contains outliers using IQR."""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return ((series < lower_bound) | (series > upper_bound)).any()

def impute_missing_values(df):
    """Impute missing values using mean, median, or mode."""
    df_imputed = df.copy()
    imputation_log = {}

    for column in df_imputed.columns:
        if df_imputed[column].isnull().sum() > 0:
            if df_imputed[column].dtype in ['float64', 'int64']:
                if detect_outliers(df_imputed[column]):
                    fill_value = df_imputed[column].median()
                    df_imputed[column] = df_imputed[column].fillna(fill_value)
                    imputation_log[column] = ("median", fill_value)
                else:
                    fill_value = df_imputed[column].mean()
                    df_imputed[column] = df_imputed[column].fillna(fill_value)
                    imputation_log[column] = ("mean", fill_value)
            else:
                fill_value = df_imputed[column].mode()[0]
                df_imputed[column] = df_imputed[column].fillna(fill_value)
                imputation_log[column] = ("mode", fill_value)
    return df_imputed, imputation_log
if __name__ == "__main__":
    file_path = r"C:\Users\iamre\OneDrive\Desktop\ML\Lab Session Data.xlsx"
    sheet_name = "thyroid0387_UCI"

    df = load_data(file_path, sheet_name)
    df_imputed, imputation_log = impute_missing_values(df)

    for column, (method, value) in imputation_log.items():
        print(f"Filled missing values in '{column}' with {method}: {value}")

