import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


file_path =  r"C:\Users\Likhita\OneDrive\Desktop\PFL\Lab Session Data.xlsx"
df = pd.read_excel(file_path, sheet_name='thyroid0387_UCI')
df=df.replace('?', np.nan)  

df_imputed = df.copy()

print(df.dtypes)

for col in df_imputed.columns:
    if df_imputed[col].isnull().sum() > 0: 
        if df_imputed[col].dtype in ['float64', 'int64']:  

            Q1 = df_imputed[col].quantile(0.25)
            Q3 = df_imputed[col].quantile(0.75)
            IQR = Q3 - Q1
            lowerboundvalue = Q1 - 1.5 * IQR
            upperboundvalue = Q3 + 1.5 * IQR
            hasanyoutliers = ((df_imputed[col] < lowerboundvalue) | (df_imputed[col] > upperboundvalue)).any()
            
            if hasanyoutliers:
                medianvalue = df_imputed[col].median()
                df_imputed[col].fillna(medianvalue)
                print(f"Filled missing values in '{col}' with median: {medianvalue}")
            else:
                meanvalue = df_imputed[col].mean()
                df_imputed[col].fillna(meanvalue)
                print(f"Filled missing values in '{col}' with mean: {meanvalue}")
        else:
            modevalue = df_imputed[col].mode()[0]
            df_imputed[col].fillna(modevalue)
            print(f"Filled missing values in '{col}' with mode: {modevalue}")