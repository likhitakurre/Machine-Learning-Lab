import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_excel("Lab Session Data.xlsx", sheet_name="thyroid0387_UCI")

# 🔁 Replace '?' with np.nan for uniform missing value handling
df.replace('?', np.nan, inplace=True)

# 1️⃣ Inspect attributes and data types
print("Column Names:\n", df.columns)
print("\nData Types:\n", df.dtypes)
print("\nSample Values:\n", df.head())

# 2️⃣ Identify categorical vs numeric
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# 🔄 Try converting any numeric-looking object columns
for col in categorical_cols[:]:  # work on a copy of the list
    try:
        df[col] = pd.to_numeric(df[col])
        numeric_cols.append(col)
        categorical_cols.remove(col)
    except ValueError:
        continue

print("\nCategorical Columns:", categorical_cols)
print("Numeric Columns:", numeric_cols)

# 3️⃣ Encoding suggestions
print("\nSuggested Encoding Schemes:")
for col in categorical_cols:
    unique_vals = df[col].dropna().unique().tolist()
    if sorted(unique_vals) == unique_vals:
        print(f"Ordinal: {col} → Use Label Encoding")
    else:
        print(f"Nominal: {col} → Use One-Hot Encoding")

# 4️⃣ Data range for numeric variables
print("\nRange for Numeric Variables:")
for col in numeric_cols:
    print(f"{col}: Min = {df[col].min()}, Max = {df[col].max()}")

# 5️⃣ Check for missing values
print("\nMissing Values per Column:")
print(df.isnull().sum())

# 6️⃣ Outlier Detection (Boxplots)
print("\nGenerating boxplots for numeric columns to spot outliers...")
for col in numeric_cols:
    plt.figure(figsize=(6, 1.5))
    sns.boxplot(data=df, x=col)
    plt.title(f"Outliers in {col}")
    plt.tight_layout()
    plt.show()

# 6️⃣➕ Outlier count summary (optional but useful)
print("\nOutlier Summary (IQR method):")
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
    print(f"{col}: {len(outliers)} outliers")

# 7️⃣ Mean and Standard Deviation for numeric columns
print("\nMean and Std Deviation for Numeric Columns:")
for col in numeric_cols:
    mean = df[col].mean()
    std = df[col].std()
    print(f"{col}: Mean = {mean:.2f}, Std Dev = {std:.2f}")

