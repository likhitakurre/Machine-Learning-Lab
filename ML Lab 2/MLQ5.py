import pandas as pd

# Load the Excel file
df = pd.read_excel("Lab Session Data.xlsx", sheet_name="thyroid0387_UCI")

# Clean: replace '?' with NaN and drop rows with missing values
df.replace("?", pd.NA, inplace=True)
df.dropna(inplace=True)

# Convert all entries to string and check for binary attributes
binary_cols = []
for col in df.columns:
    unique_vals = set(df[col].astype(str).unique())
    if unique_vals.issubset({'0', '1'}):
        binary_cols.append(col)

print("Binary Attributes Used:", binary_cols)

# Convert binary columns to integer
df[binary_cols] = df[binary_cols].astype(int)

# Reset index to ensure access to first 2 rows using iloc
df.reset_index(drop=True, inplace=True)

# Get first 2 vectors
vec1 = df.iloc[0][binary_cols]
vec2 = df.iloc[1][binary_cols]

# Compute similarity counts
f11 = ((vec1 == 1) & (vec2 == 1)).sum()
f00 = ((vec1 == 0) & (vec2 == 0)).sum()
f10 = ((vec1 == 1) & (vec2 == 0)).sum()
f01 = ((vec1 == 0) & (vec2 == 1)).sum()

# Jaccard Coefficient
jaccard = f11 / (f11 + f10 + f01) if (f11 + f10 + f01) > 0 else 0

# Simple Matching Coefficient
smc = (f11 + f00) / (f11 + f10 + f01 + f00)

# Print results
print(f"\nf11 = {f11}, f00 = {f00}, f10 = {f10}, f01 = {f01}")
print(f"Jaccard Coefficient (JC): {jaccard:.4f}")
print(f"Simple Matching Coefficient (SMC): {smc:.4f}")

# Appropriateness
if jaccard > smc:
    print("\n✅ JC is more conservative, best when 1s are more significant (like 'present' features).")
else:
    print("\n✅ SMC is more general, treating 0-0 and 1-1 matches equally.")
