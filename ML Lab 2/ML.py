import pandas as pd
import numpy as np

# Load Excel file
file_path = "Lab Session Data.xlsx"  # Ensure it's in the same folder
df = pd.read_excel(file_path, sheet_name="Purchase data")

# Extract relevant columns and drop NaNs
filtered_data = df[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)', 'Payment (Rs)']].dropna()

# A: Quantity matrix (m x n), C: Total payment vector (m x 1)
A = filtered_data[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']].to_numpy()
C = filtered_data[['Payment (Rs)']].to_numpy()

# 1. Dimensionality of the vector space
dimensionality = A.shape[1]

# 2. Number of vectors
num_vectors = A.shape[0]

# 3. Rank of Matrix A
rank_A = np.linalg.matrix_rank(A)

# 4. Cost of each product using pseudo-inverse
A_pinv = np.linalg.pinv(A)
X = A_pinv @ C  # Solving AX = C → X = A⁺C

# Display results
print(f"Dimensionality of vector space: {dimensionality}")
print(f"Number of vectors in the vector space: {num_vectors}")
print(f"Rank of Matrix A: {rank_A}")
print("Cost per unit of each product (Candy, Mango, Milk):")
print(X.flatten())
