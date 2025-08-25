'''
A1. Please refer to the “Purchase Data” worksheet of Lab Session Data.xlsx. Please load the data 
and segregate them into 2 matrices A & C (following the nomenclature of AX = C). Do the following 
activities. 
• What is the dimensionality of the vector space for this data? 
• How many vectors exist in this vector space? 
• What is the rank of Matrix A? 
• Using Pseudo-Inverse find the cost of each product available for sale.  
(Suggestion: If you use Python, you can use numpy.linalg.pinv() function to get a 
pseudo-inverse.)
'''
import numpy as np
import pandas as pd

def load_purchase_data(file_path, sheet_name="Purchase data"):
    data = pd.read_excel(file_path, sheet_name=sheet_name)
    selected_columns = ['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)', 'Payment (Rs)']
    data = data[selected_columns]
    data = data.dropna()
    return data

def create_matrices(data):
    A = data.iloc[:, :-1].values
    C = data.iloc[:, -1].values.reshape(-1, 1)
    column_names = data.columns[:-1].tolist()
    return A, C, column_names

def analyze_matrix(A):
    dimensionality = A.shape[1]
    num_vectors = A.shape[0]
    rank = np.linalg.matrix_rank(A)
    return dimensionality, num_vectors, rank

def compute_product_costs(A, C):
    pseudo_inverse = np.linalg.pinv(A)
    product_costs = np.dot(pseudo_inverse, C)
    return product_costs

if __name__ == "__main__":
    file_path = r"C:\Users\iamre\OneDrive\Desktop\ML\Lab Session Data.xlsx"
    purchase_data = load_purchase_data(file_path)
    A, C, product_names = create_matrices(purchase_data)
    dimensionality, num_vectors, rank = analyze_matrix(A)
    product_costs = compute_product_costs(A, C)
    
    print("Dimensionality of the vector space:", dimensionality)
    print("Number of vectors in the vector space:", num_vectors)
    print("Rank of matrix A:", rank)
    print("\nCost per product:")
    for i, product in enumerate(product_names):
        print(f" {product}: Rs.{product_costs[i][0]:.2f}")
