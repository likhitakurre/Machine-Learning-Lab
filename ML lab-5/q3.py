import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score

# Load the dataset
df = pd.read_excel(r"\Users\iamre\OneDrive\Desktop\ML\ml_dataset.xlsx")

# Drop non-numeric or irrelevant columns
df_numeric = df.drop(columns=["Timestamp", "Patient_ID", "Gender", "Anaesthesia_Level"])

# Drop rows with missing values (if any)
df_numeric = df_numeric.dropna()

# Features (X) and Target (y)
X = df_numeric.drop(columns=["BIS"])
y = df_numeric["BIS"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
reg = LinearRegression()
reg.fit(X_train, y_train)

# Predictions
y_train_pred = reg.predict(X_train)
y_test_pred = reg.predict(X_test)

# Function to calculate metrics
def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mape, r2

# Evaluate on train and test sets
train_mse, train_rmse, train_mape, train_r2 = calculate_metrics(y_train, y_train_pred)
test_mse, test_rmse, test_mape, test_r2 = calculate_metrics(y_test, y_test_pred)

# Print results
print("Train Set Metrics:")
print(f"MSE: {train_mse:.2f}")
print(f"RMSE: {train_rmse:.2f}")
print(f"MAPE: {train_mape:.2e}")
print(f"R² Score: {train_r2:.4f}\n")

print("Test Set Metrics:")
print(f"MSE: {test_mse:.2f}")
print(f"RMSE: {test_rmse:.2f}")
print(f"MAPE: {test_mape:.2e}")
print(f"R² Score: {test_r2:.4f}")
