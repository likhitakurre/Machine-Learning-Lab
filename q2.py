import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score

# Load the Excel dataset
df = pd.read_excel(r"\Users\iamre\OneDrive\Desktop\ML\ml_dataset.xlsx", sheet_name='Sheet1')

# Define features (X) and target (y)
X = df[['EEG_Channel_1', 'EEG_Channel_2', 'EEG_Channel_3',
        'ECG_Channel_1', 'ECG_Channel_2', 'ECG_Channel_3', 'BIS']]
y = df['Anaesthesia_Level']

# Encode target labels into numeric values
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3, random_state=42
)

# Train kNN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Predictions
y_train_pred = knn.predict(X_train)
y_test_pred = knn.predict(X_test)

# Regression-style metrics (even though this is a classification problem)
# ---- Training Set
mse_train = mean_squared_error(y_train, y_train_pred)
rmse_train = np.sqrt(mse_train)
mape_train = mean_absolute_percentage_error(y_train, y_train_pred)
r2_train = r2_score(y_train, y_train_pred)

# ---- Test Set
mse_test = mean_squared_error(y_test, y_test_pred)
rmse_test = np.sqrt(mse_test)
mape_test = mean_absolute_percentage_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)

# Create a summary DataFrame for comparison
metrics_df = pd.DataFrame({
    "MSE": [mse_train, mse_test],
    "RMSE": [rmse_train, rmse_test],
    "MAPE": [mape_train, mape_test],
    "R²": [r2_train, r2_test]
}, index=["Train Set", "Test Set"])

print("=== Model Performance Metrics ===")
print(metrics_df)