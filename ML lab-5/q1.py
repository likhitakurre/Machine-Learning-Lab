import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

# Load dataset
file_path = r"C:\Users\iamre\OneDrive\Desktop\ML\ml_dataset.xlsx"
df = pd.read_excel(file_path, sheet_name="Sheet1")

# ---------------------- REGRESSION ----------------------
# X = one numerical attribute (EEG_Channel_1), y = BIS
X_train_reg = df[['EEG_Channel_1']]
y_train_reg = df['BIS']

# Train linear regression model
reg = LinearRegression().fit(X_train_reg, y_train_reg)

# Predictions
y_train_pred_reg = reg.predict(X_train_reg)

# Evaluate regression
print("----- Regression -----")
print("Coefficient:", reg.coef_[0])
print("Intercept:", reg.intercept_)
print("MSE:", mean_squared_error(y_train_reg, y_train_pred_reg))
print("R² Score:", r2_score(y_train_reg, y_train_pred_reg))
print("First 10 Predictions:", y_train_pred_reg[:10])

# ---------------------- CLASSIFICATION ----------------------
# X = one numerical attribute (EEG_Channel_1), y = Anaesthesia_Level
X_train_clf = df[['EEG_Channel_1']]
y_train_clf = df['Anaesthesia_Level']

# Train logistic regression model (for classification)
clf = LogisticRegression(max_iter=1000).fit(X_train_clf, y_train_clf)

# Predictions
y_train_pred_clf = clf.predict(X_train_clf)

# Evaluate classification
print("\n----- Classification -----")
print("Accuracy:", accuracy_score(y_train_clf, y_train_pred_clf))
print("First 10 Predictions:", y_train_pred_clf[:10])
