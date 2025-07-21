import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Step 1: Load the data
df = pd.read_excel("Lab Session Data.xlsx", sheet_name="Purchase data")

# Step 2: Clean and select relevant columns
df = df[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)', 'Payment (Rs)']].dropna()

# Step 3: Create the 'Label' column (RICH = 1, POOR = 0)
df['Label'] = df['Payment (Rs)'].apply(lambda x: 1 if x > 200 else 0)

# Step 4: Features (X) and Labels (y)
X = df[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']]
y = df['Label']

# Step 5: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 6: Train classifier (Decision Tree)
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train, y_train)

# Step 7: Predict and evaluate
y_pred = clf.predict(X_test)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["POOR", "RICH"]))
