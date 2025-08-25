'''
A2. Mark all customers (in “Purchase Data” table) with payments above Rs. 200 as RICH and others 
as POOR. Develop a classifier model to categorize customers into RICH or POOR class based on 
purchase behavior. 
'''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def load_data(file_path, sheet_name="Purchase data"):
    data = pd.read_excel(file_path, sheet_name=sheet_name)
    selected_columns = ['Customer', 'Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)', 'Payment (Rs)']
    data = data[selected_columns].copy()
    return data

def assign_class_labels(data, threshold=200):
    data['Class'] = data['Payment (Rs)'].apply(lambda x: 'RICH' if x > threshold else 'POOR')
    return data

def split_features_labels(data):
    X = data[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']]
    y = data['Class']
    return X, y

def train_classifier(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    model = RandomForestClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    return model, X_test, y_test

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    return report

if __name__ == "__main__":
    file_path = r"C:\Users\iamre\OneDrive\Desktop\ML\Lab Session Data.xlsx"
    
    data = load_data(file_path)
    data = assign_class_labels(data)
    X, y = split_features_labels(data)
    
    model, X_test, y_test = train_classifier(X, y)
    report = evaluate_model(model, X_test, y_test)
    
    print("Classification Report:\n")
    print(report)
    print("\nSample of updated dataset:")
    print(data.head())