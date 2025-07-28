import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# ------------------ A1: Intraclass Spread and Interclass Distance ------------------
def analyze_class_spread_and_distance(X, y, class1=0, class2=1):
    X1 = X[y == class1]
    X2 = X[y == class2]

    centroid1 = X1.mean(axis=0)
    centroid2 = X2.mean(axis=0)

    spread1 = X1.std(axis=0)
    spread2 = X2.std(axis=0)

    interclass_dist = np.linalg.norm(centroid1 - centroid2)

    print(f"A1: Spread and Interclass Distance")
    print(f"Class {class1} - Centroid: {centroid1}, Spread: {spread1}")
    print(f"Class {class2} - Centroid: {centroid2}, Spread: {spread2}")
    print(f"Interclass Distance: {interclass_dist:.4f}\n")

# ------------------ A2: Feature Density Analysis ------------------
def analyze_feature_density(X, feature_index=0):
    feature = X[:, feature_index]
    mean = np.mean(feature)
    variance = np.var(feature)

    print(f"A2: Feature {feature_index} - Mean: {mean:.4f}, Variance: {variance:.4f}")

    plt.hist(feature, bins=10, color='skyblue', edgecolor='black')
    plt.title(f"Histogram of Feature {feature_index}")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

# ------------------ A3: Minkowski Distance Plot ------------------
def minkowski_distance_plot(X, idx1=0, idx2=1):
    vec1 = X[idx1]
    vec2 = X[idx2]

    r_values = np.arange(1, 11)
    distances = [np.linalg.norm(vec1 - vec2, ord=r) for r in r_values]

    print(f"A3: Minkowski Distances between vector {idx1} and {idx2}: {distances}")

    plt.plot(r_values, distances, marker='o')
    plt.title("Minkowski Distance (r=1 to 10)")
    plt.xlabel("r (Order)")
    plt.ylabel("Distance")
    plt.grid(True)
    plt.show()

# ------------------ A4: Train/Test Split ------------------
def perform_train_test_split(X, y, class1=0, class2=1, test_size=0.3):
    mask = np.isin(y, [class1, class2])
    X_filtered = X[mask]
    y_filtered = y[mask]
    X_train, X_test, y_train, y_test = train_test_split(X_filtered, y_filtered, test_size=test_size, random_state=42)
    print(f"A4: Train/Test Split Done - Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test

# ------------------ A5/A6/A7: kNN Training, Testing and Prediction ------------------
def train_and_test_knn(X_train, X_test, y_train, y_test, k=3):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    predictions = model.predict(X_test)

    print(f"A6: Accuracy with k={k}: {accuracy:.4f}")
    print(f"A7: Predictions on test set:\n{predictions}")
    return model, accuracy, predictions

# ------------------ A8: Accuracy vs K Plot ------------------
def accuracy_vs_k_plot(X_train, X_test, y_train, y_test, max_k=11):
    ks = list(range(1, max_k+1))
    accuracies = []

    for k in ks:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)
        accuracies.append(acc)

    plt.plot(ks, accuracies, marker='o')
    plt.title("Accuracy vs k in kNN")
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.show()

    print(f"A8: Accuracy scores for k=1 to {max_k}:\n{accuracies}")

# ------------------ A9: Confusion Matrix & Report ------------------
def evaluate_model(model, X_train, y_train, X_test, y_test):
    print("A9: Evaluation Metrics")
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    print("\nTrain Classification Report:")
    print(classification_report(y_train, y_train_pred))
    print("Train Confusion Matrix:")
    print(confusion_matrix(y_train, y_train_pred))

    print("\nTest Classification Report:")
    print(classification_report(y_test, y_test_pred))
    print("Test Confusion Matrix:")
    print(confusion_matrix(y_test, y_test_pred))

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    if train_acc > 0.95 and test_acc < 0.8:
        print("Model is likely Overfitting.")
    elif train_acc < 0.7 and test_acc < 0.7:
        print("Model is likely Underfitting.")
    else:
        print("Model appears to be Regularfit.")

# ------------------ Main Driver ------------------
def main():
    data = load_wine()
    X = data.data
    y = data.target
    feature_names = data.feature_names

    analyze_class_spread_and_distance(X, y)
    analyze_feature_density(X, feature_index=0)
    minkowski_distance_plot(X, idx1=0, idx2=1)

    X_train, X_test, y_train, y_test = perform_train_test_split(X, y, class1=0, class2=1)
    model, accuracy, predictions = train_and_test_knn(X_train, X_test, y_train, y_test, k=3)
    accuracy_vs_k_plot(X_train, X_test, y_train, y_test)
    evaluate_model(model, X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    main()
