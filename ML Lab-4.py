# knn_digits_modular.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA


def load_and_split_data():
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(
        digits.data, digits.target, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, digits


def evaluate_classification(knn, X_train, X_test, y_train, y_test):
    y_train_pred = knn.predict(X_train)
    y_test_pred = knn.predict(X_test)
    print("Train Confusion Matrix:\n", confusion_matrix(y_train, y_train_pred))
    print("Test Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))
    print("\nTrain Classification Report:\n", classification_report(y_train, y_train_pred))
    print("\nTest Classification Report:\n", classification_report(y_test, y_test_pred))


def regression_analysis(X, y):
    X_reg = np.mean(X, axis=1).reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X_reg, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"MSE: {mse:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}, R2 Score: {r2:.2f}")


def generate_2D_class_data():
    X = np.random.uniform(1, 10, (20, 2))
    y = np.random.choice([0, 1], 20)
    plt.scatter(X[:, 0], X[:, 1], c=['blue' if i == 0 else 'red' for i in y])
    plt.title("Training Data")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()
    return X, y


def classify_and_plot(X_train, y_train, k):
    x_vals = np.arange(0, 10.1, 0.1)
    y_vals = np.arange(0, 10.1, 0.1)
    X_test = np.array([[x, y] for x in x_vals for y in y_vals])
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=['blue' if i == 0 else 'red' for i in y_pred], s=1)
    plt.title(f"Test Data Prediction with k={k}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()


def pca_and_plot(digits):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(digits.data)
    X_train, X_test, y_train, y_test = train_test_split(X_pca, digits.target, test_size=0.2, random_state=42)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='tab10', s=10)
    plt.colorbar()
    plt.title("PCA-Reduced Digits")
    plt.xlabel("PCA1")
    plt.ylabel("PCA2")
    plt.grid(True)
    plt.show()


def hyperparameter_tuning(X_train, y_train):
    param_grid = {'n_neighbors': list(range(1, 20))}
    grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
    grid.fit(X_train, y_train)
    print("Best k:", grid.best_params_['n_neighbors'])
    print("Best score:", grid.best_score_)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, digits = load_and_split_data()

    print("\n--- A1: Classification Evaluation ---")
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    evaluate_classification(knn, X_train, X_test, y_train, y_test)

    print("\n--- A2: Regression Metrics ---")
    regression_analysis(digits.data, digits.target)

    print("\n--- A3: Generate and Plot 2D Training Data ---")
    X2D, y2D = generate_2D_class_data()

    print("\n--- A4 & A5: Classify Grid Points with Different k ---")
    for k in [1, 3, 5, 7, 9]:
        classify_and_plot(X2D, y2D, k)

    print("\n--- A6: PCA Plot of Digits Data ---")
    pca_and_plot(digits)

    print("\n--- A7: Hyperparameter Tuning ---")
    hyperparameter_tuning(X_train, y_train)
