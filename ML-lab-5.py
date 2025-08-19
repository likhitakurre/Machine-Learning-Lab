import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_iris, load_diabetes
from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score,
    mean_absolute_percentage_error,
    silhouette_score, calinski_harabasz_score, davies_bouldin_score
)
from kneed import KneeLocator


# ---------------------- DATA LOADERS ----------------------
def load_classification_data():
    """Load Iris dataset for classification examples."""
    data = load_iris(as_frame=True)
    return data.frame, data.target_names

def load_regression_data():
    """Load Diabetes dataset for regression examples."""
    data = load_diabetes(as_frame=True)
    return data.frame, data.target


# ---------------------- Q1: Regression + Logistic Classification ----------------------
def regression_and_classification():
    # Regression (Diabetes dataset)
    df, target = load_regression_data()
    X_reg = df[["bmi"]]   # one feature
    y_reg = target
    reg = LinearRegression().fit(X_reg, y_reg)
    y_pred_reg = reg.predict(X_reg)

    print("\n----- Regression (Diabetes dataset) -----")
    print("Coefficient:", reg.coef_[0])
    print("Intercept:", reg.intercept_)
    print("MSE:", mean_squared_error(y_reg, y_pred_reg))
    print("R² Score:", r2_score(y_reg, y_pred_reg))
    print("First 10 Predictions:", y_pred_reg[:10])

    # Classification (Iris dataset)
    df_cls, _ = load_classification_data()
    X_clf = df_cls[["sepal length (cm)"]]
    y_clf = df_cls["target"]
    clf = LogisticRegression(max_iter=1000).fit(X_clf, y_clf)
    y_pred_clf = clf.predict(X_clf)

    print("\n----- Classification (Iris dataset) -----")
    print("Accuracy:", accuracy_score(y_clf, y_pred_clf))
    print("First 10 Predictions:", y_pred_clf[:10])


# ---------------------- Q2: KNN Classification ----------------------
def knn_classification():
    df_cls, _ = load_classification_data()
    X = df_cls.drop(columns="target")
    y = df_cls["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)

    y_train_pred, y_test_pred = knn.predict(X_train), knn.predict(X_test)

    def metrics(y_true, y_pred):
        return {
            "MSE": mean_squared_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "MAPE": mean_absolute_percentage_error(y_true, y_pred),
            "R²": r2_score(y_true, y_pred)
        }

    metrics_df = pd.DataFrame([metrics(y_train, y_train_pred),
                               metrics(y_test, y_test_pred)],
                              index=["Train Set", "Test Set"])
    print("\n=== KNN Performance Metrics (Iris dataset) ===")
    print(metrics_df)


# ---------------------- Q3: Linear Regression with all numeric ----------------------
def linear_regression_all():
    df, target = load_regression_data()
    X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.2, random_state=42)

    reg = LinearRegression().fit(X_train, y_train)
    y_train_pred, y_test_pred = reg.predict(X_train), reg.predict(X_test)

    def metrics(y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return mse, rmse, mape, r2

    print("\n=== Linear Regression (Diabetes dataset, all features) ===")
    print("Train:", metrics(y_train, y_train_pred))
    print("Test :", metrics(y_test, y_test_pred))


# ---------------------- Q4: Simple KMeans ----------------------
def simple_kmeans():
    df_cls, _ = load_classification_data()
    X = df_cls.drop(columns="target")

    kmeans = KMeans(n_clusters=3, random_state=0, n_init="auto").fit(X)

    print("\n=== Simple KMeans (Iris dataset) ===")
    print("Cluster Labels:", np.unique(kmeans.labels_))
    print("Cluster Centers:\n", kmeans.cluster_centers_)


# ---------------------- Q5: KMeans with Metrics ----------------------
def kmeans_with_metrics():
    df_cls, _ = load_classification_data()
    X = df_cls.drop(columns="target")
    kmeans = KMeans(n_clusters=3, random_state=0, n_init="auto").fit(X)
    labels = kmeans.labels_

    print("\n=== KMeans with Metrics (Iris dataset) ===")
    print("Silhouette Score:", silhouette_score(X, labels))
    print("Calinski-Harabasz Score:", calinski_harabasz_score(X, labels))
    print("Davies-Bouldin Index:", davies_bouldin_score(X, labels))
    print("Centers:\n", kmeans.cluster_centers_)


# ---------------------- Q6: KMeans Metrics across K ----------------------
def kmeans_metrics_vs_k():
    df_cls, _ = load_classification_data()
    X = df_cls.drop(columns="target")
    k_values = range(2, 8)
    sil, ch, db = [], [], []

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(X)
        labels = kmeans.labels_
        sil.append(silhouette_score(X, labels))
        ch.append(calinski_harabasz_score(X, labels))
        db.append(davies_bouldin_score(X, labels))

    plt.figure(figsize=(14, 5))
    plt.subplot(1, 3, 1); plt.plot(k_values, sil, marker='o'); plt.title("Silhouette vs K")
    plt.subplot(1, 3, 2); plt.plot(k_values, ch, marker='o', color='g'); plt.title("CH Score vs K")
    plt.subplot(1, 3, 3); plt.plot(k_values, db, marker='o', color='r'); plt.title("DB Index vs K")
    plt.tight_layout(); plt.show()


# ---------------------- Q7: Elbow Method ----------------------
def elbow_method():
    df_cls, _ = load_classification_data()
    X = df_cls.drop(columns="target")
    distortions, K_range = [], range(2, 10)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(X)
        distortions.append(kmeans.inertia_)

    kneedle = KneeLocator(list(K_range), distortions, curve="convex", direction="decreasing")
    print("\n=== Elbow Method (Iris dataset) ===")
    print("Optimal K:", kneedle.knee)

    plt.plot(K_range, distortions, marker='o')
    plt.title("Elbow Method for Optimal K")
    plt.xlabel("K"); plt.ylabel("Inertia"); plt.grid(True)
    plt.show()


# ---------------------- MAIN ----------------------
def main():
    regression_and_classification()
    knn_classification()
    linear_regression_all()
    simple_kmeans()
    kmeans_with_metrics()
    kmeans_metrics_vs_k()
    elbow_method()


if __name__ == "__main__":
    main()
