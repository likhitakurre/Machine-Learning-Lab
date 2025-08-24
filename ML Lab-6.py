import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from collections import Counter
import networkx as nx

########################################
# A1. Entropy + Equal Width Binning
########################################

def equal_width_binning(data, bins=4):
    """Perform equal width binning for continuous data."""
    min_val, max_val = np.min(data), np.max(data)
    bin_width = (max_val - min_val) / bins
    bin_edges = [min_val + i * bin_width for i in range(bins+1)]
    binned = np.digitize(data, bin_edges, right=False)
    return binned, bin_edges

def entropy(y):
    """Calculate entropy of a dataset."""
    counts = np.bincount(y)
    probabilities = counts[np.nonzero(counts)] / len(y)
    return -np.sum(probabilities * np.log2(probabilities))

########################################
# A2. Gini Index
########################################

def gini_index(y):
    counts = np.bincount(y)
    probabilities = counts[np.nonzero(counts)] / len(y)
    return 1 - np.sum(probabilities ** 2)

########################################
# A3. Root Node Selection using Information Gain
########################################

def information_gain(y, x):
    """Compute information gain of feature x for target y."""
    total_entropy = entropy(y)
    values, counts = np.unique(x, return_counts=True)
    weighted_entropy = np.sum([
        (counts[i] / np.sum(counts)) * entropy(y[x == values[i]])
        for i in range(len(values))
    ])
    return total_entropy - weighted_entropy

def best_feature(X, y):
    """Select the best feature as root node."""
    gains = [information_gain(y, X[:, i]) for i in range(X.shape[1])]
    return np.argmax(gains), gains

########################################
# A4. Flexible Binning Function
########################################

def binning(data, bins=4, method='equal_width'):
    """Binning function with method selection."""
    if method == 'equal_width':
        return equal_width_binning(data, bins)[0]
    elif method == 'frequency':
        return pd.qcut(data, q=bins, labels=False, duplicates='drop')
    else:
        raise ValueError("Method must be 'equal_width' or 'frequency'")

########################################
# A5. Decision Tree Module (Recursive)
########################################
class DecisionTree:
    def __init__(self, max_depth=3, bins=4, binning_method='equal_width'):
        self.max_depth = max_depth
        self.bins = bins
        self.binning_method = binning_method
        self.tree = None

    def fit(self, X, y, depth=0):
        if len(set(y)) == 1 or depth == self.max_depth:
            return Counter(y).most_common(1)[0][0]

        # Bin continuous features
        X_binned = np.zeros_like(X)
        for i in range(X.shape[1]):
            X_binned[:, i] = binning(X[:, i], self.bins, self.binning_method)

        best_feat, _ = best_feature(X_binned, y)
        tree = {f'Feature {best_feat}': {}}

        for value in np.unique(X_binned[:, best_feat]):
            idx = np.where(X_binned[:, best_feat] == value)
            subtree = self.fit(X[idx], y[idx], depth + 1)
            tree[f'Feature {best_feat}'][value] = subtree
        self.tree = tree
        return tree

########################################
# A6. Visualize Decision Tree with networkx
########################################

def visualize_tree(tree):
    """Visualize a decision tree dictionary using networkx."""
    G = nx.DiGraph()

    def add_edges(parent_name, child_dict):
        for edge_label, child in child_dict.items():
            if isinstance(child, dict):
                for feature, subtree in child.items():
                    node_name = f"{feature}|{edge_label}"
                    G.add_edge(parent_name, node_name, label=str(edge_label))
                    add_edges(node_name, subtree)
            else:
                leaf_name = f"Class {child}|{edge_label}"
                G.add_edge(parent_name, leaf_name, label=str(edge_label))

    # Root
    root = list(tree.keys())[0]
    add_edges(root, tree[root])

    pos = nx.spring_layout(G)
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw(G, pos, with_labels=True, node_size=2500, node_color='lightblue', font_size=8)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title("Decision Tree Visualization")
    plt.show()

########################################
# A7. Decision Boundary Visualization
########################################
from sklearn.tree import DecisionTreeClassifier
from matplotlib.colors import ListedColormap

def plot_decision_boundary(X, y):
    clf = DecisionTreeClassifier(max_depth=3)
    clf.fit(X, y)
    
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k')
    plt.title("Decision Boundary (2 features)")
    plt.show()

########################################
# Run Example on Iris dataset
########################################
if __name__ == "__main__":
    iris = load_iris()
    X, y = iris.data, iris.target

    # A1 Example
    petal_length_binned, edges = equal_width_binning(X[:, 2])
    print("Entropy of target:", entropy(y))

    # A2 Example
    print("Gini Index of target:", gini_index(y))

    # A3 Example
    idx, gains = best_feature(X[:, :2], y)  # first 2 features
    print("Best feature index:", idx, "with gains:", gains)

    # A5 Example (build tree)
    dt = DecisionTree(max_depth=2)
    tree_struct = dt.fit(X[:, :2], y)
    print("Constructed Tree:", tree_struct)

    # A6 Example (visualize our custom tree)
    visualize_tree(tree_struct)

    # A7 Example (2 features)
    plot_decision_boundary(X[:, :2], y)
