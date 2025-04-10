import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# Apriori Example
def run_apriori_example():
    # Create sample transaction data
    transactions = pd.DataFrame([
        [1,0,0,1,0,1,0,1],
        [0,1,1,0,0,1,0,1],
        [1,1,0,1,0,0,0,0],
        [0,0,1,0,1,1,0,1],
        [1,1,0,1,0,1,0,1]
    ])
    transactions.columns = ['bread', 'milk', 'cheese', 'eggs', 'juice', 'butter', 'yogurt', 'coffee']
    
    # Generate frequent itemsets
    frequent_itemsets = apriori(transactions, min_support=0.3, use_colnames=True)
    
    # Generate association rules
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)
    
    return rules

# Monte Carlo Example
def estimate_pi(n_points=1000):
    # Generate random points
    points = np.random.rand(n_points, 2)
    
    # Calculate points inside circle
    inside_circle = np.sum(np.sqrt(points[:,0]**2 + points[:,1]**2) <= 1)
    
    # Estimate pi
    pi_estimate = 4 * inside_circle / n_points
    
    # Visualize
    plt.figure(figsize=(8, 8))
    plt.scatter(points[:,0], points[:,1], c='blue', alpha=0.1)
    circle = plt.Circle((0, 0), 1, fill=False, color='red')
    plt.gca().add_artist(circle)
    plt.gca().set_aspect('equal')
    plt.title(f'Pi Estimate: {pi_estimate:.4f}')
    plt.savefig('monte_carlo.png')
    plt.close()
    
    return pi_estimate

# K-means Example
def run_kmeans_example():
    # Generate sample data
    np.random.seed(42)
    X = np.concatenate([
        np.random.normal(0, 1, (100, 2)),
        np.random.normal(4, 1, (100, 2)),
        np.random.normal(8, 1, (100, 2))
    ])
    
    # Fit K-means
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X)
    
    # Visualize
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                marker='x', s=200, linewidths=3, color='r', label='Centroids')
    plt.title('K-means Clustering Results')
    plt.legend()
    plt.savefig('kmeans.png')
    plt.close()
    
    return kmeans.cluster_centers_

# Decision Tree Example
def run_decision_tree_example():
    # Generate sample data
    np.random.seed(42)
    X = np.random.rand(100, 2)
    y = (X[:, 0] + X[:, 1] > 1).astype(int)
    
    # Train decision tree
    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(X, y)
    
    # Visualize decision boundary
    plt.figure(figsize=(8, 6))
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    plt.title('Decision Tree Classification')
    plt.savefig('decision_tree.png')
    plt.close()
    
    return clf.get_depth()

if __name__ == "__main__":
    # Run all examples
    print("Running Apriori example...")
    rules = run_apriori_example()
    print("\nAssociation Rules:")
    print(rules)
    
    print("\nRunning Monte Carlo example...")
    pi = estimate_pi()
    print(f"Pi estimate: {pi}")
    
    print("\nRunning K-means example...")
    centers = run_kmeans_example()
    print("Cluster centers:")
    print(centers)
    
    print("\nRunning Decision Tree example...")
    depth = run_decision_tree_example()
    print(f"Decision tree depth: {depth}")
