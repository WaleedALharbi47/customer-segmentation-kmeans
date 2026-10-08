"""
My own simple K-means implementation (for learning).

I wrote this to make sure I really understand the algorithm from the lecture
before using scikit-learn. Steps:
  1. pick K random points as the initial centroids
  2. assign every point to the closest centroid (Euclidean distance)
  3. move each centroid to the mean of its points
  4. repeat until the clusters stop changing
"""
import numpy as np


def kmeans(X, k, max_iters=100, seed=None):
    X = np.asarray(X, dtype=float)
    rng = np.random.default_rng(seed)

    # step 1: random initial centroids (chosen from the data points)
    centroids = X[rng.choice(len(X), size=k, replace=False)]
    labels = np.zeros(len(X), dtype=int)

    for i in range(max_iters):
        # step 2: distance from every point to every centroid
        distances = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        new_labels = distances.argmin(axis=1)

        # stop when no point changes its cluster
        if i > 0 and np.array_equal(new_labels, labels):
            break
        labels = new_labels

        # step 3: recompute centroids
        for c in range(k):
            if np.any(labels == c):  # avoid empty clusters
                centroids[c] = X[labels == c].mean(axis=0)

    return labels, centroids, i + 1


def sse(X, labels, centroids):
    """Sum of Squared Errors: squared distance of each point to its centroid."""
    X = np.asarray(X, dtype=float)
    return float(((X - centroids[labels]) ** 2).sum())


if __name__ == "__main__":
    # small example from the lecture slides (7 points, K=2)
    points = np.array([[1.0, 1.0], [1.5, 2.0], [3.0, 4.0], [5.0, 7.0],
                       [3.5, 5.0], [4.5, 5.0], [3.5, 4.5]])
    labels, centroids, iters = kmeans(points, k=2, seed=0)
    print("labels:", labels)
    print("centroids:\n", centroids.round(2))
    print("iterations:", iters, "| SSE:", round(sse(points, labels, centroids), 2))
