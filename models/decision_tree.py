import numpy as np


class Node:
    def __init__(self, feature=None, threshold=None,
                 left=None, right=None, value=None):
        self.feature   = feature
        self.threshold = threshold
        self.left      = left
        self.right     = right
        self.value     = value

    def is_leaf(self):
        return self.value is not None


class DecisionTreeClassifier:
    """
    A binary decision tree that splits on the feature and threshold
    that minimize weighted Gini impurity at each node.

    When n_features is set, only a random subset of features is considered
    at each split — this is what Random Forest and Extra Trees rely on.
    """

    def __init__(self, max_depth=10, min_samples_split=2, n_features=None):
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.n_features        = n_features
        self.root              = None

    def _gini(self, y):
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1 - np.sum(probs ** 2)

    def _best_split(self, X, y, feature_indices):
        best_gini      = float('inf')
        best_feature   = None
        best_threshold = None

        for feature in feature_indices:
            for threshold in np.unique(X[:, feature]):
                left_mask  = X[:, feature] <= threshold
                right_mask = ~left_mask

                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue

                n = len(y)
                gini = (left_mask.sum()  / n) * self._gini(y[left_mask]) + \
                       (right_mask.sum() / n) * self._gini(y[right_mask])

                if gini < best_gini:
                    best_gini      = gini
                    best_feature   = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _build_tree(self, X, y, depth=0):
        n_samples, n_cols = X.shape

        # Stop growing if we've reached max depth, the node is pure, or too few samples remain
        if depth >= self.max_depth or len(np.unique(y)) == 1 or n_samples < self.min_samples_split:
            return Node(value=int(np.bincount(y.astype(int)).argmax()))

        # Randomly sample a subset of features if n_features is set (used by RF and Extra Trees)
        n_features = self.n_features or n_cols
        feature_indices = np.random.choice(n_cols, n_features, replace=False)

        feature, threshold = self._best_split(X, y, feature_indices)

        # If no valid split exists (e.g. all values identical), treat as leaf
        if feature is None:
            return Node(value=int(np.bincount(y.astype(int)).argmax()))

        left_mask  = X[:, feature] <= threshold
        right_mask = ~left_mask

        left  = self._build_tree(X[left_mask],  y[left_mask],  depth + 1)
        right = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return Node(feature=feature, threshold=threshold, left=left, right=right)

    def _traverse(self, node, x):
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse(node.left, x)
        return self._traverse(node.right, x)

    def fit(self, X, y):
        self.root = self._build_tree(np.array(X), np.array(y))
        return self

    def predict(self, X):
        return np.array([self._traverse(self.root, x) for x in np.array(X)])

    def predict_proba(self, X):
        preds = self.predict(X)
        proba = np.zeros((len(preds), 2))
        proba[:, 0] = (preds == 0).astype(float)
        proba[:, 1] = (preds == 1).astype(float)
        return proba