import numpy as np
from models.decision_tree import DecisionTreeClassifier


class ExtraTreesNode:
    def __init__(self, feature=None, threshold=None,
                 left=None, right=None, value=None):
        self.feature   = feature
        self.threshold = threshold
        self.left      = left
        self.right     = right
        self.value     = value

    def is_leaf(self):
        return self.value is not None


class ExtraDecisionTree:
    """
    A decision tree variant used by Extra Trees.

    The key difference from a standard decision tree: instead of searching
    for the best threshold for each feature, we pick a random threshold
    within the feature's range. This makes each tree faster to build and
    introduces more diversity across the ensemble.
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

    def _random_split(self, X, y, feature_indices):
        best_gini      = float('inf')
        best_feature   = None
        best_threshold = None

        np.random.shuffle(feature_indices)
        for feature in feature_indices:
            col_min = X[:, feature].min()
            col_max = X[:, feature].max()
            if col_min == col_max:
                continue

            # Pick a uniformly random threshold — no grid search
            threshold  = np.random.uniform(col_min, col_max)
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

        if (depth >= self.max_depth or
                len(np.unique(y)) == 1 or
                n_samples < self.min_samples_split):
            return ExtraTreesNode(value=int(np.bincount(y.astype(int)).argmax()))

        n_features = self.n_features or n_cols
        feature_indices = np.random.choice(n_cols, n_features, replace=False)

        feature, threshold = self._random_split(X, y, feature_indices)

        if feature is None:
            return ExtraTreesNode(value=int(np.bincount(y.astype(int)).argmax()))

        left_mask  = X[:, feature] <= threshold
        right_mask = ~left_mask

        left  = self._build_tree(X[left_mask],  y[left_mask],  depth + 1)
        right = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return ExtraTreesNode(feature=feature, threshold=threshold, left=left, right=right)

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


class ExtraTreesClassifier:
    """
    Extra Trees (Extremely Randomized Trees) is similar to Random Forest
    but with two important differences:
      1. No bootstrap sampling — every tree trains on the full dataset
      2. Split thresholds are chosen randomly, not by exhaustive search

    This makes it faster than Random Forest and often just as accurate,
    since the extra randomness acts as a strong regularizer.
    """

    def __init__(self, n_estimators=100, max_depth=10,
                 min_samples_split=2, n_features=None, random_state=42):
        self.n_estimators      = n_estimators
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.n_features        = n_features
        self.random_state      = random_state
        self.trees             = []

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        np.random.seed(self.random_state)

        self.trees = []
        n_samples, n_cols = X.shape
        self.n_features = self.n_features or int(np.sqrt(n_cols))

        print(f"   Training {self.n_estimators} extra trees ", end="", flush=True)

        for i in range(self.n_estimators):
            # Unlike Random Forest, we use the full dataset — no bootstrap
            tree = ExtraDecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_features=self.n_features
            )
            tree.fit(X, y)
            self.trees.append(tree)

            if (i + 1) % 10 == 0:
                print(".", end="", flush=True)

        print(" done")
        return self

    def predict(self, X):
        X = np.array(X)
        all_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.array([
            np.bincount(all_preds[:, i].astype(int)).argmax()
            for i in range(X.shape[0])
        ])

    def predict_proba(self, X):
        X = np.array(X)
        all_preds = np.array([tree.predict(X) for tree in self.trees])
        proba = np.zeros((X.shape[0], 2))
        for i in range(X.shape[0]):
            counts = np.bincount(all_preds[:, i].astype(int), minlength=2)
            proba[i] = counts / self.n_estimators
        return proba