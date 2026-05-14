import numpy as np


class GBTree:
    """
    A shallow regression tree used to fit residuals at each boosting step.
    Unlike the classification tree, this one predicts continuous values
    (the gradient residuals), not class labels.
    """

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

    def __init__(self, max_depth=3, min_samples_split=2):
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.root              = None

    def _mse(self, y):
        return np.var(y) if len(y) > 0 else 0

    def _best_split(self, X, y):
        best_mse       = float('inf')
        best_feature   = None
        best_threshold = None

        for feature in range(X.shape[1]):
            for threshold in np.unique(X[:, feature]):
                left_mask  = X[:, feature] <= threshold
                right_mask = ~left_mask

                if left_mask.sum() < self.min_samples_split or \
                   right_mask.sum() < self.min_samples_split:
                    continue

                n = len(y)
                mse = (left_mask.sum()  / n) * self._mse(y[left_mask]) + \
                      (right_mask.sum() / n) * self._mse(y[right_mask])

                if mse < best_mse:
                    best_mse       = mse
                    best_feature   = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _build_tree(self, X, y, depth=0):
        if (depth >= self.max_depth or
                len(y) < self.min_samples_split or
                len(np.unique(y)) == 1):
            return self.Node(value=np.mean(y))

        feature, threshold = self._best_split(X, y)

        if feature is None:
            return self.Node(value=np.mean(y))

        left_mask  = X[:, feature] <= threshold
        right_mask = ~left_mask

        return self.Node(
            feature=feature,
            threshold=threshold,
            left=self._build_tree(X[left_mask],  y[left_mask],  depth + 1),
            right=self._build_tree(X[right_mask], y[right_mask], depth + 1)
        )

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


class GradientBoostingClassifier:
    """
    Gradient Boosting builds an ensemble of trees sequentially. Each tree
    learns to correct the mistakes of all previous trees by fitting on the
    residuals (the difference between true labels and current predictions).

    We use log-loss as the objective:
      - Predictions are kept in log-odds space (F)
      - Residuals are computed as: y - sigmoid(F)
      - Each tree predicts these residuals, and F is updated by a small step

    Keeping max_depth shallow (default=3) is important — each tree should
    be a weak learner that only partially corrects the previous error.
    """

    def __init__(self, n_estimators=50, learning_rate=0.1,
                 max_depth=3, min_samples_split=2, random_state=42):
        self.n_estimators      = n_estimators
        self.learning_rate     = learning_rate
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.random_state      = random_state
        self.trees             = []
        self.initial_pred      = None

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        np.random.seed(self.random_state)

        # Start with a constant prediction: the log-odds of the mean class rate
        p = np.clip(np.mean(y), 1e-10, 1 - 1e-10)
        self.initial_pred = np.log(p / (1 - p))
        F = np.full(len(y), self.initial_pred, dtype=float)

        self.trees = []
        print(f"   Training {self.n_estimators} boosting rounds ", end="", flush=True)

        for i in range(self.n_estimators):
            # Residuals tell us how wrong the current ensemble is
            residuals = y - self._sigmoid(F)

            # Fit a regression tree to those residuals
            tree = GBTree(max_depth=self.max_depth,
                          min_samples_split=self.min_samples_split)
            tree.fit(X, residuals)
            self.trees.append(tree)

            # Take a small step in the direction of the residuals
            F += self.learning_rate * tree.predict(X)

            if (i + 1) % 10 == 0:
                print(".", end="", flush=True)

        print(" done")
        return self

    def _decision_function(self, X):
        X = np.array(X, dtype=float)
        F = np.full(X.shape[0], self.initial_pred, dtype=float)
        for tree in self.trees:
            F += self.learning_rate * tree.predict(X)
        return F

    def predict_proba(self, X):
        prob_positive = self._sigmoid(self._decision_function(X))
        return np.column_stack([1 - prob_positive, prob_positive])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)