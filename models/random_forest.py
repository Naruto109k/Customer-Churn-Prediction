import numpy as np
from models.decision_tree import DecisionTreeClassifier


class RandomForestClassifier:
    """
    Random Forest builds many decision trees on bootstrap samples of the data
    and combines their predictions by majority vote.

    Two sources of randomness make it robust:
      1. Each tree sees a different bootstrap sample (row-level randomness)
      2. Each split considers only sqrt(n_features) features (column-level randomness)
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

        print(f"   Training {self.n_estimators} trees ", end="", flush=True)

        for i in range(self.n_estimators):
            # Sample with replacement to create a bootstrap dataset
            indices = np.random.choice(n_samples, n_samples, replace=True)
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_features=self.n_features
            )
            tree.fit(X[indices], y[indices])
            self.trees.append(tree)

            if (i + 1) % 10 == 0:
                print(".", end="", flush=True)

        print(" done")
        return self

    def predict(self, X):
        X = np.array(X)
        # Collect every tree's predictions, then take the majority vote per sample
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