import numpy as np


class SVM:
    """
    A linear Support Vector Machine trained with gradient descent on hinge loss.

    The SVM tries to find a hyperplane (w · x + b = 0) that separates the two
    classes with the largest possible margin. Points that fall on the wrong side
    or inside the margin are penalized via hinge loss.

    Internally, labels are converted from 0/1 to -1/+1 because the math
    of the margin (y * (w·x + b) >= 1) requires that convention.

    lambda_param controls the regularization strength — higher values push
    the weights toward zero, which widens the margin but may underfit.
    """

    def __init__(self, learning_rate=0.001, lambda_param=0.01,
                 n_iterations=1000, random_state=42):
        self.learning_rate = learning_rate
        self.lambda_param  = lambda_param
        self.n_iterations  = n_iterations
        self.random_state  = random_state
        self.weights       = None
        self.bias          = None

    def _convert_labels(self, y):
        return np.where(y == 0, -1, 1)

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = self._convert_labels(np.array(y))
        np.random.seed(self.random_state)

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias    = 0.0

        print(f"   Training SVM for {self.n_iterations} iterations ", end="", flush=True)

        for i in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                # Check if the point satisfies the margin condition: y(w·x + b) >= 1
                correctly_outside_margin = y[idx] * (np.dot(x_i, self.weights) + self.bias) >= 1

                if correctly_outside_margin:
                    # No hinge loss — just shrink weights toward zero (regularization)
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights)
                else:
                    # Hinge loss is active — update weights and bias to push point to correct side
                    self.weights -= self.learning_rate * (
                        2 * self.lambda_param * self.weights - np.dot(x_i, y[idx])
                    )
                    self.bias += self.learning_rate * y[idx]

            if (i + 1) % (self.n_iterations // 10) == 0:
                print(".", end="", flush=True)

        print(" done")
        return self

    def predict(self, X):
        linear_output = np.dot(np.array(X, dtype=float), self.weights) + self.bias
        # Sign of the output gives us -1 or +1; convert back to 0/1
        return np.where(np.sign(linear_output) == -1, 0, 1)

    def predict_proba(self, X):
        """
        SVMs don't naturally output probabilities, so we approximate them
        by passing the raw decision scores through a sigmoid function.
        """
        scores = np.dot(np.array(X, dtype=float), self.weights) + self.bias
        prob_positive = 1 / (1 + np.exp(-np.clip(scores, -500, 500)))
        return np.column_stack([1 - prob_positive, prob_positive])