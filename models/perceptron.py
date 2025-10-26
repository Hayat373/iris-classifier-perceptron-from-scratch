import numpy as np
from numpy.typing import NDArray
from typing import Optional

class Perceptron:
    """
    Simple Perceptron implementation from scratch using NumPy.
    Supports binary classification with labels {0, 1} or {-1, 1}.
    """

    def __init__(self, learning_rate: float = 0.01, n_iters: int = 1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights: Optional[NDArray[np.float64]] = None
        self.bias: Optional[float] = None

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        """
        Train the Perceptron model on the given dataset.
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        y_ = np.where(y <= 0, -1, 1)

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                assert self.weights is not None and self.bias is not None
                linear_output = np.dot(x_i, self.weights) + self.bias
                if y_[idx] * linear_output <= 0:
                    self.weights += self.lr * y_[idx] * x_i
                    self.bias += self.lr * y_[idx]

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.int_]:
        """
        Predict binary class labels for samples in X.
        """
        assert self.weights is not None and self.bias is not None
        linear_output = np.dot(X, self.weights) + self.bias
        y_pred = np.sign(linear_output)
        return np.where(y_pred >= 0, 1, 0)