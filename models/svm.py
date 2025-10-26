import numpy as np
from numpy.typing import NDArray
from typing import Optional

class SVM:
    """
    Linear Support Vector Machine implementation using gradient descent on the hinge loss.
    """

    def __init__(self, learning_rate: float = 0.0001, lambda_param: float = 0.001, n_iters: int = 5000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.weights: Optional[NDArray[np.float64]] = None
        self.bias: Optional[float] = None
        self.loss_history: list = []

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        """
        Train the SVM classifier.
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []

        y_ = np.where(y <= 0, -1, 1)

        for epoch in range(self.n_iters):
            total_loss = 0.0
            for idx, x_i in enumerate(X):
                assert self.weights is not None and self.bias is not None
                score = np.dot(x_i, self.weights) + self.bias
                condition = y_[idx] * score >= 1
                if condition:
                    self.weights -= self.lr * (2 * self.lambda_param * self.weights)
                    total_loss += 0
                else:
                    hinge_loss = 1 - y_[idx] * score
                    self.weights -= self.lr * (2 * self.lambda_param * self.weights - y_[idx] * x_i)
                    self.bias -= self.lr * y_[idx]
                    total_loss += hinge_loss

            self.loss_history.append(total_loss / n_samples)
            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss / n_samples:.4f}")

        print(f"Final weights: {self.weights}, Final bias: {self.bias}")

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.int_]:
        """
        Predict binary class labels for samples in X.
        """
        assert self.weights is not None and self.bias is not None
        linear_output = np.dot(X, self.weights) + self.bias
        y_pred = np.sign(linear_output)
        return np.where(y_pred >= 0, 1, 0)