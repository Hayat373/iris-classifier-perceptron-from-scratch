import numpy as np

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)  # ensure labels are -1, +1

        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.n_iters):
            # compute margins
            condition = y_ * (np.dot(X, self.weights) + self.bias) >= 1

            # gradient updates
            dw = np.zeros(n_features)
            db = 0

            for idx, cond in enumerate(condition):
                if cond:
                    dw += 2 * self.lambda_param * self.weights
                    db += 0
                else:
                    dw += 2 * self.lambda_param * self.weights - np.dot(X[idx], y_[idx])
                    db += -y_[idx]

            self.weights -= self.lr * dw / n_samples
            self.bias -= self.lr * db / n_samples

            # optional: track loss every 500 iters
            if i % 500 == 0 or i == self.n_iters - 1:
                hinge_losses = np.maximum(0, 1 - y_ * (np.dot(X, self.weights) + self.bias))
                loss = self.lambda_param * np.dot(self.weights, self.weights) + np.mean(hinge_losses)
                print(f"Epoch {i}, Loss: {loss:.4f}")

        print("Final weights:", self.weights, "Final bias:", self.bias)

    def predict(self, X):
        approx = np.dot(X, self.weights) + self.bias
        return np.sign(approx)
