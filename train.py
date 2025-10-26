import numpy as np
from utils.csv_utils import load_csv_data
from models.perceptron import Perceptron
from models.svm import SVM

# Load data
X, y = load_csv_data()

# Filter to binary (setosa=0, versicolor=1)
mask = y < 2
X = X[mask]
y = y[mask]

# Standardize
mean = X.mean(axis=0)
std = X.std(axis=0)
X = (X - mean) / std

# Shuffle and split
indices = np.arange(X.shape[0])
np.random.seed(42)
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

n_train = int(0.8 * len(X))
X_train, X_test = X[:n_train], X[n_train:]
y_train, y_test = y[:n_train], y[n_train:]

print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# Train Perceptron
print("\n🔹 Training Perceptron...")
p = Perceptron(learning_rate=0.01, n_iters=1000)
p.fit(X_train, y_train)
y_pred_p = p.predict(X_test)
accuracy_p = np.mean(y_pred_p == y_test)
print(f"Perceptron Accuracy: {accuracy_p:.2f}")

# Train SVM
print("\n🔹 Training SVM...")
s = SVM(learning_rate=0.0001, lambda_param=0.001, n_iters=5000)
s.fit(X_train, y_train)
y_pred_s = s.predict(X_test)
accuracy_s = np.mean(y_pred_s == y_test)
print(f"SVM Accuracy: {accuracy_s:.2f}")

# Comparison
print("\n✅ Comparison:")
print(f"Perceptron Predictions: {y_pred_p}")
print(f"SVM Predictions:        {y_pred_s}")
print(f"True Labels:            {y_test}")

# Save scaling and models
np.save("data/mean.npy", mean)
np.save("data/std.npy", std)
np.save("models/perceptron_weights.npy", p.weights)
np.save("models/perceptron_bias.npy", p.bias)
np.save("models/svm_weights.npy", s.weights)
np.save("models/svm_bias.npy", s.bias)