import numpy as np
import matplotlib.pyplot as plt
from utils.csv_utils import load_csv_data
from models.perceptron import Perceptron
from models.svm import SVM
import os

def plot_data_with_boundary(X, y, model, title="Decision Boundary"):
    """Plot 2D scatter of data points with the model’s decision boundary."""
    if X.shape[1] < 2:
        print("❌ Not enough features to plot decision boundary.")
        return

    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', s=30, edgecolors='k')

    # Decision boundary: w1*x1 + w2*x2 + b = 0  ->  x2 = -(w1*x1 + b)/w2
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x1 = np.linspace(x_min, x_max, 100)
    w = model.weights
    b = model.bias
    x2 = -(w[0] * x1 + b) / w[1]

    plt.plot(x1, x2, 'k-', linewidth=2, label=f"{model.__class__.__name__} Boundary")
    plt.title(title)
    plt.xlabel("Feature 1 (Standardized)")
    plt.ylabel("Feature 2 (Standardized)")
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    # 1️⃣ Load and preprocess data
    X, y = load_csv_data(path="data/iris.csv", label_col=-1)

    # Use petal features (columns 2 and 3) — these are linearly separable
    X = X[:, 2:4]

    # Binary classification: Setosa vs. Versicolor+Virginica
    y = np.where(y == 0, -1, 1)  # FIXED: use -1, +1 for correct math

    # Standardize
    mean, std = X.mean(axis=0), X.std(axis=0)
    X = (X - mean) / std

    # Train-test split
    np.random.seed(42)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    split = int(0.8 * len(X))
    X_train, X_test = X[indices[:split]], X[indices[split:]]
    y_train, y_test = y[indices[:split]], y[indices[split:]]

    print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

    # 3️⃣ Train Perceptron
    print("\n🔹 Training Perceptron...")
    p = Perceptron(learning_rate=0.01, n_iters=1000)
    p.fit(X_train, y_train)
    y_pred_p = p.predict(X_test)
    acc_p = np.mean(y_pred_p == y_test)
    print(f"Perceptron Accuracy: {acc_p:.2f}")

    # 4️⃣ Train SVM
    print("\n🔹 Training SVM...")
    print("\n🔹 Training SVM...")
    s = SVM(learning_rate=0.001, lambda_param=0.01, n_iters=5000)

    s.fit(X_train, y_train)
    y_pred_s = s.predict(X_test)
    acc_s = np.mean(y_pred_s == y_test)
    print(f"SVM Accuracy: {acc_s:.2f}")

    # 5️⃣ Plot boundaries
    plot_data_with_boundary(X_train, y_train, p, "Perceptron Decision Boundary")
    plot_data_with_boundary(X_train, y_train, s, "SVM Decision Boundary")

    # 6️⃣ Save models and normalization
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    np.save("data/mean.npy", mean)
    np.save("data/std.npy", std)
    np.save("models/perceptron_weights.npy", p.weights)
    np.save("models/perceptron_bias.npy", p.bias)
    np.save("models/svm_weights.npy", s.weights)
    np.save("models/svm_bias.npy", s.bias)

    print("\n✅ Models and scaling saved successfully!")


if __name__ == "__main__":
    main()
