import numpy as np
import matplotlib.pyplot as plt
from utils.csv_utils import load_csv_data, train_test_split
from models.perceptron import Perceptron
from models.svm import SVM

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def plot_data(X, y, title="Data"):
    plt.figure(figsize=(8,6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', s=30)
    plt.title(title)
    plt.xlabel("Feature 1 (Sepal Length)")
    plt.ylabel("Feature 2 (Sepal Width)")
    plt.show()

def main():
    # Load data
    X, y = load_csv_data()
    
    # Filter to binary classes (0: setosa, 1: versicolor)
    mask = y < 2
    X = X[mask]
    y = y[mask]
    
    # Standardize features
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    X = (X - mean) / std

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Visualize (first two features)
    if X_train.shape[1] >= 2:
        plot_data(X_train[:, :2], y_train, title="Training Data (first two features)")

    # Train Perceptron
    perceptron = Perceptron(learning_rate=0.01, n_iters=1000)
    perceptron.fit(X_train, y_train)
    y_pred_p = perceptron.predict(X_test)
    print("Perceptron Accuracy:", accuracy(y_test, y_pred_p))

    # Train SVM
    svm_model = SVM(learning_rate=0.0001, lambda_param=0.001, n_iters=5000)
    svm_model.fit(X_train, y_train)
    y_pred_svm = svm_model.predict(X_test)
    print("SVM Accuracy:", accuracy(y_test, y_pred_svm))

    # Visualize predictions
    if X_test.shape[1] >= 2:
        plot_data(X_test[:, :2], y_pred_p, title="Perceptron Predictions (first two features)")
        plot_data(X_test[:, :2], y_pred_svm, title="SVM Predictions (first two features)")

    # Save scaling params and models
    np.save("data/mean.npy", mean)
    np.save("data/std.npy", std)
    np.save("models/perceptron_weights.npy", perceptron.weights)
    np.save("models/perceptron_bias.npy", perceptron.bias)
    np.save("models/svm_weights.npy", svm_model.weights)
    np.save("models/svm_bias.npy", svm_model.bias)

if __name__ == "__main__":
    main()