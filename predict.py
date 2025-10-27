import numpy as np
from models.perceptron import Perceptron
from models.svm import SVM

def load_model(model_type: str, weights_path: str, bias_path: str):
    """Load saved weights and bias for a model."""
    try:
        weights = np.load(weights_path)
        bias = np.load(bias_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Model files {weights_path} or {bias_path} not found. Run main.py or train.py first.")

    if model_type == "perceptron":
        model = Perceptron()
    elif model_type == "svm":
        model = SVM()
    else:
        raise ValueError("Model type must be 'perceptron' or 'svm'")

    model.weights = weights
    model.bias = float(bias)
    return model


def get_confidence(model, X: np.ndarray) -> np.ndarray:
    """Compute confidence (absolute distance to decision boundary)."""
    linear_output = np.dot(X, model.weights) + model.bias
    return np.abs(linear_output)


def main():
    # Load scaling parameters
    try:
        mean = np.load("data/mean.npy")
        std = np.load("data/std.npy")
    except FileNotFoundError:
        print("❌ Scaling files (mean.npy, std.npy) not found. Run main.py or train.py first.")
        return

    # Load models
    try:
        perceptron = load_model("perceptron", "models/perceptron_weights.npy", "models/perceptron_bias.npy")
        svm = load_model("svm", "models/svm_weights.npy", "models/svm_bias.npy")
    except FileNotFoundError as e:
        print(e)
        return

    print("\n🌸 Iris Flower Classifier 🌸")
    print("Enter measurements for classification (Setosa vs. Versicolor)\n")

    try:
        sepal_length = float(input("Sepal length (cm): "))
        sepal_width = float(input("Sepal width (cm): "))
        petal_length = float(input("Petal length (cm): "))
        petal_width = float(input("Petal width (cm): "))
    except ValueError:
        print("❌ Error: Please enter valid numeric values.")
        return

    # Prepare input
    X = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    X = (X - mean) / std

    # Predictions
    pred_p = perceptron.predict(X)[0]
    pred_svm = svm.predict(X)[0]
    conf_p = get_confidence(perceptron, X)[0]
    conf_svm = get_confidence(svm, X)[0]

    class_names = {0: "Setosa 🌿", 1: "Versicolor 🌺"}

    print("\n--- RESULTS ---")
    print(f"Perceptron Prediction: {class_names[pred_p]}  (Confidence: {conf_p:.3f})")
    print(f"SVM Prediction:        {class_names[pred_svm]}  (Confidence: {conf_svm:.3f})")
    print("-----------------------")


if __name__ == "__main__":
    main()
