import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from models.perceptron import Perceptron
from models.svm import SVM
from utils.csv_utils import load_csv_data

# ------------------ Model Loading ------------------ #
@st.cache_resource
def load_models():
    try:
        # Load mean and std for first 2 features
        mean = np.load("data/mean.npy")[:2]
        std = np.load("data/std.npy")[:2]

        perceptron = Perceptron()
        perceptron.weights = np.load("models/perceptron_weights.npy")
        perceptron.bias = float(np.load("models/perceptron_bias.npy"))

        svm = SVM()
        svm.weights = np.load("models/svm_weights.npy")
        svm.bias = float(np.load("models/svm_bias.npy"))

    except FileNotFoundError:
        st.error("⚠️ Model or scaling files not found. Please run main.py first.")
        return None, None, None, None

    return perceptron, svm, mean, std

perceptron, svm, mean, std = load_models()

# ------------------ Dataset Loading ------------------ #
X, y = load_csv_data()
y = np.where(y == 0, 0, 1)

# Use only first two features
X_2f = X[:, :2]

# Recompute mean/std if shape mismatch
if mean.shape[0] != X_2f.shape[1] or std.shape[0] != X_2f.shape[1]:
    st.warning("Mean/Std shape mismatch, recomputing from data...")
    mean = X_2f.mean(axis=0)
    std = X_2f.std(axis=0)
    np.save("data/mean.npy", mean)
    np.save("data/std.npy", std)

X_norm = (X_2f - mean) / std

# ------------------ App UI ------------------ #
st.title("🌸 Iris Classifier – Perceptron vs SVM")
st.markdown("""
This interactive app uses **custom-built Perceptron and SVM models** (no scikit-learn)  
to classify *Iris flowers* as **Setosa (0)** or **Versicolor (1)** using **Sepal Length & Width**.
""")

# ------------------ User Input ------------------ #
col1, col2 = st.columns(2)
with col1:
    sepal_length = st.number_input("Sepal Length (cm)", 4.0, 8.0, 5.0, 0.1)
with col2:
    sepal_width = st.number_input("Sepal Width (cm)", 2.0, 5.0, 3.5, 0.1)

# ------------------ Prediction ------------------ #
if st.button("🔍 Classify Flower"):
    if perceptron is None or svm is None:
        st.error("Models not loaded.")
    else:
        X_input = np.array([[sepal_length, sepal_width]])
        X_input_norm = (X_input - mean) / std

        pred_p = perceptron.predict(X_input_norm)[0]
        pred_s = svm.predict(X_input_norm)[0]

        conf_p = abs(np.dot(X_input_norm, perceptron.weights) + perceptron.bias)[0]
        conf_s = abs(np.dot(X_input_norm, svm.weights) + svm.bias)[0]

        classes = {0: "Setosa 🌼", 1: "Versicolor 🌹"}

        st.success(f"**Perceptron Prediction:** {classes[pred_p]} (Confidence: {conf_p:.2f})")
        st.success(f"**SVM Prediction:** {classes[pred_s]} (Confidence: {conf_s:.2f})")

# ------------------ Data Visualization ------------------ #
st.header("📊 Dataset Visualization")

if st.checkbox("Show Scatter Plot (Sepal Length vs Width)"):
    fig, ax = plt.subplots()
    ax.scatter(X_norm[y == 0, 0], X_norm[y == 0, 1], color="blue", label="Setosa")
    ax.scatter(X_norm[y == 1, 0], X_norm[y == 1, 1], color="red", label="Versicolor")
    ax.set_xlabel("Sepal Length (normalized)")
    ax.set_ylabel("Sepal Width (normalized)")
    ax.legend()
    st.pyplot(fig)

