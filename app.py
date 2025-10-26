import streamlit as st
import numpy as np
from models.perceptron import Perceptron
from models.svm import SVM
from utils.csv_utils import load_csv_data
import matplotlib.pyplot as plt

# Load saved models and scaling
@st.cache_resource
def load_models():
    try:
        mean = np.load("data/mean.npy")
        std = np.load("data/std.npy")
        perceptron = Perceptron()
        perceptron.weights = np.load("models/perceptron_weights.npy")
        perceptron.bias = float(np.load("models/perceptron_bias.npy"))
        svm = SVM()
        svm.weights = np.load("models/svm_weights.npy")
        svm.bias = float(np.load("models/svm_bias.npy"))
    except FileNotFoundError:
        st.error("Model or scaling files not found. Run main.py or train.py first.")
        return None, None, None, None
    return perceptron, svm, mean, std

perceptron, svm, mean, std = load_models()

# Load data for visualization
X, y = load_csv_data()
mask = y < 2
X = X[mask]
y = y[mask]
X_norm = (X - mean) / std

def get_confidence(model, X):
    linear_output = np.dot(X, model.weights) + model.bias
    return np.abs(linear_output)

# Streamlit App
st.title("🌺 Iris Flower Classifier")
st.markdown("""
This app classifies Iris flowers as **Setosa** or **Versicolor** using pre-trained Perceptron and SVM models. 
Enter measurements below for prediction. Built for botanical research use case.
""")

# User inputs
col1, col2 = st.columns(2)
with col1:
    sepal_length = st.number_input("Sepal Length (cm)", min_value=4.0, max_value=8.0, value=5.0, step=0.1)
    petal_length = st.number_input("Petal Length (cm)", min_value=1.0, max_value=7.0, value=1.4, step=0.1)
with col2:
    sepal_width = st.number_input("Sepal Width (cm)", min_value=2.0, max_value=5.0, value=3.5, step=0.1)
    petal_width = st.number_input("Petal Width (cm)", min_value=0.1, max_value=2.5, value=0.2, step=0.1)

if st.button("Classify Flower"):
    if perceptron is None or svm is None:
        st.error("Models not loaded. Ensure training is complete.")
    else:
        X_input = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        X_input_norm = (X_input - mean) / std
        
        pred_p = perceptron.predict(X_input_norm)[0]
        pred_s = svm.predict(X_input_norm)[0]
        conf_p = get_confidence(perceptron, X_input_norm)[0]
        conf_s = get_confidence(svm, X_input_norm)[0]
        
        class_names = {0: "Setosa", 1: "Versicolor"}
        
        st.success(f"**Perceptron Prediction:** {class_names[pred_p]} (Confidence: {conf_p:.2f})")
        st.success(f"**SVM Prediction:** {class_names[pred_s]} (Confidence: {conf_s:.2f})")

# Data Visualization
st.header("Explore the Dataset")
if st.checkbox("Show Scatter Plot (Sepal Length vs. Width)"):
    fig, ax = plt.subplots()
    ax.scatter(X_norm[y==0, 0], X_norm[y==0, 1], label="Setosa", color="blue")
    ax.scatter(X_norm[y==1, 0], X_norm[y==1, 1], label="Versicolor", color="red")
    ax.set_xlabel("Sepal Length (normalized)")
    ax.set_ylabel("Sepal Width (normalized)")
    ax.legend()
    st.pyplot(fig)

if st.checkbox("Show Model Decision Boundary (on first 2 features)"):
    fig, ax = plt.subplots()
    ax.scatter(X_norm[y==0, 0], X_norm[y==0, 1], label="Setosa", color="blue")
    ax.scatter(X_norm[y==1, 0], X_norm[y==1, 1], label="Versicolor", color="red")
    
    x_vals = np.array(ax.get_xlim())
    y_vals_p = -(perceptron.weights[0] * x_vals + perceptron.bias) / perceptron.weights[1]
    ax.plot(x_vals, y_vals_p, label="Perceptron Boundary", color="green")
    
    y_vals_s = -(svm.weights[0] * x_vals + svm.bias) / svm.weights[1]
    ax.plot(x_vals, y_vals_s, label="SVM Boundary", color="purple")
    
    ax.set_xlabel("Sepal Length (normalized)")
    ax.set_ylabel("Sepal Width (normalized)")
    ax.legend()
    st.pyplot(fig)