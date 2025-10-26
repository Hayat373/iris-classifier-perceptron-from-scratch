# Iris Flower Classifier From Scratch

A simple, educational ML project implementing Perceptron and SVM classifiers from scratch (using only NumPy) for binary Iris flower classification (Setosa vs. Versicolor). Includes training scripts, command-line prediction, and a Streamlit web app for practical use in botanical research.

## Features
- **From-scratch models**: No scikit-learn—pure NumPy for Perceptron and hinge-loss SVM.
- **Dataset**: Standard Iris CSV (filtered to 2 classes).
- **Frontend**: Streamlit app for inputting measurements and viewing predictions/plots.


## Setup
1. Clone: `git clone https://github.com/Hayat373/iris-classifier-perceptron-from-scratch.git`
2. Install: `pip install -r requirements.txt`
3. Train: `python main.py` (generates models).
4. Predict: `python predict.py` or `streamlit run app.py`.

## Use Case
Botanists can input sepal/petal measurements via the web app to classify flowers quickly in the field.

## Structure
- `data/`: iris.csv, scaling params.
- `models/`: perceptron.py, svm.py.
- `utils/`: csv_utils.py.
- `app.py`: Streamlit frontend.
- `main.py`/`train.py`: Training.

