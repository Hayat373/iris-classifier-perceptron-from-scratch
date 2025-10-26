import numpy as np
import pandas as pd
from typing import Tuple

def load_csv_data(path: str = 'data/iris.csv', label_col: int = -1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load Iris data from CSV file. Raises error if file not found.
    Automatically encodes string labels to integers.
    """
    df = pd.read_csv(path)  # No fallback; file must exist
    X = df.drop(df.columns[label_col], axis=1).values
    y = df.iloc[:, label_col].values

    # Encode labels
    if len(y) > 0 and isinstance(y[0], str):
        classes, y_encoded = np.unique(y, return_inverse=True)
        print(f"✅ Encoded labels: {dict(zip(classes, range(len(classes))))}")
        y = y_encoded

    return X.astype(float), y.astype(float)

def train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42):
    """
    Split arrays or matrices into random train and test subsets.
    """
    np.random.seed(random_state)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    test_size = int(test_size * len(X))
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    return X_train, X_test, y_train, y_test