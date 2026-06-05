"""
Synthetic 2-class and multi-class classification examples.

Demonstrates:
  - Dense layers with BatchNorm + Dropout
  - Adam optimizer
  - Categorical cross-entropy loss
  - Accuracy metric
  - train_test_split utility
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from snn.model import Sequential
from snn.layers import Dense, Dropout, BatchNormalization, Flatten
from snn.utils import to_categorical, train_test_split, standardize


def make_moons(n=500, noise=0.15, seed=42):
    rng = np.random.default_rng(seed)
    n_half = n // 2
    theta0 = rng.uniform(0, np.pi, n_half)
    theta1 = rng.uniform(0, np.pi, n_half)
    X0 = np.stack([np.cos(theta0), np.sin(theta0)], axis=1)
    X1 = np.stack([1 - np.cos(theta1), 1 - np.sin(theta1) - 0.5], axis=1)
    X = np.vstack([X0, X1]) + rng.normal(0, noise, (n, 2))
    y = np.array([0] * n_half + [1] * n_half)
    return X.astype(np.float64), y


def make_blobs(n=600, centers=4, std=0.8, seed=0):
    rng = np.random.default_rng(seed)
    center_pts = rng.uniform(-5, 5, (centers, 2))
    X_list, y_list = [], []
    per = n // centers
    for i, c in enumerate(center_pts):
        X_list.append(rng.normal(c, std, (per, 2)))
        y_list.extend([i] * per)
    return np.vstack(X_list).astype(np.float64), np.array(y_list)


def binary_classification():
    print("=" * 55)
    print("Binary Classification (moons dataset)")
    print("=" * 55)

    X, y = make_moons(n=800)
    X = standardize(X)
    y_bin = y.reshape(-1, 1).astype(np.float64)

    X_train, X_test, y_train, y_test = train_test_split(X, y_bin, test_size=0.2, seed=1)

    model = Sequential([
        Dense(32, activation="relu"),
        BatchNormalization(),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["binary_accuracy"])
    model.fit(X_train, y_train, epochs=50, batch_size=32,
              validation_data=(X_test, y_test), verbose=1)

    print("\nTest evaluation:")
    model.evaluate(X_test, y_test)


def multiclass_classification():
    print("\n" + "=" * 55)
    print("Multi-class Classification (blobs dataset, 4 classes)")
    print("=" * 55)

    X, y_int = make_blobs(n=800, centers=4)
    X = standardize(X)
    y = to_categorical(y_int, num_classes=4)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, seed=2)

    model = Sequential([
        Dense(64, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(4, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy",
                  metrics=["categorical_accuracy"])
    model.fit(X_train, y_train, epochs=60, batch_size=32,
              validation_data=(X_test, y_test), verbose=1)

    print("\nTest evaluation:")
    model.evaluate(X_test, y_test)

    from snn.metrics import confusion_matrix
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_pred, y_test, n_classes=4)
    print("\nConfusion matrix:")
    print(cm)


if __name__ == "__main__":
    binary_classification()
    multiclass_classification()
