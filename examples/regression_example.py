"""
Regression examples using snn.

Demonstrates:
  - Dense layers for regression
  - MSE / MAE / Huber losses
  - RMSprop and SGD with momentum
  - R² metric
  - Learning rate scheduling via ReduceLROnPlateau
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from snn.model import Sequential
from snn.layers import Dense, BatchNormalization, Dropout
from snn.optimizers import Adam, RMSprop
from snn.utils import train_test_split, standardize


def make_sine_wave(n=500, noise=0.1, seed=7):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-np.pi, np.pi, (n, 1))
    y = np.sin(X) + rng.normal(0, noise, (n, 1))
    return X.astype(np.float64), y.astype(np.float64)


def make_polynomial(n=500, degree=3, noise=0.5, seed=3):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-3, 3, (n, 1))
    y = 0.5 * X ** 3 - 2.0 * X ** 2 + X + rng.normal(0, noise, (n, 1))
    return X.astype(np.float64), y.astype(np.float64)


def sine_regression():
    print("=" * 55)
    print("Regression: sin(x) approximation")
    print("=" * 55)

    X, y = make_sine_wave(n=600)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, seed=1)

    model = Sequential([
        Dense(64, activation="tanh"),
        Dense(64, activation="tanh"),
        Dense(32, activation="tanh"),
        Dense(1),
    ])
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mse", "r2_score"],
    )
    model.fit(X_train, y_train, epochs=200, batch_size=32,
              validation_data=(X_test, y_test), verbose=1)

    print("\nTest evaluation:")
    model.evaluate(X_test, y_test)


def polynomial_regression():
    print("\n" + "=" * 55)
    print("Regression: polynomial approximation")
    print("=" * 55)

    X, y = make_polynomial(n=600)
    X = standardize(X)
    y = standardize(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, seed=2)

    model = Sequential([
        Dense(128, activation="relu"),
        BatchNormalization(),
        Dense(64, activation="relu"),
        Dropout(0.1),
        Dense(32, activation="relu"),
        Dense(1),
    ])
    model.compile(
        optimizer=RMSprop(learning_rate=1e-3),
        loss="huber",
        metrics=["mae", "r2_score"],
    )
    model.fit(X_train, y_train, epochs=150, batch_size=32,
              validation_data=(X_test, y_test), verbose=1)

    print("\nTest evaluation:")
    model.evaluate(X_test, y_test)


if __name__ == "__main__":
    sine_regression()
    polynomial_regression()
