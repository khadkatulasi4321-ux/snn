"""
XOR problem — the classic non-linearly-separable benchmark.

A 2-input, 1-output network learning the XOR truth table.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from snn.model import Sequential
from snn.layers import Dense


def main():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
    y = np.array([[0], [1], [1], [0]], dtype=np.float64)

    model = Sequential([
        Dense(8, activation="tanh"),
        Dense(4, activation="tanh"),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["binary_accuracy"])

    print("Training XOR network...")
    model.fit(X, y, epochs=1000, batch_size=4, verbose=0)

    preds = model.predict(X)
    print("\nXOR predictions:")
    for xi, yi, pi in zip(X, y.flatten(), preds.flatten()):
        print(f"  {int(xi[0])} XOR {int(xi[1])} = {int(yi)}  →  predicted: {pi:.4f}  (rounded: {round(pi)})")

    results = model.evaluate(X, y)
    print(f"\nFinal — loss: {results['loss']:.4f}, accuracy: {results.get('binary_accuracy', 'n/a'):.4f}")


if __name__ == "__main__":
    main()
