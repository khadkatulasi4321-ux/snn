"""
nonlinear_regression.py
=======================
Demonstrates how snn can learn non-linear functions — specifically y = x² —
using different activation functions, comparing their fit quality.

Run:
    python3 snn/examples/nonlinear_regression.py
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from snn.model import Sequential
from snn.layers import Dense
from snn.optimizers import Adam
from snn.utils import train_test_split

np.random.seed(42)

# ── Dataset: y = x² ──────────────────────────────────────────────────────────
X = np.linspace(-3.0, 3.0, 600).reshape(-1, 1)
y = X ** 2  # perfect quadratic, no noise

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


def build_and_train(activation, epochs=600, lr=3e-3):
    model = Sequential([
        Dense(64, activation=activation),
        Dense(64, activation=activation),
        Dense(32, activation=activation),
        Dense(1),
    ])
    model.compile(optimizer=Adam(learning_rate=lr), loss="mse", metrics=["r2"])
    model.fit(X_train, y_train, epochs=epochs, batch_size=64,
              validation_data=(X_test, y_test), verbose=0)
    results = model.evaluate(X_test, y_test, verbose=0)
    return model, results


# ── Compare activations on y = x² ────────────────────────────────────────────
activations = ["sine", "gelu", "bent_identity", "squareplus", "swish", "relu"]

print("=" * 58)
print(f"  Learning y = x²  —  test-set performance")
print("=" * 58)
print(f"  {'Activation':<16}  {'MSE':>10}  {'R²':>8}")
print("-" * 58)

best_model = None
best_r2 = -np.inf
best_name = ""

for act_name in activations:
    model, res = build_and_train(act_name)
    mse = res.get("loss", float("nan"))
    r2  = res.get("r2",   float("nan"))
    marker = " ◀ best" if r2 > best_r2 else ""
    # update best (done after printing so marker assignment is correct below)
    if r2 > best_r2:
        best_r2 = r2
        best_model = model
        best_name = act_name
    print(f"  {act_name:<16}  {mse:>10.6f}  {r2:>8.4f}")

print("=" * 58)
print(f"  Best: {best_name}  (R² = {best_r2:.4f})")

# ── Spot-check predictions with best model ───────────────────────────────────
print(f"\n  Spot-check ({best_name}):")
print(f"  {'x':>6}  {'y=x²':>8}  {'predicted':>12}  {'error':>8}")
print("  " + "-" * 40)
test_pts = np.array([[-3.0], [-2.0], [-1.0], [0.0], [1.0], [2.0], [3.0]])
preds = best_model.predict(test_pts)
for xi, yi_true, yi_pred in zip(test_pts.flatten(), (test_pts**2).flatten(), preds.flatten()):
    err = abs(yi_true - yi_pred)
    print(f"  {xi:>6.1f}  {yi_true:>8.4f}  {yi_pred:>12.4f}  {err:>8.4f}")
