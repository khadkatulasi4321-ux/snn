# Tutorial 3 — Regression

Regression means predicting a **continuous number** rather than a class label.
This tutorial covers linear regression, polynomial regression, and learning
the non-linear function y = x².

---

## When to use regression

| Problem | Loss | Output activation |
|---|---|---|
| Predict a price, age, or score | `"mse"` | None (Linear) |
| Predict when outliers matter less | `"huber"` | None |
| Predict a bounded value in (0, 1) | `"mse"` | `"sigmoid"` |

---

## Linear regression — temperature prediction

```python
import numpy as np
from snn.model import Sequential
from snn.layers import Dense

rng = np.random.default_rng(1)

# y = 3x + 7 + noise
X = rng.uniform(-5, 5, size=(500, 1))
y = 3 * X + 7 + rng.normal(0, 0.5, size=(500, 1))

model = Sequential([
    Dense(1),     # a single neuron with no activation is linear regression
])

model.compile("sgd", "mse", learning_rate=0.01)
model.fit(X, y, epochs=100, batch_size=32, verbose=0)

# should be very close to W≈3, b≈7
W = model._layers[0].W
b = model._layers[0].b
print(f"Learned: y = {W[0,0]:.3f}x + {b[0,0]:.3f}")
```

---

## Polynomial regression — y = x²

A single linear layer cannot fit a curve. You need hidden layers **with
non-linear activations**. Here we compare several activations on y = x²:

```python
import numpy as np
from snn.model import Sequential
from snn.layers import Dense
from snn.utils import train_test_split

rng = np.random.default_rng(42)
X = np.linspace(-3, 3, 600).reshape(-1, 1)
y = X ** 2    # perfect quadratic, no noise

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

def build(activation):
    m = Sequential([
        Dense(64, activation=activation),
        Dense(64, activation=activation),
        Dense(32, activation=activation),
        Dense(1),
    ])
    m.compile("adam", "mse", metrics=["r2"], learning_rate=3e-3)
    m.fit(X_train, y_train, epochs=500, batch_size=64, verbose=0)
    return m.evaluate(X_test, y_test, verbose=0)

for act in ["relu", "gelu", "sine", "bent_identity", "squareplus"]:
    res = build(act)
    print(f"  {act:<16}  MSE={res['loss']:.5f}  R²={res['r2']:.4f}")
```

Expected results:

```
  relu              MSE=0.00012  R²=1.0000
  gelu              MSE=0.00025  R²=1.0000
  sine              MSE=0.03142  R²=0.9952
  bent_identity     MSE=0.00186  R²=0.9997
  squareplus        MSE=0.00014  R²=1.0000
```

All activations achieve R² ≥ 0.995.

---

## Huber loss — robust to outliers

MSE penalises large errors with x². If your dataset has outliers, the Huber
loss is more robust:

```python
model.compile(
    "adam",
    {"name": "huber", "delta": 1.5},   # dict config — set delta directly
    metrics=["r2"],
)
```

`delta` controls the crossover point:
- Errors smaller than `delta` → behaves like MSE (quadratic)
- Errors larger than `delta` → behaves like MAE (linear)

A large `delta` (e.g. 5.0) is close to MSE; a small `delta` (e.g. 0.1) is
close to MAE.

---

## Multiple outputs

Predict several values at once by setting the output layer's units > 1:

```python
# Predict (sin(x), cos(x)) simultaneously
X = np.linspace(0, 2*np.pi, 500).reshape(-1, 1)
y = np.column_stack([np.sin(X), np.cos(X)])  # shape (500, 2)

model = Sequential([
    Dense(64, activation="gelu"),
    Dense(64, activation="gelu"),
    Dense(2),           # two outputs
])

model.compile("adam", "mse", metrics=["r2"])
model.fit(X, y, epochs=300, batch_size=32)
```

---

## Gradient clipping — when training is unstable

If your loss explodes during training, gradient clipping limits how large
any single update can be:

```python
model.compile(
    "adam", "mse",
    clip_norm=1.0,     # clip global gradient norm to 1.0
)
```

This is equivalent to `Trainer(model, clip_grad_norm=1.0)` but requires no
extra imports.

---

## Evaluating regression

| Metric | String key | Good value |
|---|---|---|
| Mean Squared Error | `"mse"` / `"mean_squared_error"` | Near 0 |
| Mean Absolute Error | `"mae"` / `"mean_absolute_error"` | Near 0 |
| R² (coefficient of determination) | `"r2"` | Near 1.0 |

```python
model.compile("adam", "mse", metrics=["r2", "mae"])
```

---

## What next?

- **[Tutorial 4](04_activations.md)** — detailed guide to every activation,
  when to use each one, and how they affect learning.
- **[Tutorial 5](05_sequences.md)** — LSTM and GRU for time-series data.
