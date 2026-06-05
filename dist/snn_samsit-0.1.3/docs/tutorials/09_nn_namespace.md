# Tutorial 9 — snn.nn: Everything in One Place

`snn.nn` is a single flat namespace that collects every building block in the
library — layers, activations, losses, optimizers, utilities, and models —
so you can import what you need from one place instead of remembering which
sub-module it lives in.

It is also the home of **NNFS-compatible aliases**: the `Layer_Dense`,
`Activation_ReLU`, `Loss_CCE`, `Optimizer_Adam` naming style from the
*Neural Networks from Scratch* book works out of the box.

---

## 1 — Flat-namespace style

The cleanest way to use snn for most projects:

```python
from snn.nn import (
    Dense, BatchNormalization, Dropout,
    ReLU, Softmax,
    Adam, CCE,
    Sequential, to_categorical,
)
import numpy as np

X = np.random.randn(500, 20)
y = to_categorical(np.random.randint(0, 5, 500), num_classes=5)

model = Sequential([
    Dense(64),
    BatchNormalization(),
    ReLU(),
    Dropout(0.3),
    Dense(5),
    Softmax(),
])
model.compile(Adam(1e-3), CCE(), metrics=["accuracy"])
history = model.fit(X, y, epochs=10, batch_size=32, verbose=1)
```

---

## 2 — NNFS-style manual training loop

If you learned from *Neural Networks from Scratch* (Kinsley & Kukiela), every
class you know has a drop-in alias:

| NNFS class name | snn equivalent |
|---|---|
| `Layer_Dense` | `Dense` |
| `Activation_ReLU` | `ReLU` |
| `Activation_Softmax` | `Softmax` |
| `Activation_Sigmoid` | `Sigmoid` |
| `Activation_Linear` | `Linear` |
| `Loss_CategoricalCrossentropy` | `Loss_CCE` → `CCE` |
| `Loss_BinaryCrossentropy` | `Loss_BCE` → `BCE` |
| `Loss_MeanSquaredError` | `Loss_MSE` → `MSE` |
| `Optimizer_Adam` | `Adam` |
| `Optimizer_SGD` | `SGD` |
| `Optimizer_RMSprop` | `RMSprop` |
| `Optimizer_Adagrad` | `Adagrad` |
| `Layer_Dropout` | `Dropout` |

```python
from snn.nn import (
    Layer_Dense, Layer_Dropout,
    Activation_ReLU, Activation_Softmax,
    Loss_CCE,
    Optimizer_Adam,
    to_categorical,
)
import numpy as np

# Data
X = np.random.randn(300, 10)
y = to_categorical(np.random.randint(0, 3, 300), 3)

# Build layers exactly as in the NNFS book
dense1   = Layer_Dense(10, 64)
relu     = Activation_ReLU()
dropout  = Layer_Dropout(0.2)
dense2   = Layer_Dense(64, 3)
softmax  = Activation_Softmax()
loss_fn  = Loss_CCE()
opt      = Optimizer_Adam(learning_rate=0.001, decay=1e-4)

# Manual training loop (NNFS style)
for epoch in range(201):
    # ── Forward ──────────────────────────────────────────────
    out = dense1.forward(X, training=True)
    out = relu.forward(out)
    out = dropout.forward(out, training=True)
    out = dense2.forward(out, training=True)
    out = softmax.forward(out)

    loss = loss_fn.forward(out, y)

    # ── Backward ─────────────────────────────────────────────
    grad = loss_fn.backward(out, y)
    grad = softmax.backward(grad)
    grad = dense2.backward(grad)
    grad = dropout.backward(grad)
    grad = relu.backward(grad)
    dense1.backward(grad)

    # ── Update ───────────────────────────────────────────────
    opt.update(dense1)
    opt.update(dense2)

    if epoch % 50 == 0:
        preds = np.argmax(out, axis=1)
        acc   = np.mean(preds == np.argmax(y, axis=1))
        print(f"Epoch {epoch:3d} | loss {loss:.4f} | acc {acc:.3f}")
```

---

## 3 — All 24 activations as classes

Unlike string-based activation parameters, `snn.nn` lets you use every
activation as a **standalone layer** in a `Sequential` or `GraphModel`:

```python
from snn.nn import (
    Dense, Sequential,
    ReLU, LeakyReLU, ELU, SELU, GELU,
    Swish, Mish, PReLU,
    Sigmoid, Tanh, Softmax, Softplus,
    CELU, Softsign, Tanhshrink,
    Hardswish, ReLU6, Hardsigmoid,
    LogSoftmax, Sparsemax,
    BentIdentity, Squareplus, Sine, Linear,
)

# Any of these works the same way
model = Sequential([
    Dense(64),
    Mish(),          # standalone activation layer
    Dense(32),
    GELU(),
    Dense(10),
    Softmax(),
])
```

---

## 4 — Imports cheat-sheet

```python
# ── Everything at once ────────────────────────────────────────────
from snn import nn  # access as nn.Dense, nn.ReLU, nn.Adam, …

# ── Or cherry-pick ────────────────────────────────────────────────
from snn.nn import Dense, LSTM, Conv2D                    # layers
from snn.nn import ReLU, Swish, Mish, GELU, Softmax      # activations
from snn.nn import Adam, AdamW, LAMB, Lookahead, Adan     # optimizers
from snn.nn import MSE, MAE, BCE, CCE, Huber              # losses (short)
from snn.nn import Sequential, Model, GraphModel, Input   # models
from snn.nn import EarlyStopping, ReduceLROnPlateau       # utilities
from snn.nn import Trainer, Checkpoint                    # trainer
from snn.nn import to_categorical, train_test_split       # data utils

# ── NNFS aliases (all available) ─────────────────────────────────
from snn.nn import (
    Layer_Dense, Layer_Dropout, Layer_LSTM, Layer_Conv2D,
    Activation_ReLU, Activation_Sigmoid, Activation_Softmax,
    Loss_MSE, Loss_BCE, Loss_CCE,
    Optimizer_SGD, Optimizer_Adam, Optimizer_RMSprop,
)
```

---

## 5 — Serving the docs

After installing the package, serve the full docs locally with one command:

```bash
python -m snn.docs.serve_docs       # serves on port 5000
PORT=8080 python -m snn.docs.serve_docs

# Or use the console script (if installed via setup.py / pip):
snn-docs
```
