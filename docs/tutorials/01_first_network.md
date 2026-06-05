# Tutorial 1 — Your First Network

This tutorial builds a neural network from scratch, step by step. No
assumptions about prior ML experience.

---

## What is a neural network?

A neural network is a function that learns to map inputs to outputs by
adjusting its internal parameters (called **weights**) based on examples.

It is made of **layers**. Each layer takes a batch of numbers in and produces
a batch of numbers out. Stack several layers together and you have a network.

```
Input → [Layer 1] → [Layer 2] → [Layer 3] → Output
         (learn)     (learn)     (learn)
```

---

## The problem: XOR

XOR is the classic "hello world" for neural nets. It cannot be solved by
a single straight line (it is *not* linearly separable), so a network needs
at least one hidden layer.

| x₁ | x₂ | x₁ XOR x₂ |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

---

## Step 1 — Create the data

```python
import numpy as np

X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]], dtype=np.float64)

y = np.array([[0], [1], [1], [0]], dtype=np.float64)
```

`X` has shape `(4, 2)` — four samples, two features each.
`y` has shape `(4, 1)` — one binary target per sample.

---

## Step 2 — Build the model

```python
from snn.model import Sequential
from snn.layers import Dense

model = Sequential([
    Dense(8, activation="tanh"),   # hidden layer: 2 → 8 neurons
    Dense(4, activation="tanh"),   # hidden layer: 8 → 4 neurons
    Dense(1, activation="sigmoid"),# output: probability in (0, 1)
])
```

**Dense layer** — the most basic layer. Every input neuron connects to every
output neuron. The computation is:

```
output = activation(input @ W + b)
```

where `W` is a weight matrix and `b` is a bias vector, both learned during
training.

**Activation function** — applies a non-linear transformation element-wise.
Without activations, stacking Dense layers is identical to a single layer
(pure matrix multiplication is linear). The activations are what let the
network learn curved, non-linear patterns.

- `"tanh"` → output in (−1, 1), smooth, good for hidden layers
- `"sigmoid"` → output in (0, 1), good for binary output probabilities

---

## Step 3 — Compile

```python
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["binary_accuracy"],
)
```

**Optimizer** — the algorithm that adjusts weights after each batch.
Adam is a safe default: it adapts the learning rate per parameter.

**Loss** — the function that measures how wrong the predictions are.
Binary crossentropy is the standard loss for binary (0/1) problems.

**Metrics** — extra things to track during training (not used to update
weights, just for monitoring).

---

## Step 4 — Train

```python
history = model.fit(
    X, y,
    epochs=1000,
    batch_size=4,  # all 4 samples at once (dataset is tiny)
    verbose=0,     # silent — we'll print manually
)

print(f"Final loss:     {history['loss'][-1]:.4f}")
print(f"Final accuracy: {history['binary_accuracy'][-1]:.4f}")
```

Each **epoch** is one full pass over the training data:
1. Forward pass — predict outputs from current weights
2. Compute loss
3. Backward pass — compute how much each weight contributed to the loss
4. Optimizer step — nudge every weight in the direction that reduces the loss

After 1000 epochs you should see loss < 0.02 and accuracy = 1.0.

---

## Step 5 — Inspect predictions

```python
preds = model.predict(X)
for xi, yi, pi in zip(X, y.flatten(), preds.flatten()):
    print(f"  {xi}  →  target={int(yi)}  predicted={pi:.4f}  "
          f"(rounded: {int(round(pi))})")
```

Expected output:
```
  [0. 0.]  →  target=0  predicted=0.0089  (rounded: 0)
  [0. 1.]  →  target=1  predicted=0.9921  (rounded: 1)
  [1. 0.]  →  target=1  predicted=0.9935  (rounded: 1)
  [1. 1.]  →  target=0  predicted=0.0143  (rounded: 0)
```

---

## Step 6 — Model summary

```python
model.summary()
```

```
=================================================================
Layer                     Output Shape         Params
=================================================================
dense_0                   ?                            24
dense_1                   ?                            36
dense_2                   ?                             5
=================================================================
Total params: 65
=================================================================
```

Param count for a Dense layer: `(input_size + 1) × output_size`
(the `+ 1` is the bias vector).

---

## Full code

```python
import numpy as np
from snn.model import Sequential
from snn.layers import Dense

# Data
X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float64)
y = np.array([[0],[1],[1],[0]], dtype=np.float64)

# Model
model = Sequential([
    Dense(8, activation="tanh"),
    Dense(4, activation="tanh"),
    Dense(1, activation="sigmoid"),
])

model.compile("adam", "binary_crossentropy", metrics=["binary_accuracy"])
model.fit(X, y, epochs=1000, batch_size=4, verbose=0)

# Results
preds = model.predict(X)
print(preds.round(3))
# [[0.009], [0.992], [0.993], [0.014]]
```

---

## What next?

- **[Tutorial 2](02_classification.md)** — multi-class classification with
  real data, train/test splits, and batch normalisation.
- **[Tutorial 3](03_regression.md)** — predicting continuous values and
  learning non-linear functions like y = x².
