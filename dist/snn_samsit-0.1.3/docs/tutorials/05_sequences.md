# Tutorial 5 — Sequences with LSTM and GRU

Use recurrent layers (LSTM and GRU) when your data has a **temporal or
sequential structure** — time series, text, sensor readings, or any problem
where the order of inputs matters.

---

## When to use a recurrent layer

| Problem | Recommended layer |
|---|---|
| Short sequences, simple patterns | `SimpleRNN` |
| Long sequences, long-range dependencies | `LSTM` |
| Long sequences, want lighter compute | `GRU` |
| Very long sequences | Stack 2 LSTM/GRU layers |

---

## Input shape convention

Recurrent layers expect input of shape `(batch, time_steps, features)`.

```python
# 100 samples, 20 time steps, 3 features per step
X = np.random.randn(100, 20, 3)
```

---

## LSTM — Long Short-Term Memory

```python
from snn.model import Sequential
from snn.layers import LSTM, Dense
import numpy as np

# Sequence → value regression
# Predict the sum of a random sequence of length 10
rng = np.random.default_rng(0)
N = 2000
X = rng.uniform(-1, 1, (N, 10, 1))
y = X.sum(axis=1)               # shape (N, 1)

model = Sequential([
    LSTM(32, return_sequences=False),   # reads sequence, outputs last state
    Dense(16, activation="relu"),
    Dense(1),
])

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["r2"],
    learning_rate=1e-3,
)

model.fit(X, y, epochs=30, batch_size=64,
          validation_data=(X[-200:], y[-200:]))

model.evaluate(X[-200:], y[-200:])
```

Expected: R² > 0.99 within 20-30 epochs.

---

## GRU — Gated Recurrent Unit

GRU is simpler than LSTM (fewer gates, fewer parameters) but often performs
comparably. Use it when training speed matters.

```python
from snn.layers import GRU

model = Sequential([
    GRU(32, return_sequences=False),
    Dense(1),
])
```

---

## Stacked recurrent layers

For harder tasks, stack two recurrent layers. The first must output a full
sequence (`return_sequences=True`) so the second has a sequence to read:

```python
model = Sequential([
    LSTM(64, return_sequences=True),    # outputs (batch, time, 64)
    LSTM(32, return_sequences=False),   # reads that, outputs (batch, 32)
    Dense(16, activation="relu"),
    Dense(1),
])
```

---

## Sequence classification

Classify whether a sequence belongs to class 0 or 1:

```python
rng = np.random.default_rng(1)

# Class 0: random noise. Class 1: random noise + linear trend.
X0 = rng.normal(0, 1, (500, 20, 1))
X1 = rng.normal(0, 1, (500, 20, 1)) + np.linspace(0, 1, 20).reshape(1, 20, 1)
X  = np.vstack([X0, X1])
y  = np.array([[0]] * 500 + [[1]] * 500, dtype=np.float64)

model = Sequential([
    GRU(32, return_sequences=False),
    Dense(16, activation="relu"),
    Dense(1,  activation="sigmoid"),
])

model.compile("adam", "binary_crossentropy", metrics=["binary_accuracy"])
model.fit(X, y, epochs=20, batch_size=64)
```

---

## Many-to-many: sequence-to-sequence

Output a prediction at every time step by setting `return_sequences=True`
in the last recurrent layer too, then applying a Dense layer:

```python
model = Sequential([
    LSTM(32, return_sequences=True),   # outputs (batch, time, 32)
    Dense(1),                          # applied at each time step → (batch, time, 1)
])

# targets must also be (batch, time, 1)
y_seq = X.cumsum(axis=1)
model.compile("adam", "mse")
model.fit(X, y_seq, epochs=20)
```

---

## SimpleRNN — the basics

For very short sequences or educational purposes:

```python
from snn.layers import RNN

model = Sequential([
    RNN(16, return_sequences=False),
    Dense(1),
])
```

SimpleRNN has a vanishing gradient problem on sequences longer than ~10-20
steps. Use LSTM or GRU for anything longer.

---

## Notes on training recurrent networks

- **Gradient clipping is recommended** — recurrent gradients can explode.

  ```python
  model.compile("adam", "mse", clip_norm=1.0)
  ```

- **Scale your inputs** — standardise features before feeding to LSTM/GRU.

- **Dropout in recurrent layers** — use a `Dropout` layer after the
  recurrent layer, not inside it.

  ```python
  Sequential([
      LSTM(64, return_sequences=False),
      Dropout(0.3),
      Dense(1),
  ])
  ```

- **Learning rate** — recurrent nets often benefit from a lower learning rate
  (`1e-3` to `3e-4`).

---

## What next?

- **[Tutorial 6](06_trainer.md)** — gradient accumulation, mixed precision,
  and checkpointing with the Trainer class.
