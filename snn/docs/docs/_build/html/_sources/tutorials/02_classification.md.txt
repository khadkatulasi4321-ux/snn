# Tutorial 2 — Classification

Learn how to build a classifier for multi-class problems. We will use synthetic
data, train/test splits, batch normalisation, dropout, and callbacks.

---

## Binary classification

For problems with two classes (spam/not-spam, pass/fail, …) the output is a
single probability and the loss is **binary crossentropy**.

```python
import numpy as np
from snn.model import Sequential
from snn.layers import Dense, Dropout, BatchNormalization
from snn.utils import train_test_split, standardize

rng = np.random.default_rng(42)

# Two blobs of points
X0 = rng.normal(loc=[-2, -2], scale=1.0, size=(300, 2))
X1 = rng.normal(loc=[ 2,  2], scale=1.0, size=(300, 2))
X  = np.vstack([X0, X1])
y  = np.array([[0]] * 300 + [[1]] * 300, dtype=np.float64)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train = standardize(X_train)   # zero-mean, unit variance
X_test  = standardize(X_test)

model = Sequential([
    Dense(32, activation="relu"),
    Dropout(0.2),
    Dense(16, activation="relu"),
    Dense(1,  activation="sigmoid"),
])

model.compile("adam", "binary_crossentropy", metrics=["binary_accuracy"])
model.fit(X_train, y_train, epochs=30, batch_size=32,
          validation_data=(X_test, y_test))

model.evaluate(X_test, y_test)
```

---

## Multi-class classification

For problems with 3+ classes encode targets as **one-hot vectors** and use
**categorical crossentropy**.

### One-hot encoding

```python
from snn.utils import to_categorical

y_int = np.array([0, 1, 2, 0, 2, 1])     # integer class labels
y_hot = to_categorical(y_int, num_classes=3)
# [[1,0,0], [0,1,0], [0,0,1], [1,0,0], [0,0,1], [0,1,0]]
```

### Full example

```python
import numpy as np
from snn.model import Sequential
from snn.layers import Dense, Dropout, BatchNormalization
from snn.utils import to_categorical, train_test_split, standardize

rng = np.random.default_rng(0)

# 4-class synthetic data
X = rng.normal(size=(800, 6))
y_int = (
    (X[:, 0] + X[:, 1] > 0).astype(int) * 2
    + (X[:, 2] > 0).astype(int)
)                                         # 4 classes: 0-3
y = to_categorical(y_int, num_classes=4)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_test = standardize(X_train), standardize(X_test)

model = Sequential([
    Dense(64, activation="relu"),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation="relu"),
    Dense(4,  activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["categorical_accuracy"],
    learning_rate=3e-4,
    verbose=True,        # print compile summary
)

model.fit(X_train, y_train, epochs=40, batch_size=32,
          validation_data=(X_test, y_test))

results = model.evaluate(X_test, y_test)
print(f"\nTest accuracy: {results['categorical_accuracy']:.4f}")
```

---

## Integer targets (no one-hot needed)

If your targets are plain integers (0, 1, 2, …) you can use
**sparse categorical crossentropy** — no one-hot encoding required.

```python
model.compile(
    "adam",
    "sparse_categorical_crossentropy",
    metrics=["categorical_accuracy"],
)

# y_train stays as shape (N,) integers — no to_categorical
model.fit(X_train, y_train_int, epochs=40, batch_size=32)
```

---

## Early stopping

Stop training automatically when validation loss stops improving:

```python
from snn.utils import EarlyStopping

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,          # stop if no improvement for 10 epochs
    min_delta=1e-4,
    restore_best_weights=False,
    verbose=1,
)

model.fit(
    X_train, y_train,
    epochs=200,
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
)
```

---

## From-logits mode

If your last Dense layer has **no activation** (outputs raw scores), pass
`from_logits=True` so the loss applies softmax internally for numerical
stability:

```python
model = Sequential([
    Dense(64, activation="relu"),
    Dense(4),              # no activation — raw logits
])

model.compile(
    "adam",
    "categorical_crossentropy",
    from_logits=True,      # ← loss handles the softmax
)
```

---

## Precision, recall, and F1

```python
from snn.metrics import precision, recall, f1_score, confusion_matrix

y_pred = model.predict(X_test).argmax(axis=1)
y_true = y_test.argmax(axis=1)

print("Precision:", precision(model.predict(X_test), y_test))
print("Recall:   ", recall   (model.predict(X_test), y_test))
print("F1:       ", f1_score (model.predict(X_test), y_test))
print(confusion_matrix(y_pred, y_true))
```

---

## What next?

- **[Tutorial 3](03_regression.md)** — regression and non-linear function fitting.
- **[Tutorial 4](04_activations.md)** — deep dive into choosing the right activation.
