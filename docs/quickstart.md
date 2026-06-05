# Quickstart

Five minutes from zero to a trained classifier.

---

## 1 — Install

```bash
pip install -e snn/
```

The only runtime dependency is **NumPy ≥ 1.24**.

---

## 2 — XOR (the "hello world" of neural nets)

```python
import numpy as np
from snn.model import Sequential
from snn.layers import Dense

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
y = np.array([[0], [1], [1], [0]], dtype=np.float64)

model = Sequential([
    Dense(8, activation="tanh"),
    Dense(4, activation="tanh"),
    Dense(1, activation="sigmoid"),
])

model.compile(optimizer="adam", loss="binary_crossentropy",
              metrics=["binary_accuracy"])

model.fit(X, y, epochs=1000, batch_size=4, verbose=0)

preds = model.predict(X)
print(preds.round(2))
# [[0.01], [0.99], [0.99], [0.01]]
```

---

## 3 — Multi-class classification

```python
from snn.utils import to_categorical, train_test_split, standardize

# --- create data ---
rng = np.random.default_rng(0)
X = rng.normal(size=(600, 4))
y_int = (X[:, 0] + X[:, 1] > 0).astype(int) * 2 + (X[:, 2] > 0).astype(int)
y = to_categorical(y_int, num_classes=4)   # one-hot

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train = standardize(X_train)
X_test  = standardize(X_test)

# --- build model ---
from snn.layers import BatchNormalization, Dropout

model = Sequential([
    Dense(64, activation="relu"),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation="relu"),
    Dense(4, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["categorical_accuracy"],
)

model.fit(
    X_train, y_train,
    epochs=40,
    batch_size=32,
    validation_data=(X_test, y_test),
)

model.evaluate(X_test, y_test)
```

---

## 4 — Regression

```python
from snn.optimizers import Adam

X = rng.uniform(-np.pi, np.pi, (500, 1))
y = np.sin(X) + rng.normal(0, 0.1, (500, 1))

model = Sequential([
    Dense(64, activation="tanh"),
    Dense(64, activation="tanh"),
    Dense(1),
])

model.compile(optimizer=Adam(learning_rate=1e-3), loss="mse",
              metrics=["r2_score"])

model.fit(X, y, epochs=200, batch_size=32)
```

---

## 5 — Gradient accumulation & checkpointing (Trainer)

For larger models or limited memory:

```python
from snn.trainer import Trainer, Checkpoint

model.compile(optimizer="adam", loss="categorical_crossentropy",
              metrics=["categorical_accuracy"])

ckpt_loss = Checkpoint(monitor="val_loss",     mode="min",  save_path="best_loss")
ckpt_acc  = Checkpoint(monitor="val_categorical_accuracy", mode="max",  save_path="best_acc")

trainer = Trainer(
    model,
    gradient_accumulation_steps=4,   # effective batch = 32 × 4 = 128
    mixed_precision=True,             # simulate FP16
    clip_grad_norm=1.0,
    checkpoints=[ckpt_loss, ckpt_acc],
)

history = trainer.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
)

# Restore the checkpoint with the best validation accuracy
ckpt_acc.restore(model)
model.evaluate(X_test, y_test)
```

---

## 6 — Save & load weights

```python
model.save_weights("my_model")    # writes my_model.npz
model.load_weights("my_model")    # reloads all layer parameters
```

---

## 7 — Convolutional network

```python
from snn.layers import Conv2D, MaxPooling2D, Flatten

cnn = Sequential([
    Conv2D(16, kernel_size=3, padding="same", activation="relu"),
    MaxPooling2D(pool_size=2),
    Conv2D(32, kernel_size=3, padding="same", activation="relu"),
    MaxPooling2D(pool_size=2),
    Flatten(),
    Dense(64, activation="relu"),
    Dropout(0.4),
    Dense(10, activation="softmax"),
])

cnn.compile(optimizer="adam", loss="categorical_crossentropy",
            metrics=["accuracy"])

# input shape: (N, H, W, C)  — channels-last
cnn.fit(X_images, y_labels, epochs=20, batch_size=64)
```

---

## 8 — Sequence model (LSTM)

```python
from snn.layers import LSTM

rnn = Sequential([
    LSTM(64, return_sequences=True),
    LSTM(32),
    Dense(1, activation="sigmoid"),
])

rnn.compile(optimizer="adam", loss="binary_crossentropy",
            metrics=["binary_accuracy"])

# input shape: (N, T, features)
rnn.fit(X_seq, y_seq, epochs=30, batch_size=32)
```
