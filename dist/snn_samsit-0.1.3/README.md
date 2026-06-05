# snn

A neural network / deep learning library built **purely on NumPy** — no TensorFlow, no PyTorch, no autograd. Every forward pass, backward pass, and weight update is implemented from scratch in vectorized NumPy.

---

## Install

```bash
pip install snn-samsit 
```

Or just drop the `snn/` folder next to your script and `import snn`.

---

## Quickstart

```python
from snn.model import Sequential
from snn.layers import Dense, Dropout, BatchNormalization
from snn.utils import to_categorical, train_test_split

# Build a model
model = Sequential([
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax'),
])

# Compile
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

# Train
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=64,
    validation_data=(X_val, y_val),
)

# Evaluate
model.evaluate(X_test, y_test)

# Predict
y_pred = model.predict(X_test)

# Save / load weights
model.save_weights('my_weights')
model.load_weights('my_weights')
```

---

## What's Included

### Layers

| Layer | Description |
|---|---|
| `Dense` | Fully-connected layer |
| `Conv2D` | 2D convolution (channels-last) |
| `MaxPooling2D` | Max pooling |
| `AveragePooling2D` | Average pooling |
| `GlobalAveragePooling2D` | Global average pool |
| `GlobalMaxPooling2D` | Global max pool |
| `BatchNormalization` | Batch norm (train/inference modes) |
| `LayerNormalization` | Layer norm |
| `Dropout` | Inverted dropout |
| `SpatialDropout2D` | Channel-wise dropout for feature maps |
| `Flatten` | Flatten to (N, -1) |
| `Reshape` | Reshape (excluding batch dim) |
| `SimpleRNN` | Vanilla RNN with BPTT |
| `LSTM` | Long Short-Term Memory |
| `GRU` | Gated Recurrent Unit |

### Activations

`linear`, `relu`, `leaky_relu`, `elu`, `selu`, `sigmoid`, `tanh`, `softmax`, `softplus`, `swish`, `mish`

### Loss Functions

| Loss | String key |
|---|---|
| Mean Squared Error | `'mse'` |
| Mean Absolute Error | `'mae'` |
| Huber | `'huber'` |
| Binary Cross-entropy | `'binary_crossentropy'` |
| Categorical Cross-entropy | `'categorical_crossentropy'` |
| Sparse Categorical CE | `'sparse_categorical_crossentropy'` |
| KL Divergence | `'kl_divergence'` |

### Optimizers

| Optimizer | String key |
|---|---|
| SGD (+ momentum + Nesterov) | `'sgd'` |
| Adam | `'adam'` |
| AdamW | `'adamw'` |
| RMSprop | `'rmsprop'` |
| Adagrad | `'adagrad'` |
| Adadelta | `'adadelta'` |

### Initializers

`zeros`, `ones`, `random_normal`, `random_uniform`, `glorot_uniform`, `glorot_normal`, `he_uniform`, `he_normal`, `lecun_uniform`, `lecun_normal`

### Metrics

`accuracy`, `binary_accuracy`, `categorical_accuracy`, `sparse_categorical_accuracy`, `top_k_accuracy`, `precision`, `recall`, `f1_score`, `mse`, `mae`, `r2_score`, `confusion_matrix`

### Utilities

```python
from snn.utils import (
    to_categorical,        # integer labels → one-hot
    normalize,             # min-max [0,1]
    standardize,           # zero mean, unit variance
    train_test_split,      # split arrays into train/test
    batch_generator,       # mini-batch iterator
    clip_gradients,        # global gradient norm clipping
    learning_rate_schedule,# exponential decay schedule
    cosine_annealing,      # cosine annealing schedule
    warmup_schedule,       # linear warmup + base schedule
    EarlyStopping,         # stop when metric stagnates
    ReduceLROnPlateau,     # reduce LR when metric plateaus
)
```

---

## Examples

```bash
python examples/xor_example.py          # XOR truth table
python examples/classification_example.py  # moons + blobs datasets
python examples/regression_example.py      # sin(x) + polynomial fitting
```

---

## Architecture

Every layer exposes:
- `forward(x, training=False)` — computes the output
- `backward(grad)` — computes and stores parameter gradients, returns input gradient
- `params` property — dict of trainable arrays
- `grads` property — dict of corresponding gradients

The `Sequential` model wires them together, calls `optimizer.apply_gradients(params, grads)` after each batch, and supports `save_weights` / `load_weights` via `np.savez`.

---

## Design Principles

- **Pure NumPy** — zero deep-learning framework dependencies
- **Vectorized** — all operations work on full batches; no Python loops over samples
- **Keras-like API** — familiar `compile → fit → evaluate → predict` interface
- **Explicit gradients** — every backward pass is hand-derived, making it a great learning resource

## NOTE : to get docs you need to install source file from pypi https://pypi.org/project/snn-samsit/#files
