# Models

## Sequential

`Sequential` is snn's only model class. It holds a linear stack of layers
and wires them together for training and inference.

---

### Building a model

Pass a list of layers to the constructor:

```python
from snn.model import Sequential
from snn.layers import Dense, Dropout

model = Sequential([
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(10, activation="softmax"),
])
```

Or add layers incrementally with `.add()` — it returns `self` so you can chain:

```python
model = (
    Sequential()
    .add(Dense(128, activation="relu"))
    .add(Dropout(0.3))
    .add(Dense(10, activation="softmax"))
)
```

---

### Compiling — `.compile()`

```python
model.compile(
    optimizer,
    loss,
    metrics=None,
    *,
    learning_rate=None,
    weight_decay=None,
    clip_norm=None,
    from_logits=False,
    verbose=False,
)
```

**`optimizer`** — three accepted forms:

| Form | Example |
|---|---|
| String key | `"adam"`, `"sgd"`, `"adamw"`, `"rmsprop"` |
| Optimizer instance | `Adam(learning_rate=3e-4, weight_decay=1e-5)` |
| Config dict | `{"name": "adam", "learning_rate": 1e-3}` |

```python
# string shorthand
model.compile("adam", "mse")

# instance — full control
from snn.optimizers import AdamW
model.compile(AdamW(learning_rate=3e-4, weight_decay=1e-4), "mse")

# dict config — all hyperparams in one place
model.compile(
    {"name": "adamw", "learning_rate": 3e-4, "weight_decay": 1e-4},
    "categorical_crossentropy",
)
```

**`loss`** — three accepted forms:

| Form | Example |
|---|---|
| String key | `"mse"`, `"categorical_crossentropy"`, `"huber"` |
| Loss instance | `HuberLoss(delta=2.0)` |
| Config dict | `{"name": "huber", "delta": 2.0}` |

```python
model.compile("adam", "mse")
model.compile("adam", {"name": "huber", "delta": 1.5})
from snn.losses import CategoricalCrossentropy
model.compile("adam", CategoricalCrossentropy(from_logits=True))
```

**`metrics`** — a string or list of strings / callables:

```python
model.compile("adam", "mse", metrics="r2")                     # single string
model.compile("adam", "mse", metrics=["r2", "mae"])            # list
model.compile("adam", "mse", metrics=["r2", my_custom_metric]) # callable
```

**`learning_rate`** — shortcut to set the optimizer LR without constructing an instance:

```python
model.compile("adam", "mse", learning_rate=3e-4)
# equivalent to: model.compile(Adam(learning_rate=3e-4), "mse")
```

**`weight_decay`** — shortcut for optimizers that support it (Adam, AdamW, SGD):

```python
model.compile("adamw", "mse", learning_rate=1e-3, weight_decay=1e-4)
```

**`clip_norm`** — clip the global gradient L2 norm before each optimizer step.
Prevents gradient explosion, especially useful for recurrent networks:

```python
model.compile("adam", "mse", clip_norm=1.0)
```

**`from_logits`** — when `True`, the last Dense layer should have **no activation**
and the loss (BCE or CCE) applies softmax/sigmoid internally for numerical
stability:

```python
model = Sequential([
    Dense(64, activation="relu"),
    Dense(10),    # no activation
])
model.compile("adam", "categorical_crossentropy", from_logits=True)
```

**`verbose`** — print a compile summary table:

```python
model.compile("adam", "mse", metrics="r2", clip_norm=1.0, verbose=True)
```

```
┌────────────────────────────────────────────────────┐
│                   Compile Summary                  │
├────────────────────────────────────────────────────┤
│  Optimizer          Adam (lr=0.001)                │
│  Loss               MeanSquaredError               │
│  Metrics            r2                             │
│  Clip norm          1.0                            │
└────────────────────────────────────────────────────┘
```

**`get_compile_config()`** — inspect the current compile state:

```python
cfg = model.get_compile_config()
# {
#   'optimizer': 'Adam',
#   'learning_rate': 0.001,
#   'weight_decay': 0.0,
#   'loss': 'MeanSquaredError',
#   'from_logits': False,
#   'metrics': ['r2'],
#   'clip_norm': 1.0,
# }
```

---

### Training — `.fit()`

```python
history = model.fit(
    x,
    y,
    epochs=10,
    batch_size=32,
    validation_data=None,
    shuffle=True,
    verbose=1,
    callbacks=None,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `x` | ndarray | — | Input data, shape `(N, ...)` |
| `y` | ndarray | — | Target data, shape `(N, ...)` |
| `epochs` | int | 10 | Full passes over the data |
| `batch_size` | int | 32 | Samples per gradient update |
| `validation_data` | `(X_val, y_val)` | None | Evaluated each epoch |
| `shuffle` | bool | True | Shuffle training data each epoch |
| `verbose` | int | 1 | `0` = silent, `1` = one line per epoch |
| `callbacks` | list | None | Called as `callback(epoch, history)` at epoch end |

**Return value** — a `dict` with training history:

```python
history["loss"]           # list of training losses (one per epoch)
history["r2"]             # list of training R² values
history["val_loss"]       # list of validation losses
history["val_r2"]         # list of validation R² values
```

**Callbacks** — any callable `(epoch, history) → None`:

```python
from snn.utils import EarlyStopping, ReduceLROnPlateau

early_stop = EarlyStopping(monitor="val_loss", patience=10, verbose=1)
reduce_lr  = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5)

opt = Adam(learning_rate=1e-3)
model.compile(opt, "mse")

model.fit(
    X_train, y_train,
    epochs=300,
    validation_data=(X_val, y_val),
    callbacks=[
        early_stop,
        lambda epoch, hist: reduce_lr(epoch, hist, optimizer=opt),
    ],
)
```

---

### Prediction — `.predict()`

```python
y_pred = model.predict(x)                  # all at once
y_pred = model.predict(x, batch_size=64)  # in batches (saves memory)
```

Dropout and BatchNorm switch to inference mode automatically — no `.eval()`
call needed.

---

### Evaluation — `.evaluate()`

```python
results = model.evaluate(x, y)
# {'loss': 0.023, 'r2': 0.997}

results = model.evaluate(x, y, verbose=0)  # silent
```

---

### Introspection — `.summary()`

```python
model.summary()
```

```
=================================================================
Layer                     Output Shape         Params
=================================================================
dense_0                   ?                        25,728
batchnormalization_1      ?                           256
dropout_2                 ?                             0
dense_3                   ?                         5,130
=================================================================
Total params: 31,114
=================================================================
```

---

### Saving & loading weights

```python
model.save_weights("my_model")     # writes my_model.npz
model.load_weights("my_model")     # restores all layer parameters
```

Weights are stored as a plain NumPy `.npz` keyed by `layer_{i}_{param_name}`:

```python
import numpy as np
data = np.load("my_model.npz")
print(list(data.keys()))   # ['layer_0_W', 'layer_0_b', 'layer_1_gamma', ...]
print(data["layer_0_W"].shape)
```

---

### Programmatic weight access

```python
# Get a snapshot of all weights (list of per-layer dicts)
weights = model.get_weights()

# Do something with them
weights[0]["W"] *= 0.9   # scale layer 0's weight matrix

# Restore
model.set_weights(weights)
```

---

### Freezing layers

Set `layer.trainable = False` to exclude a layer from parameter updates:

```python
# Freeze the first layer
model._layers[0].trainable = False
model.fit(X, y, epochs=20)   # only layers 1+ are updated
```
