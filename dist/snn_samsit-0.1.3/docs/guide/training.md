# Training

## Optimizers

All optimizers live in `snn.optimizers`. Pass an instance to `model.compile()` or use a string shorthand.

| String key | Class | Key hyperparams |
|---|---|---|
| `"sgd"` | `SGD` | `learning_rate`, `momentum`, `nesterov`, `weight_decay` |
| `"adam"` | `Adam` | `learning_rate`, `beta_1`, `beta_2`, `epsilon`, `weight_decay`, `amsgrad` |
| `"adamw"` | `AdamW` | `learning_rate`, `beta_1`, `beta_2`, `epsilon`, `weight_decay` |
| `"rmsprop"` | `RMSprop` | `learning_rate`, `rho`, `epsilon`, `momentum`, `centered` |
| `"adagrad"` | `Adagrad` | `learning_rate`, `epsilon`, `initial_accumulator_value` |
| `"adadelta"` | `Adadelta` | `learning_rate`, `rho`, `epsilon` |

```python
from snn.optimizers import SGD, Adam, AdamW, RMSprop

# SGD with Nesterov momentum
opt = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)

# AdamW with weight decay
opt = AdamW(learning_rate=3e-4, weight_decay=0.01)

# Centred RMSprop
opt = RMSprop(learning_rate=1e-3, centered=True)
```

---

## Loss functions

| String key | Class | Notes |
|---|---|---|
| `"mse"` | `MeanSquaredError` | Regression |
| `"mae"` | `MeanAbsoluteError` | Regression, robust to outliers |
| `"huber"` | `HuberLoss(delta=1.0)` | Blend of MSE + MAE |
| `"binary_crossentropy"` | `BinaryCrossentropy` | Binary classification |
| `"categorical_crossentropy"` | `CategoricalCrossentropy` | Multi-class, one-hot targets |
| `"sparse_categorical_crossentropy"` | `SparseCategoricalCrossentropy` | Multi-class, integer targets |
| `"kl_divergence"` | `KLDivergence` | Distribution matching |

Pass `from_logits=True` to BCE or CCE when your last layer has **no activation**
(e.g., `Dense(10)` with no softmax). The loss will apply softmax internally for
numerical stability.

---

## Callbacks

Callbacks are callables invoked at the end of each epoch by `model.fit()`.
The signature expected by the built-in utilities is:

```python
def my_callback(epoch, history):
    ...
```

### EarlyStopping

```python
from snn.utils import EarlyStopping

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    min_delta=1e-4,
    mode="auto",              # "min" / "max" / "auto"
    restore_best_weights=False,
    verbose=1,
)

model.fit(X, y, epochs=200, callbacks=[early_stop])
```

When patience is exhausted, the callback raises `StopIteration` which `fit()`
catches and stops training gracefully.

### ReduceLROnPlateau

```python
from snn.utils import ReduceLROnPlateau

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    min_lr=1e-6,
)

# ReduceLROnPlateau needs the optimizer, so wrap it:
opt = Adam(learning_rate=1e-3)
model.compile(optimizer=opt, loss="mse")

def lr_callback(epoch, history):
    reduce_lr(epoch, history, optimizer=opt)

model.fit(X, y, epochs=100, callbacks=[lr_callback])
```

---

## Learning rate schedules

```python
from snn.utils import (
    learning_rate_schedule,
    cosine_annealing,
    warmup_schedule,
)

# Exponential decay
schedule = learning_rate_schedule(initial_lr=1e-3, decay_rate=0.96, decay_steps=1000)

# Cosine annealing
schedule = cosine_annealing(initial_lr=1e-3, t_max=100, eta_min=1e-6)

# Linear warmup → cosine annealing
base = cosine_annealing(initial_lr=1e-3, t_max=200)
schedule = warmup_schedule(initial_lr=1e-3, warmup_steps=10, base_schedule=base)

# Apply manually each epoch:
opt = Adam(learning_rate=1e-3)
def lr_sched_callback(epoch, history):
    opt.learning_rate = schedule(epoch)

model.fit(X, y, epochs=200, callbacks=[lr_sched_callback])
```

---

## Advanced Trainer

`Trainer` extends the training loop with three power-user features.

### Gradient accumulation

Accumulate gradients over `N` micro-batches before updating weights.
Effective batch size = `batch_size × gradient_accumulation_steps`.

```python
from snn.trainer import Trainer

trainer = Trainer(model, gradient_accumulation_steps=8)
trainer.fit(X, y, epochs=20, batch_size=16)
# effective batch size = 128
```

### Mixed-precision simulation

Cast activations and gradients to `float16` and back to `float64` during
forward/backward passes — simulates the precision loss of real FP16 hardware
without requiring a GPU.

```python
trainer = Trainer(model, mixed_precision=True)
```

### Gradient clipping

Clip the global gradient norm before each optimizer step:

```python
trainer = Trainer(model, clip_grad_norm=1.0)
```

### Checkpointing

Save best weights for multiple metrics independently:

```python
from snn.trainer import Checkpoint

ckpt_loss = Checkpoint(monitor="val_loss",     mode="min")
ckpt_acc  = Checkpoint(monitor="val_accuracy", mode="max")

trainer = Trainer(model, checkpoints=[ckpt_loss, ckpt_acc])
trainer.fit(X, y, epochs=50, validation_data=(X_val, y_val))

ckpt_acc.restore(model)     # apply best-accuracy weights
ckpt_loss.save_to_disk(model)  # persist to .npz
```

### Full Trainer example

```python
from snn.trainer import Trainer, Checkpoint
from snn.utils import EarlyStopping

model.compile(
    optimizer=AdamW(learning_rate=3e-4, weight_decay=1e-4),
    loss="categorical_crossentropy",
    metrics=["categorical_accuracy"],
)

ckpt = Checkpoint(monitor="val_categorical_accuracy", mode="max")

def lr_warmup(epoch, history, model, optimizer):
    if epoch <= 5:
        optimizer.learning_rate = 3e-4 * epoch / 5

trainer = Trainer(
    model,
    gradient_accumulation_steps=4,
    mixed_precision=True,
    clip_grad_norm=1.0,
    checkpoints=[ckpt],
    verbose=1,
)

history = trainer.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[lr_warmup],
)

ckpt.restore(model)
model.evaluate(X_test, y_test)
```
