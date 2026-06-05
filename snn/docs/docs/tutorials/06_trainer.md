# Tutorial 6 — Advanced Training with Trainer

The `Trainer` class wraps a compiled model and adds three professional-grade
features on top of the basic `model.fit()` loop:

1. **Gradient accumulation** — simulate larger batch sizes without using more memory
2. **Mixed-precision simulation** — test FP16 precision effects
3. **Multi-metric checkpointing** — save the best model weights automatically

---

## Import

```python
from snn.trainer import Trainer, Checkpoint
```

---

## Gradient accumulation

When you need a large effective batch size (for stable training) but cannot
fit it in memory at once, accumulate gradients over N micro-batches before
doing one weight update.

```
Effective batch size = batch_size × gradient_accumulation_steps
```

```python
from snn.model import Sequential
from snn.layers import Dense, BatchNormalization, Dropout

model = Sequential([
    Dense(128, activation="relu"),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dense(10, activation="softmax"),
])
model.compile("adamw", "categorical_crossentropy",
              metrics=["categorical_accuracy"],
              learning_rate=3e-4, weight_decay=1e-4)

# Accumulate 8 micro-batches before each update
# → effective batch size = 16 × 8 = 128
trainer = Trainer(model, gradient_accumulation_steps=8)
trainer.fit(X_train, y_train, epochs=30, batch_size=16)
```

---

## Mixed-precision simulation

Cast activations and gradients to `float16` and back during each forward/
backward pass. This simulates the precision loss that would occur on real FP16
hardware (e.g. A100 GPU `torch.cuda.amp`), useful for testing whether your
architecture is numerically stable in lower precision.

```python
trainer = Trainer(model, mixed_precision=True)
```

There is no speed benefit in pure NumPy (we are still doing `float64` math
after the cast), but it reveals numerical instability before you deploy.

---

## Gradient clipping (Trainer version)

```python
trainer = Trainer(model, clip_grad_norm=1.0)
```

This is equivalent to `model.compile(..., clip_norm=1.0)` but applies through
the Trainer's accumulation-aware loop (clips after all micro-batch gradients
have been accumulated, not per micro-batch).

---

## Checkpointing

`Checkpoint` monitors a metric and saves a copy of the best weights seen so far.

```python
from snn.trainer import Checkpoint

ckpt_loss = Checkpoint(monitor="val_loss",             mode="min")
ckpt_acc  = Checkpoint(monitor="val_categorical_accuracy", mode="max")
```

| Parameter | Description |
|---|---|
| `monitor` | Metric name to watch (must appear in `history`) |
| `mode` | `"min"` for losses, `"max"` for accuracies, `"auto"` to infer |
| `save_path` | File prefix for disk saves (default `"checkpoint"`) |
| `verbose` | `1` = print when a new best is found |

```python
trainer = Trainer(model, checkpoints=[ckpt_loss, ckpt_acc])
history = trainer.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, y_val),
)

# After training — restore the weights that gave best val_loss
ckpt_loss.restore(model)
model.evaluate(X_test, y_test)

# Also write the best weights to disk
ckpt_acc.save_to_disk(model)   # → checkpoint.npz
```

---

## Trainer callbacks

Trainer callbacks receive `(epoch, history, model, optimizer)` — four
arguments, unlike `model.fit()` callbacks which only receive two.

```python
def lr_warmup(epoch, history, model, optimizer):
    """Linear warmup for the first 5 epochs."""
    if epoch <= 5:
        optimizer.learning_rate = 3e-4 * (epoch / 5)

def reduce_on_plateau(epoch, history, model, optimizer):
    """Halve LR if val_loss hasn't improved for 10 epochs."""
    if epoch > 10:
        recent = history["val_loss"][-10:]
        if min(recent) == recent[-1]:    # no improvement
            optimizer.learning_rate *= 0.5

trainer = Trainer(model, checkpoints=[ckpt_loss])
trainer.fit(
    X_train, y_train,
    epochs=100,
    validation_data=(X_val, y_val),
    callbacks=[lr_warmup, reduce_on_plateau],
)
```

---

## Full Trainer example

```python
import numpy as np
from snn.model import Sequential
from snn.layers import Dense, BatchNormalization, Dropout
from snn.trainer import Trainer, Checkpoint
from snn.utils import EarlyStopping, cosine_annealing

# -- Build and compile -------------------------------------------------------
model = Sequential([
    Dense(256, activation="gelu"),
    BatchNormalization(),
    Dropout(0.4),
    Dense(128, activation="gelu"),
    BatchNormalization(),
    Dropout(0.3),
    Dense(10, activation="softmax"),
])

model.compile(
    optimizer={"name": "adamw", "learning_rate": 3e-4, "weight_decay": 1e-4},
    loss="categorical_crossentropy",
    metrics=["categorical_accuracy"],
    verbose=True,    # print compile summary
)

# -- Callbacks ---------------------------------------------------------------
schedule = cosine_annealing(initial_lr=3e-4, t_max=100, eta_min=1e-6)

def lr_schedule(epoch, history, model, optimizer):
    optimizer.learning_rate = schedule(epoch)

early_stop = EarlyStopping(monitor="val_loss", patience=15, verbose=1)

def early_stop_cb(epoch, history):
    early_stop(epoch, history)

# -- Checkpoints -------------------------------------------------------------
ckpt = Checkpoint(monitor="val_categorical_accuracy", mode="max", verbose=1)

# -- Train -------------------------------------------------------------------
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
    callbacks=[lr_schedule, early_stop_cb],
)

# -- Restore best and evaluate -----------------------------------------------
ckpt.restore(model)
results = model.evaluate(X_test, y_test)
print(f"\nBest test accuracy: {results['categorical_accuracy']:.4f}")
```

---

## When to use Trainer vs model.fit()

| Scenario | Use |
|---|---|
| Quick experiments | `model.fit()` |
| Need gradient clipping only | `model.compile(..., clip_norm=1.0)` |
| Large effective batch size | `Trainer(gradient_accumulation_steps=N)` |
| Best-weights checkpointing | `Trainer(checkpoints=[...])` |
| FP16 stability testing | `Trainer(mixed_precision=True)` |
| Production training with all features | `Trainer` with all options |
