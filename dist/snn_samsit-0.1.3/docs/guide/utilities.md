# Utilities

## Data preprocessing

```python
from snn.utils import normalize, standardize, to_categorical, train_test_split

# Min-max scale to [0, 1]
X_norm = normalize(X)                       # global min/max
X_norm = normalize(X, axis=0)               # per-feature

# Standardize to zero mean / unit variance
X_std = standardize(X, axis=0)

# Integer labels → one-hot matrix
y_one_hot = to_categorical(y_int, num_classes=10)

# Random train / test split (no sklearn required)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, seed=42
)
```

---

## Mini-batch iteration

```python
from snn.utils import batch_generator

for xb, yb in batch_generator(X_train, y_train, batch_size=32):
    # custom training loop
    ...
```

---

## Gradient clipping

```python
from snn.utils import clip_gradients

# Returns a new dict with clipped values
grads_clipped = clip_gradients(grads, max_norm=5.0)
```

---

## Activations

All activations can be used by string, instance, or as a standalone function:

```python
import snn.activations as A

act = A.get("swish")          # returns a Swish() instance
out = act.forward(x)
grad = act.backward(out_grad)

# Direct construction
act = A.ReLU()
act = A.LeakyReLU(alpha=0.01)
act = A.ELU(alpha=1.0)
act = A.Softmax()
```

Available: `linear`, `relu`, `leaky_relu`, `elu`, `selu`, `sigmoid`, `tanh`,
`softmax`, `softplus`, `swish`, `mish`.

---

## Initializers

```python
import snn.initializers as I

W = I.he_normal((fan_in, fan_out))
W = I.glorot_uniform((fan_in, fan_out))
b = I.zeros((1, units))

init_fn = I.get("he_normal")     # returns the function
W = init_fn((64, 128))
```

Available: `zeros`, `ones`, `random_normal`, `random_uniform`,
`glorot_uniform`, `glorot_normal`, `he_uniform`, `he_normal`,
`lecun_uniform`, `lecun_normal`.

---

## Metrics

```python
import snn.metrics as M

acc   = M.accuracy(y_pred, y_true)
bacc  = M.binary_accuracy(y_pred, y_true, threshold=0.5)
cacc  = M.categorical_accuracy(y_pred, y_true)
sacc  = M.sparse_categorical_accuracy(y_pred, y_true)
topk  = M.top_k_accuracy(y_pred, y_true, k=5)

prec  = M.precision(y_pred, y_true)
rec   = M.recall(y_pred, y_true)
f1    = M.f1_score(y_pred, y_true)

mse   = M.mean_squared_error(y_pred, y_true)
mae   = M.mean_absolute_error(y_pred, y_true)
r2    = M.r2_score(y_pred, y_true)

cm    = M.confusion_matrix(y_pred, y_true, n_classes=4)
print(cm)
```

Use strings in `model.compile(metrics=[...])` for automatic logging:

```python
model.compile(
    ...
    metrics=["accuracy", "precision", "recall", "r2_score"],
)
```
