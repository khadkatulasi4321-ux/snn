# Layers

Every layer inherits from `snn.layers.base.Layer` and exposes:

| Method / property | Description |
|---|---|
| `forward(x, training=False)` | Compute output; cache intermediate values for backward |
| `backward(grad)` | Compute & store param gradients, return input gradient |
| `params` | `dict` of trainable arrays |
| `grads` | `dict` of corresponding gradients |
| `count_params()` | Total number of trainable scalar values |
| `trainable` | Set to `False` to freeze the layer |

---

## Dense (Fully-Connected)

```python
from snn.layers import Dense

layer = Dense(
    units=128,
    activation="relu",
    use_bias=True,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
)
```

Applies `output = activation(x @ W + b)`.

**Initializer strings:** `"glorot_uniform"`, `"glorot_normal"`, `"he_uniform"`,
`"he_normal"`, `"lecun_uniform"`, `"lecun_normal"`, `"random_normal"`,
`"random_uniform"`, `"zeros"`, `"ones"`.

---

## Conv2D

```python
from snn.layers import Conv2D

layer = Conv2D(
    filters=32,
    kernel_size=3,         # or (3, 5) for non-square kernels
    stride=1,
    padding="same",        # "valid", "same", or an integer
    activation="relu",
    use_bias=True,
    kernel_initializer="he_normal",
)
```

**Input shape:** `(N, H, W, C)` — **channels-last** convention throughout.

Internally uses `im2col` + matrix multiplication for efficiency. Very large
feature maps will be slow because the im2col extraction is pure Python.

---

## Pooling

```python
from snn.layers import (
    MaxPooling2D, AveragePooling2D,
    GlobalAveragePooling2D, GlobalMaxPooling2D,
)

MaxPooling2D(pool_size=2, stride=None)          # stride defaults to pool_size
AveragePooling2D(pool_size=2, stride=None)
GlobalAveragePooling2D()    # (N,H,W,C) → (N,C)
GlobalMaxPooling2D()        # (N,H,W,C) → (N,C)
```

---

## Dropout

Inverted dropout: at **train time** random units are set to zero and the
remaining values are scaled by `1 / (1 - rate)`, so inference needs no
rescaling.

```python
from snn.layers import Dropout, SpatialDropout2D

Dropout(rate=0.5)            # element-wise dropout
SpatialDropout2D(rate=0.3)   # drops entire feature maps (channels)
```

The `training` flag is passed by `Sequential` automatically — you never set it
manually.

---

## BatchNormalization

Normalises activations over the batch dimension during training; uses
exponential moving-average statistics at inference.

```python
from snn.layers import BatchNormalization, LayerNormalization

BatchNormalization(
    axis=-1,           # feature axis (last by default)
    momentum=0.99,     # EMA momentum for running stats
    epsilon=1e-5,
    center=True,       # learnable beta offset
    scale=True,        # learnable gamma scale
)

LayerNormalization(epsilon=1e-6)  # normalises over the feature axis per sample
```

---

## Flatten & Reshape

```python
from snn.layers import Flatten, Reshape

Flatten()                    # (N, H, W, C) → (N, H*W*C)
Reshape(target_shape=(4, 8)) # (N, 32) → (N, 4, 8)
```

---

## Recurrent Layers

### SimpleRNN

Vanilla (Elman) RNN with full BPTT:

```python
from snn.layers import SimpleRNN

SimpleRNN(
    units=64,
    activation="tanh",
    return_sequences=False,   # True → output at every step
    return_state=False,       # True → return (output, h_T) tuple
)
```

### LSTM

Long Short-Term Memory with fused gate matrices for efficiency:

```python
from snn.layers import LSTM

LSTM(
    units=128,
    return_sequences=True,
    return_state=False,
)
```

The weight matrices are stored fused: `W` is `(input_dim, 4*units)` (all four
gate projections concatenated), and `U` is `(units, 4*units)`. This avoids four
separate matmuls per time step.

### GRU

Gated Recurrent Unit — fewer gates than LSTM, often faster:

```python
from snn.layers import GRU

GRU(
    units=64,
    return_sequences=False,
    return_state=False,
)
```

**Input shape for all recurrent layers:** `(N, T, features)` where `T` is the
sequence length.

---

## Stacking recurrent layers

Use `return_sequences=True` on all but the last recurrent layer:

```python
model = Sequential([
    LSTM(128, return_sequences=True),
    Dropout(0.3),
    GRU(64),
    Dense(10, activation="softmax"),
])
```
