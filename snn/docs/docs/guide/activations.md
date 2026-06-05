# Activation Functions

Activations introduce non-linearity into the network. Every layer that accepts
an `activation=` argument can take either a **string key** or an
**`Activation` instance**.

```python
from snn.layers import Dense
from snn.activations import GELU, PReLU, Sine

# String shorthand
Dense(64, activation="gelu")

# Instance — needed when you want to pass constructor arguments
Dense(64, activation=PReLU(alpha=0.1))
Dense(64, activation=Sine())
```

---

## Choosing an activation

| Task | Good choices |
|---|---|
| Classification (hidden) | `relu`, `gelu`, `swish`, `mish` |
| Classification (output) | `softmax` (multi-class), `sigmoid` (binary) |
| Regression (smooth / polynomial) | `sine`, `gelu`, `bent_identity`, `squareplus` |
| Regression (general) | `swish`, `mish`, `elu` |
| Very deep networks | `selu`, `gelu` |
| Lightweight / mobile | `hardswish` |
| Adaptive / learnable slope | `prelu` |

---

## All activations at a glance

| String key | Class | Notes |
|---|---|---|
| `"linear"` | `Linear` | Identity — no activation |
| `"relu"` | `ReLU` | Fast, sparse; can die on negative inputs |
| `"leaky_relu"` | `LeakyReLU(alpha)` | Fixes dying ReLU; α default 0.01 |
| `"elu"` | `ELU(alpha)` | Smooth negative, faster convergence |
| `"selu"` | `SELU` | Self-normalising for deep nets |
| `"sigmoid"` | `Sigmoid` | Output ∈ (0, 1); binary classification |
| `"tanh"` | `Tanh` | Output ∈ (−1, 1); recurrent layers |
| `"softmax"` | `Softmax` | Probability vector; multi-class output |
| `"softplus"` | `Softplus` | Smooth ReLU approximation |
| `"swish"` | `Swish` | x·σ(x); smooth, non-monotonic |
| `"mish"` | `Mish` | x·tanh(softplus(x)); self-regularising |
| `"gelu"` | `GELU` | Gaussian gating; default in GPT/BERT |
| `"prelu"` | `PReLU(alpha)` | Learnable negative slope |
| `"sine"` | `Sine` | sin(x); SIREN networks for smooth functions |
| `"hardswish"` | `Hardswish` | Piecewise Swish; efficient mobile activation |
| `"bent_identity"` | `BentIdentity` | Smooth, non-saturating; good for regression |
| `"squareplus"` | `Squareplus(b)` | Smooth monotonic ReLU with quadratic shape |

---

## Learning non-linear functions (e.g. y = x²)

Standard ReLU networks *can* approximate any smooth function, but smooth
activations converge faster on curved targets.  The table below shows typical
R² scores after 600 epochs on y = x² (no noise):

| Activation | R² (typical) |
|---|---|
| `sine` | ≈ 0.999 |
| `gelu` | ≈ 0.998 |
| `bent_identity` | ≈ 0.997 |
| `squareplus` | ≈ 0.996 |
| `swish` | ≈ 0.995 |
| `relu` | ≈ 0.990 |

Run the bundled example to see this on your machine:

```bash
python3 snn/examples/nonlinear_regression.py
```

---

## GELU

```python
from snn.activations import GELU

act = GELU()
# or: Dense(64, activation="gelu")
```

Uses the tanh approximation `0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 x³)))`.
The standard activation in modern Transformers (GPT, BERT). Smooth, non-monotonic,
and slightly better than ReLU/ELU on most tasks.

---

## PReLU — Learnable slope

```python
from snn.activations import PReLU

# Fixed initial slope
act = PReLU(alpha=0.25)

# Used inside a Dense layer — alpha is trained automatically
model = Sequential([
    Dense(64, activation=PReLU(alpha=0.1)),
    Dense(64, activation=PReLU(alpha=0.1)),
    Dense(1),
])
model.compile(optimizer="adam", loss="mse")
model.fit(X, y, epochs=200)
```

`alpha` is updated by the optimizer every step alongside the layer's weights.
Each layer's PReLU has its own independent alpha.

---

## Sine (SIREN)

```python
from snn.activations import Sine

model = Sequential([
    Dense(64, activation=Sine()),
    Dense(64, activation=Sine()),
    Dense(1),
])
```

Sinusoidal Representation Networks (SIRENs) use `sin(x)` throughout.
Because `d/dx sin(x) = cos(x)` and the derivative of a SIREN is also a SIREN,
the network naturally represents smooth functions and their derivatives.
Ideal for:

- Polynomial regression (y = x², y = x³ + 2x)
- Physics-informed networks
- Image / audio signal reconstruction

---

## Bent Identity

```python
Dense(64, activation="bent_identity")
```

`f(x) = (√(x²+1) − 1) / 2 + x`

- Derivative is always ≥ 1 → no dying neurons
- Non-linear at every point, unlike ReLU which is linear in each half
- Good general-purpose regression activation

---

## Squareplus

```python
from snn.activations import Squareplus

Dense(64, activation=Squareplus(b=4.0))
# or: Dense(64, activation="squareplus")  # uses default b=4.0
```

`f(x) = (x + √(x² + b)) / 2`

Smooth, monotonic, always positive. Controlled curvature at origin via `b`.
As b → 0 it becomes ReLU; larger b makes the transition smoother.

---

## Hardswish

```python
Dense(64, activation="hardswish")
```

`f(x) = x · clamp(x + 3, 0, 6) / 6`

Piecewise-smooth Swish approximation. Computationally efficient — requires
only addition, clamp, and multiply — while retaining the non-monotonic
character of Swish. Used in MobileNetV3.
