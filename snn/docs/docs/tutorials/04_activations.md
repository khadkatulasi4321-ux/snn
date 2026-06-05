# Tutorial 4 — Choosing Activation Functions

Activation functions are what make neural networks non-linear. This tutorial
explains every activation in snn, when to use each, and the intuition behind
them.

---

## Why activations matter

Without activations, a stack of Dense layers collapses into a single matrix
multiplication — no matter how many layers you add. Activations break this
linearity and let networks learn curves, boundaries, and complex functions.

```python
# This is just matrix multiplication — useless depth
x → [Dense] → [Dense] → [Dense] → y
# equivalent to: x @ (W1 @ W2 @ W3) + b

# With activations — genuinely deep
x → [Dense + ReLU] → [Dense + ReLU] → [Dense + Sigmoid] → y
# learns complex non-linear patterns
```

---

## All activations at a glance

| Key | Formula | Range | Best for |
|---|---|---|---|
| `"linear"` | x | (−∞, ∞) | Output layer for regression |
| `"relu"` | max(0, x) | [0, ∞) | General purpose hidden layers |
| `"leaky_relu"` | x if x>0 else αx | (−∞, ∞) | Fix dying-ReLU |
| `"elu"` | x if x>0 else α(eˣ−1) | (−α, ∞) | Faster convergence than ReLU |
| `"selu"` | scale·ELU | (−∞, ∞) | Self-normalising deep nets |
| `"sigmoid"` | 1/(1+e⁻ˣ) | (0, 1) | Binary output probability |
| `"tanh"` | tanh(x) | (−1, 1) | RNN/LSTM hidden states |
| `"softmax"` | softmax(x) | (0,1), sums to 1 | Multi-class output |
| `"softplus"` | log(1+eˣ) | (0, ∞) | Smooth ReLU |
| `"swish"` | x·σ(x) | (−∞, ∞) | General purpose, often beats ReLU |
| `"mish"` | x·tanh(softplus(x)) | (−∞, ∞) | Deep nets, self-regularising |
| `"gelu"` | x·Φ(x) | (−∞, ∞) | Transformers, regression |
| `"prelu"` | x if x≥0 else αx (α learned) | (−∞, ∞) | When you want to tune slope |
| `"sine"` | sin(x) | [−1, 1] | Smooth / polynomial fitting |
| `"hardswish"` | x·clip(x+3,0,6)/6 | (−∞, ∞) | Efficient Swish for deployment |
| `"bent_identity"` | (√(x²+1)−1)/2+x | (−∞, ∞) | Regression, never-dying |
| `"squareplus"` | (x+√(x²+b))/2 | (0, ∞) | Smooth monotonic ReLU |

---

## ReLU — the workhorse

```python
Dense(64, activation="relu")
```

`f(x) = max(0, x)` — passes positive values unchanged, kills negatives.

**Pros:** fast to compute, sparse activations, works well in most tasks.

**Cons:** "dying ReLU" — if a neuron's weights push all inputs negative, it
gets a gradient of zero forever and stops learning.

**Fix:** use `LeakyReLU`, `ELU`, or `GELU` if you see dead neurons.

---

## GELU — the modern default

```python
Dense(64, activation="gelu")
```

`f(x) = x · Φ(x)` where Φ is the normal CDF.
Approximated as `0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715 x³)))`.

Used in **BERT, GPT-2, GPT-3, PaLM, and most modern Transformers**.
Slightly outperforms ReLU and ELU on the majority of tasks, especially
regression.

---

## Swish and Mish — smooth and non-monotonic

```python
Dense(64, activation="swish")   # x·σ(x)
Dense(64, activation="mish")    # x·tanh(softplus(x))
```

Both are smooth everywhere and slightly non-monotonic (they dip below zero
for small negative x). This gives the network richer gradient signals than
ReLU. Good alternatives to GELU if you find GELU too slow.

---

## Sine — for smooth functions (SIREN)

```python
from snn.activations import Sine
Dense(64, activation=Sine())
```

`f(x) = sin(x)`

Specifically designed for learning **smooth, periodic, and polynomial-like
functions** (y = x², y = sin(3x), image reconstruction, physics simulations).

The derivative of a Sine network is another Sine network, so it naturally
represents its own derivatives — powerful for physics-informed learning.

```python
# Fitting y = x² with SIREN
model = Sequential([
    Dense(64, activation="sine"),
    Dense(64, activation="sine"),
    Dense(1),
])
model.compile("adam", "mse", learning_rate=1e-3)
model.fit(X, y_squared, epochs=500)
```

---

## PReLU — learnable slope

```python
from snn.activations import PReLU
Dense(64, activation=PReLU(alpha=0.1))
```

`f(x) = x if x ≥ 0 else α·x` — identical to LeakyReLU but `α` is a
**learned parameter** updated by the optimiser every step.

Each layer gets its own independent `α`. To inspect it after training:

```python
learned_alpha = model._layers[0].activation.alpha
print(f"Layer 0 learned alpha: {learned_alpha:.4f}")
```

---

## Bent Identity — for regression

```python
Dense(64, activation="bent_identity")
```

`f(x) = (√(x²+1) − 1) / 2 + x`

The derivative is always ≥ 1, so every neuron always receives gradients
(no dying neurons). Non-linear at every single point — good for regression
tasks where you want gradients everywhere.

---

## Squareplus — smooth monotonic alternative

```python
from snn.activations import Squareplus
Dense(64, activation=Squareplus(b=4.0))
```

`f(x) = (x + √(x² + b)) / 2`

Always positive and differentiable. Parameter `b` controls how sharp the
kink at zero is (b → 0 approaches ReLU; larger b → smoother).

---

## Output layer activations

The **last layer's activation** depends on the task:

```python
# Binary classification → single probability
Dense(1, activation="sigmoid")   # output ∈ (0, 1)

# Multi-class classification → probability vector
Dense(n_classes, activation="softmax")   # sums to 1

# Regression → raw value
Dense(1)    # no activation (or activation="linear")

# Bounded regression (output must be 0..1)
Dense(1, activation="sigmoid")
```

---

## Activation do's and don'ts

| ✅ Do | ❌ Don't |
|---|---|
| Use `relu` or `gelu` for hidden layers | Use `sigmoid` or `tanh` in deep hidden layers (they saturate) |
| Use `softmax` only on the output | Use `softmax` in hidden layers |
| Match output activation to the loss | Use `sigmoid` output with CCE |
| Try `swish`/`gelu` if `relu` plateaus | Stack too many `sine` layers without SIREN init |
| Use `linear` (no activation) for regression output | Use `relu` on the output for regression (clips negatives) |

---

## What next?

- **[Tutorial 5](05_sequences.md)** — LSTM and GRU for sequential data.
- **[Tutorial 6](06_trainer.md)** — advanced training with gradient accumulation and checkpointing.
