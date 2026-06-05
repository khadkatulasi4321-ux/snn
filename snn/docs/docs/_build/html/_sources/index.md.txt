# snn

**A neural network / deep learning library built purely on NumPy.**

No TensorFlow, no PyTorch, no autograd — every forward pass, backward pass, and weight update is hand-derived and fully vectorized. The result is a library you can actually read, understand, and learn from.

---

```{note}
snn exposes a **Keras-like API** so you can get productive immediately,
then dive into the source to see exactly what happens under the hood.
```

## Highlights

| Feature | Details |
|---|---|
| **3 model types** | `Sequential`, `Model` (subclassable), `GraphModel` (functional / skip connections) |
| **35+ layer types** | Dense, Conv1D/2D, LSTM, GRU, Transformer, Residual, Add, Concatenate, TimeDistributed, … |
| **24 activations** | ReLU, GELU, Swish, Mish, Sparsemax, CELU, Softsign, Tanhshrink, … |
| **7 loss functions** | MSE, MAE, Huber, BCE, CCE, Sparse CCE, KL Divergence |
| **12 optimizers** | SGD, Adam, AdamW, Nadam, RAdam, Lion, **LAMB, Lookahead, Adan**, … |
| **10 initializers** | Glorot, He, LeCun, random normal / uniform, zeros, ones |
| **Language-capable** | Embedding, PositionalEncoding, MultiHeadAttention, TransformerBlock for NLP |
| **Advanced Trainer** | Gradient accumulation, FP16 simulation, multi-metric checkpointing |
| **`snn.nn` flat namespace** | Every class in one place + NNFS-style `Layer_Dense`, `Activation_ReLU` aliases |
| **Built-in doc server** | `python -m snn.docs.serve_docs` or `snn-docs` CLI — zero config |

---

## Install

```bash
pip install -e snn/
```

Or without installing — just add the package to your path:

```python
import sys
sys.path.insert(0, "path/to/snn")
import snn
```

---

## Quick examples

**Sequential** (linear pipeline — the simplest way):

```python
from snn.model import Sequential
from snn.layers import Dense, Dropout, BatchNormalization, Residual

model = Sequential([
    Dense(128, activation="relu"),
    BatchNormalization(),
    Residual([Dense(128, activation="relu"), Dense(128)]),   # skip connection!
    Dropout(0.3),
    Dense(10, activation="softmax"),
])
model.compile("adamw", "categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=30, validation_data=(X_val, y_val))
model.evaluate(X_test, y_test)
```

**GraphModel** (functional API — arbitrary graphs, multiple inputs):

```python
from snn.model import Input, GraphModel
from snn.layers import Dense, Add

x      = Input(shape=(784,))
skip   = Dense(128)(x)
h      = Dense(128, activation="relu")(x)
h      = Dense(128)(h)
merged = Add()([h, skip])                  # skip connection
out    = Dense(10, activation="softmax")(merged)

model = GraphModel(inputs=x, outputs=out)
model.compile("adam", "categorical_crossentropy")
model.fit(X_train, y_train, epochs=20)
```

**Model** (subclassable — PyTorch-style, full control):

```python
from snn.model import Model
from snn.layers import Dense, BatchNormalization

class MyNet(Model):
    def __init__(self):
        super().__init__()
        self.fc1 = Dense(128, activation="relu")
        self.bn  = BatchNormalization()
        self.out = Dense(10, activation="softmax")

    def call(self, x, training=False):
        x = self.fc1(x, training=training)
        x = self.bn(x,  training=training)
        return self.out(x, training=training)

    def backward(self, grad):
        grad = self.out.backward(grad)
        grad = self.bn.backward(grad)
        return self.fc1.backward(grad)

model = MyNet()
model.compile("adam", "categorical_crossentropy")
model.fit(X_train, y_train, epochs=20)
```

---

## Contents

```{toctree}
:maxdepth: 2
:caption: Getting Started

quickstart
```

```{toctree}
:maxdepth: 2
:caption: Tutorials

tutorials/index
```

```{toctree}
:maxdepth: 2
:caption: User Guide

guide/index
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api/index
```
