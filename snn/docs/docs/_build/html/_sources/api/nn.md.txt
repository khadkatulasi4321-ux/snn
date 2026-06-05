# snn.nn — Flat namespace

`snn.nn` collects **everything** into one flat import so you never have to
remember which sub-module a class lives in.

```python
from snn.nn import Dense, ReLU, Adam, CCE, Sequential
```

It also ships **NNFS-style aliases** — prefixed class names (`Layer_Dense`,
`Activation_ReLU`, `Loss_CCE`, `Optimizer_Adam` …) so code from the
*Neural Networks from Scratch* book ports without any renaming.

---

## Quick start

```python
from snn.nn import Dense, ReLU, Softmax, Adam, CCE, Sequential, to_categorical
import numpy as np

# Build a model using plain class names
model = Sequential([
    Dense(128),
    ReLU(),
    Dense(64),
    ReLU(),
    Dense(10),
    Softmax(),
])

model.compile(Adam(learning_rate=1e-3), CCE(), metrics=['accuracy'])
model.fit(X_train, y_train, epochs=30, batch_size=64)
```

---

## NNFS-style manual loop

```python
from snn.nn import Layer_Dense, Activation_ReLU, Activation_Softmax
from snn.nn import Loss_CCE, Optimizer_Adam
import numpy as np

dense1  = Layer_Dense(784, 128)
relu    = Activation_ReLU()
dense2  = Layer_Dense(128, 10)
softmax = Activation_Softmax()
loss_fn = Loss_CCE()
opt     = Optimizer_Adam(learning_rate=0.001)

for epoch in range(100):
    # Forward
    out = dense1.forward(X_train)
    out = relu.forward(out)
    out = dense2.forward(out)
    out = softmax.forward(out)
    loss = loss_fn.forward(out, y_train)

    # Backward
    grad = loss_fn.backward(out, y_train)
    grad = softmax.backward(grad)
    grad = dense2.backward(grad)
    grad = relu.backward(grad)
    dense1.backward(grad)

    # Update
    opt.update(dense1)
    opt.update(dense2)
```

---

## API reference

```{eval-rst}
.. automodule:: snn.nn
   :members:
   :undoc-members:
   :show-inheritance:
```
