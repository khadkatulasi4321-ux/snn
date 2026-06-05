# Tutorial 8 — Beyond Sequential: Model & GraphModel

`Sequential` is great for straight pipelines. But real architectures — ResNets,
multi-branch networks, shared weights — need something more. snn ships two
additional model types that cover these cases while keeping the same
`compile → fit → evaluate` API you already know.

---

## Recap: Sequential (the baseline)

```python
from snn.model import Sequential
from snn.layers import Dense, Dropout

model = Sequential([
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(10, activation='softmax'),
])
model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=20)
```

For anything with a skip connection, multiple inputs, or shared layers you need
one of the next two classes.

---

## 1 — `Model` (subclassable, PyTorch-style)

Define your layers in `__init__`, the **forward pass** in `call()`, and
the **backward pass** in `backward()`.  All Layer attributes are auto-discovered
for the optimizer.

```python
import numpy as np
from snn.model import Model
from snn.layers import Dense, BatchNormalization

class MyNet(Model):
    def __init__(self):
        super().__init__()
        self.fc1 = Dense(128, activation='relu')
        self.bn   = BatchNormalization()
        self.fc2  = Dense(10, activation='softmax')

    def call(self, x, training=False):
        x = self.fc1(x, training=training)
        x = self.bn(x,  training=training)
        return self.fc2(x, training=training)

    def backward(self, grad):
        grad = self.fc2.backward(grad)
        grad = self.bn.backward(grad)
        return self.fc1.backward(grad)

model = MyNet()
model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val))
```

### Skip connections (residual block)

```python
class ResBlock(Model):
    """One residual block: two Dense layers + skip connection."""

    def __init__(self, units):
        super().__init__()
        self.fc1  = Dense(units, activation='relu')
        self.fc2  = Dense(units)         # no activation — applied after sum
        self.proj = Dense(units)         # project input to same dim

    def call(self, x, training=False):
        shortcut = self.proj(x, training=training)
        h = self.fc1(x, training=training)
        h = self.fc2(h, training=training)
        self._pre_act = h + shortcut
        return np.maximum(0, self._pre_act)   # ReLU *after* the residual sum

    def backward(self, grad):
        d = grad * (self._pre_act > 0)        # ReLU gate
        d_proj = self.proj.backward(d)
        d = self.fc2.backward(d)
        d = self.fc1.backward(d)
        return d + d_proj                      # sum: main path + shortcut

# Use just like Sequential
block = ResBlock(units=64)
block.compile('adamw', 'mse')
block.fit(X_train, y_train, epochs=15)
```

### Stacking blocks

```python
class ResNet(Model):
    def __init__(self):
        super().__init__()
        self.stem   = Dense(64, activation='relu')
        self.block1 = ResBlock(64)
        self.block2 = ResBlock(64)
        self.head   = Dense(10, activation='softmax')

    def call(self, x, training=False):
        x = self.stem(x, training=training)
        x = self.block1.call(x, training=training)   # call sub-model's call()
        x = self.block2.call(x, training=training)
        return self.head(x, training=training)

    def backward(self, grad):
        grad = self.head.backward(grad)
        grad = self.block2.backward(grad)
        grad = self.block1.backward(grad)
        return self.stem.backward(grad)
```

> **Rule of thumb** — `call()` and `backward()` mirror each other exactly
> (reversed layer order, everything stored in `forward` is used in `backward`).

---

## 2 — `GraphModel` (functional API)

Build a computation **graph** by wiring layers to `Input` nodes.  The model
traces the DAG and handles forward + backward automatically — no `backward()`
implementation needed.

```python
from snn.model import Input, GraphModel
from snn.layers import Dense

x   = Input(shape=(784,))
h   = Dense(256, activation='relu')(x)    # returns a graph node
h   = Dense(128, activation='relu')(h)
out = Dense(10, activation='softmax')(h)

model = GraphModel(inputs=x, outputs=out)
model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=20)
```

### Skip connections with `Add`

```python
from snn.model import Input, GraphModel
from snn.layers import Dense, Add

x      = Input(shape=(128,))
shortcut = Dense(64)(x)                   # projection shortcut

h      = Dense(64, activation='relu')(x)
h      = Dense(64)(h)
merged = Add()([h, shortcut])             # element-wise add — pass a LIST
out    = Dense(10, activation='softmax')(merged)

model = GraphModel(inputs=x, outputs=out)
model.compile('adamw', 'categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=25, learning_rate=3e-4)
```

### Multi-branch / feature fusion

```python
from snn.model import Input, GraphModel
from snn.layers import Dense, Concatenate

# Two separate input streams
text_in = Input(shape=(300,))    # mean-pooled word vectors
meta_in = Input(shape=(10,))     # numerical metadata

h_text  = Dense(128, 'relu')(text_in)
h_meta  = Dense(32,  'relu')(meta_in)
merged  = Concatenate()([h_text, h_meta])   # shape (..., 160)
out     = Dense(5, 'softmax')(merged)

model = GraphModel(inputs=[text_in, meta_in], outputs=out)
model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
model.fit([X_text, X_meta], y, epochs=15)  # pass a list of arrays
```

---

## 3 — `Residual` layer (quick shortcut inside Sequential)

Don't want to write a custom model class just for a single skip connection?
Use `Residual` directly inside `Sequential`:

```python
from snn.model import Sequential
from snn.layers import Dense, Residual

model = Sequential([
    Dense(128, activation='relu'),
    Residual([                          # wraps sub-layers + adds skip
        Dense(128, activation='relu'),
        Dense(128),
    ]),
    Residual([
        Dense(128, activation='relu'),
        Dense(128),
    ]),
    Dense(10, activation='softmax'),
])
model.compile('adam', 'categorical_crossentropy')
model.fit(X_train, y_train, epochs=30)
```

When the dimensions don't match, `Residual` automatically adds a learnable
projection:

```python
Residual([Dense(64, 'relu'), Dense(64)])   # input 128 → output 64, auto-projects
```

---

## When to use each

| | `Sequential` | `Model` | `GraphModel` |
|---|---|---|---|
| Simple feedforward | ✅ Best | ✅ | ✅ |
| Skip connections | ✅ via `Residual` | ✅ | ✅ |
| Shared weights | ❌ | ✅ | ✅ |
| Multiple inputs/outputs | ❌ | ✅ | ✅ |
| Auto backward | ✅ | ❌ (write it yourself) | ✅ |
| Graph visualization | ❌ | ❌ | ✅ `.summary()` |

---

## Complete example: deep residual classifier

```python
import numpy as np
from snn.model import Sequential
from snn.layers import Dense, Dropout, BatchNormalization, Residual
from snn.utils import to_categorical

# Toy dataset
X = np.random.randn(512, 64)
y = to_categorical(np.random.randint(0, 5, 512), num_classes=5)

model = Sequential([
    Dense(128, activation='relu'),
    BatchNormalization(),
    Residual([Dense(128, activation='relu'), Dense(128)]),
    Dropout(0.2),
    Residual([Dense(128, activation='relu'), Dense(128)]),
    Dropout(0.2),
    Dense(5, activation='softmax'),
])

model.compile(
    optimizer='adamw',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
    learning_rate=1e-3,
    weight_decay=1e-4,
)
history = model.fit(X, y, epochs=30, batch_size=32, verbose=1)
print(model.evaluate(X, y))
```
