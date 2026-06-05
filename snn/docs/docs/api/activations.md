# snn.activations

## Base

```{eval-rst}
.. autoclass:: snn.activations.Activation
   :members: forward, backward
   :show-inheritance:
```

## Classic activations

```{eval-rst}
.. autoclass:: snn.activations.Linear
   :show-inheritance:

.. autoclass:: snn.activations.ReLU
   :show-inheritance:

.. autoclass:: snn.activations.LeakyReLU
   :members:
   :show-inheritance:

.. autoclass:: snn.activations.ELU
   :members:
   :show-inheritance:

.. autoclass:: snn.activations.SELU
   :show-inheritance:

.. autoclass:: snn.activations.Sigmoid
   :show-inheritance:

.. autoclass:: snn.activations.Tanh
   :show-inheritance:

.. autoclass:: snn.activations.Softmax
   :show-inheritance:

.. autoclass:: snn.activations.Softplus
   :show-inheritance:

.. autoclass:: snn.activations.Swish
   :show-inheritance:

.. autoclass:: snn.activations.Mish
   :show-inheritance:
```

## Non-linear / smooth activations

These activations are particularly effective when the target function is
smooth or polynomial-like (e.g. y = x², signal reconstruction, physics-
informed networks).

```{eval-rst}
.. autoclass:: snn.activations.GELU
   :show-inheritance:

.. autoclass:: snn.activations.PReLU
   :members:
   :show-inheritance:

.. autoclass:: snn.activations.Sine
   :show-inheritance:

.. autoclass:: snn.activations.Hardswish
   :show-inheritance:

.. autoclass:: snn.activations.BentIdentity
   :show-inheritance:

.. autoclass:: snn.activations.Squareplus
   :members:
   :show-inheritance:
```

## Recent additions (efficiency + NLP)

```{eval-rst}
.. autoclass:: snn.activations.ReLU6
   :show-inheritance:

.. autoclass:: snn.activations.Hardsigmoid
   :show-inheritance:

.. autoclass:: snn.activations.LogSoftmax
   :show-inheritance:

.. autoclass:: snn.activations.Sparsemax
   :show-inheritance:

.. autoclass:: snn.activations.CELU
   :members:
   :show-inheritance:

.. autoclass:: snn.activations.Softsign
   :show-inheritance:

.. autoclass:: snn.activations.Tanhshrink
   :show-inheritance:
```

## Factory

```{eval-rst}
.. autofunction:: snn.activations.get
```
