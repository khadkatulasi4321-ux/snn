# snn.losses

## Base

```{eval-rst}
.. autoclass:: snn.losses.Loss
   :members: forward, backward
   :show-inheritance:
```

## Regression

```{eval-rst}
.. autoclass:: snn.losses.MeanSquaredError
   :show-inheritance:

.. autoclass:: snn.losses.MeanAbsoluteError
   :show-inheritance:

.. autoclass:: snn.losses.HuberLoss
   :members:
   :show-inheritance:
```

## Classification

```{eval-rst}
.. autoclass:: snn.losses.BinaryCrossentropy
   :members:
   :show-inheritance:

.. autoclass:: snn.losses.CategoricalCrossentropy
   :members:
   :show-inheritance:

.. autoclass:: snn.losses.SparseCategoricalCrossentropy
   :members:
   :show-inheritance:
```

## Divergence

```{eval-rst}
.. autoclass:: snn.losses.KLDivergence
   :members:
   :show-inheritance:
```

## Factory

```{eval-rst}
.. autofunction:: snn.losses.get
```
