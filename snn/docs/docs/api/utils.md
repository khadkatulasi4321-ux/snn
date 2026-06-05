# snn.utils

## Data

```{eval-rst}
.. autofunction:: snn.utils.to_categorical
.. autofunction:: snn.utils.normalize
.. autofunction:: snn.utils.standardize
.. autofunction:: snn.utils.train_test_split
.. autofunction:: snn.utils.batch_generator
```

## Gradient Utilities

```{eval-rst}
.. autofunction:: snn.utils.clip_gradients
```

## Learning Rate Schedules

```{eval-rst}
.. autofunction:: snn.utils.learning_rate_schedule
.. autofunction:: snn.utils.cosine_annealing
.. autofunction:: snn.utils.warmup_schedule
```

## Callbacks

```{eval-rst}
.. autoclass:: snn.utils.EarlyStopping
   :members: __call__
   :show-inheritance:

.. autoclass:: snn.utils.ReduceLROnPlateau
   :members: __call__
   :show-inheritance:
```
