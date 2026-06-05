# snn.initializers

All initializer functions take a `shape` tuple and an optional `seed` integer
and return a NumPy array of that shape.

## Zero / Constant

```{eval-rst}
.. autofunction:: snn.initializers.zeros
.. autofunction:: snn.initializers.ones
```

## Random

```{eval-rst}
.. autofunction:: snn.initializers.random_normal
.. autofunction:: snn.initializers.random_uniform
```

## Variance Scaling

```{eval-rst}
.. autofunction:: snn.initializers.glorot_uniform
.. autofunction:: snn.initializers.glorot_normal
.. autofunction:: snn.initializers.he_uniform
.. autofunction:: snn.initializers.he_normal
.. autofunction:: snn.initializers.lecun_uniform
.. autofunction:: snn.initializers.lecun_normal
```

## Factory

```{eval-rst}
.. autofunction:: snn.initializers.get
```
