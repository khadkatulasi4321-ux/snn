# snn.model

snn ships three model types with identical `compile → fit → evaluate → predict`
APIs.  Choose the one that fits your architecture:

| | `Sequential` | `Model` | `GraphModel` |
|---|---|---|---|
| Linear pipeline | ✅ | ✅ | ✅ |
| Skip connections | ✅ via `Residual` | ✅ | ✅ |
| Multiple inputs/outputs | ❌ | ✅ | ✅ |
| Auto backward | ✅ | ❌ | ✅ |

## Sequential

```{eval-rst}
.. autoclass:: snn.model.Sequential
   :members: add, compile, fit, predict, evaluate, summary, get_weights, set_weights, save_weights, load_weights
   :member-order: bysource
   :show-inheritance:
```

## Model (subclassable)

```{eval-rst}
.. autoclass:: snn.model.Model
   :members: call, backward, compile, fit, predict, evaluate, summary, save_weights, load_weights
   :member-order: bysource
   :show-inheritance:
```

## GraphModel (functional API)

```{eval-rst}
.. autoclass:: snn.model.GraphModel
   :members: compile, fit, predict, evaluate, summary, save_weights, load_weights
   :member-order: bysource
   :show-inheritance:
```

## Input

```{eval-rst}
.. autoclass:: snn.model.Input
   :members:
   :show-inheritance:
```
