"""
merge.py — Layers that combine or branch multiple tensor paths.

These layers enable skip connections, residual blocks, multi-branch
architectures, and any graph that goes beyond a simple linear stack.
"""
import numpy as np
from .base import Layer
from ..activations import get as get_activation


class Add(Layer):
    """
    Element-wise addition of multiple inputs.

    Used in residual / skip-connection architectures.  All inputs must
    have the same shape.

    In **Sequential** context, wrap it in a :class:`Residual` block.
    In **GraphModel** context, call it with a list of Tensor nodes:

    .. code-block:: python

        h1 = Dense(64, 'relu')(x)
        h2 = Dense(64, 'relu')(x)
        merged = Add()([h1, h2])

    Input / output shape
    --------------------
    List of arrays, each ``(batch, …)`` → same shape.
    """

    def forward(self, inputs, training=False):
        if isinstance(inputs, np.ndarray):
            return inputs          # single input pass-through
        self._n = len(inputs)
        out = inputs[0].copy()
        for inp in inputs[1:]:
            out = out + inp
        return out

    def backward(self, grad):
        # Gradient flows unchanged to every input
        return [grad.copy() for _ in range(self._n)]


class Concatenate(Layer):
    """
    Concatenate multiple inputs along one axis.

    .. code-block:: python

        # GraphModel usage
        h1 = Dense(32, 'relu')(x)
        h2 = Dense(32, 'relu')(x)
        merged = Concatenate(axis=-1)([h1, h2])   # shape (..., 64)

    Parameters
    ----------
    axis : int
        Axis along which to concatenate (default ``-1``).
    """

    def __init__(self, axis=-1, name=None):
        super().__init__(name=name)
        self.axis = axis
        self._sizes = None

    def forward(self, inputs, training=False):
        if isinstance(inputs, np.ndarray):
            return inputs
        self._sizes = [inp.shape[self.axis] for inp in inputs]
        return np.concatenate(inputs, axis=self.axis)

    def backward(self, grad):
        # Split gradient back into pieces matching each input
        splits = np.cumsum(self._sizes[:-1])
        return np.split(grad, splits, axis=self.axis)


class Residual(Layer):
    """
    Residual (skip-connection) block — for use inside **Sequential**.

    Wraps a sequence of sub-layers and adds a skip connection::

        output = activation(sub_layers(x) + shortcut(x))

    If the input and output feature sizes differ, a learnable linear
    projection is built lazily and used as the shortcut.

    Parameters
    ----------
    layers : list of Layer
        The main (non-shortcut) path.
    activation : str
        Activation applied *after* the residual sum (default ``"relu"``).
        Pass ``None`` or ``"linear"`` to skip the activation.
    shortcut : Layer, optional
        Explicit shortcut layer.  Auto-built if input/output dims differ.

    Examples
    --------
    Basic residual block inside Sequential::

        model = Sequential([
            Dense(64, activation='relu'),
            Residual([
                Dense(64, activation='relu'),
                Dense(64),              # no activation — applied after sum
            ]),
            Dense(10, activation='softmax'),
        ])

    Two blocks of different sizes (auto-projection)::

        model = Sequential([
            Dense(128, activation='relu'),
            Residual([Dense(64, activation='relu'), Dense(64)]),
            # shortcut Dense(64) auto-added because 128 ≠ 64
            Dense(10, activation='softmax'),
        ])
    """

    def __init__(self, layers, activation="relu", shortcut=None, name=None):
        super().__init__(name=name)
        self._sub = layers
        self._shortcut = shortcut
        self._act = get_activation(activation)
        self._pre_act = None

    def forward(self, x, training=False):
        residual = x
        out = x
        for layer in self._sub:
            out = layer.forward(out, training=training)

        # Lazy projection when dims don't match
        if out.shape[-1] != residual.shape[-1]:
            if self._shortcut is None:
                from .dense import Dense
                self._shortcut = Dense(out.shape[-1])
        if self._shortcut is not None:
            residual = self._shortcut.forward(residual, training=training)

        self._pre_act = out + residual
        return self._act.forward(self._pre_act)

    def backward(self, grad):
        # Through post-sum activation
        d = self._act.backward(grad)

        # Both the main path and the shortcut receive the same upstream grad
        d_main = d.copy()
        d_sc = d.copy()

        # Backward through shortcut
        if self._shortcut is not None:
            d_sc = self._shortcut.backward(d_sc)
        # (else: identity shortcut — d_sc passes unchanged)

        # Backward through main path
        for layer in reversed(self._sub):
            d_main = layer.backward(d_main)

        return d_main + d_sc

    # ── param / grad routing ────────────────────────────────────────────

    @property
    def params(self):
        p = {}
        for i, layer in enumerate(self._sub):
            p.update({f"s{i}_{k}": v for k, v in layer.params.items()})
        if self._shortcut is not None:
            p.update({f"sc_{k}": v for k, v in self._shortcut.params.items()})
        return p

    @property
    def grads(self):
        g = {}
        for i, layer in enumerate(self._sub):
            g.update({f"s{i}_{k}": v for k, v in layer.grads.items()})
        if self._shortcut is not None:
            g.update({f"sc_{k}": v for k, v in self._shortcut.grads.items()})
        return g

    def set_param(self, key, val):
        for i, layer in enumerate(self._sub):
            pfx = f"s{i}_"
            if key.startswith(pfx):
                layer.set_param(key[len(pfx):], val)
                return
        if key.startswith("sc_") and self._shortcut is not None:
            self._shortcut.set_param(key[3:], val)
            return
        if hasattr(self, key):
            setattr(self, key, val)

    def count_params(self):
        total = sum(layer.count_params() for layer in self._sub)
        if self._shortcut is not None:
            total += self._shortcut.count_params()
        return total


class TimeDistributed(Layer):
    """
    Apply a layer independently to each time step of a sequence.

    Reshapes ``(batch, time, features)`` → ``(batch*time, features)``,
    runs the wrapped layer, then reshapes back.  This lets you apply
    any layer (Dense, Conv1D, …) to every position in a sequence
    without writing a Python loop.

    Parameters
    ----------
    layer : Layer
        The layer to apply at each time step.

    Input / output shape
    --------------------
    ``(batch, time, in_features)`` → ``(batch, time, out_features)``

    Examples
    --------
    >>> td = TimeDistributed(Dense(32, activation='relu'))
    >>> x = np.random.randn(8, 20, 64)
    >>> out = td.forward(x)    # (8, 20, 32)

    Typical use — dense classification head over RNN output::

        model = Sequential([
            LSTM(64, return_sequences=True),         # (batch, seq, 64)
            TimeDistributed(Dense(32, 'relu')),      # (batch, seq, 32)
            TimeDistributed(Dense(vocab_size, 'softmax')),
        ])
    """

    def __init__(self, layer, name=None):
        super().__init__(name=name)
        self._layer = layer
        self._input_shape = None

    def forward(self, x, training=False):
        # x: (batch, time, features)
        self._input_shape = x.shape
        batch, time, features = x.shape
        x_flat = x.reshape(batch * time, features)
        out_flat = self._layer.forward(x_flat, training=training)
        out_dim = out_flat.shape[-1]
        return out_flat.reshape(batch, time, out_dim)

    def backward(self, grad):
        batch, time, out_dim = grad.shape
        grad_flat = grad.reshape(batch * time, out_dim)
        dx_flat = self._layer.backward(grad_flat)
        return dx_flat.reshape(self._input_shape)

    @property
    def params(self):
        return self._layer.params

    @property
    def grads(self):
        return self._layer.grads

    def set_param(self, key, val):
        self._layer.set_param(key, val)

    def count_params(self):
        return self._layer.count_params()
