"""
gated.py — Gated linear units as standalone layers.

GLU and SwiGLU project the input to 2×units, split it, and use one half
to gate the other.  They are the standard feed-forward sub-layer in
modern LLMs (LLaMA, PaLM, Mistral).
"""
import numpy as np
from .base import Layer
from ..initializers import get as get_initializer


class GLU(Layer):
    """
    Gated Linear Unit (Dauphin et al. 2017).

    Projects the input to 2 × ``units``, splits into two halves ``(a, b)``,
    and returns ``a ⊙ σ(b)``::

        z = x @ W + bias          # shape (..., 2*units)
        a, b = z[..., :units], z[..., units:]
        output = a * sigmoid(b)   # shape (..., units)

    This halves the effective hidden size while adding a learnable gating
    mechanism.  Widely used in CNNs and early Transformer variants.

    Parameters
    ----------
    units : int
        Output dimensionality (half of the internal projection width).
    use_bias : bool
        Whether to add a bias term (default True).
    kernel_initializer : str
        Weight initialiser (default ``"glorot_uniform"``).

    Examples
    --------
    >>> glu = GLU(units=32)
    >>> x = np.random.randn(4, 20, 64)
    >>> out = glu.forward(x)   # (4, 20, 32)
    """

    def __init__(self, units, use_bias=True,
                 kernel_initializer="glorot_uniform", name=None):
        super().__init__(name=name)
        self.units = units
        self.use_bias = use_bias
        self._init = get_initializer(kernel_initializer)
        self.W = None
        self.b = None
        self._dW = None
        self._db = None

    def build(self, input_shape):
        n_in = input_shape[-1]
        self.W = self._init((n_in, 2 * self.units))
        if self.use_bias:
            self.b = np.zeros(2 * self.units)
        self._built = True

    def forward(self, x, training=False):
        if not self._built:
            self.build(x.shape)
        self._input = x
        z = x @ self.W
        if self.use_bias:
            z = z + self.b
        a, b = z[..., :self.units], z[..., self.units:]
        self._gate = 1.0 / (1.0 + np.exp(-b))     # sigmoid(b)
        self._a = a
        return a * self._gate

    def backward(self, grad):
        d_a = grad * self._gate
        d_gate = grad * self._a
        d_b = d_gate * self._gate * (1.0 - self._gate)
        d_z = np.concatenate([d_a, d_b], axis=-1)

        inp = self._input
        n = inp.shape[0]
        self._dW = inp.reshape(-1, inp.shape[-1]).T @ d_z.reshape(-1, 2 * self.units) / n
        if self.use_bias:
            self._db = d_z.reshape(-1, 2 * self.units).sum(axis=0) / n

        return d_z @ self.W.T

    @property
    def params(self):
        if self.W is None:
            return {}
        p = {"W": self.W}
        if self.use_bias:
            p["b"] = self.b
        return p

    @property
    def grads(self):
        if self._dW is None:
            return {}
        g = {"W": self._dW}
        if self.use_bias and self._db is not None:
            g["b"] = self._db
        return g

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"units": self.units, "use_bias": self.use_bias})
        return cfg


class SwiGLU(Layer):
    """
    SwiGLU — Swish-Gated Linear Unit (Shazeer 2020).

    Like :class:`GLU` but replaces the sigmoid gate with Swish (SiLU)::

        z = x @ W + bias
        a, b = z[..., :units], z[..., units:]
        output = a ⊙ swish(b)    where swish(b) = b * σ(b)

    SwiGLU is the standard FFN activation in **LLaMA, PaLM, Mistral,
    and Gemma**.  It outperforms ReLU and GELU on most language tasks
    despite having 50% more parameters than a plain Dense layer (offset
    by removing one bias in the original papers).

    Parameters
    ----------
    units : int
        Output dimensionality.
    use_bias : bool
        Whether to include a bias (default True).

    Examples
    --------
    Replace a standard FFN sub-layer::

        # Standard
        Dense(ff_dim, activation="gelu") → Dense(embed_dim)

        # SwiGLU drop-in (ff_dim must be 2/3 of original for equal params)
        SwiGLU(units=ff_dim) → Dense(embed_dim)
    """

    def __init__(self, units, use_bias=True,
                 kernel_initializer="glorot_uniform", name=None):
        super().__init__(name=name)
        self.units = units
        self.use_bias = use_bias
        self._init = get_initializer(kernel_initializer)
        self.W = None
        self.b = None
        self._dW = None
        self._db = None

    def build(self, input_shape):
        n_in = input_shape[-1]
        self.W = self._init((n_in, 2 * self.units))
        if self.use_bias:
            self.b = np.zeros(2 * self.units)
        self._built = True

    def forward(self, x, training=False):
        if not self._built:
            self.build(x.shape)
        self._input = x
        z = x @ self.W
        if self.use_bias:
            z = z + self.b
        a, b = z[..., :self.units], z[..., self.units:]
        self._sig = 1.0 / (1.0 + np.exp(-b))
        self._swish = b * self._sig          # swish(b)
        self._a = a
        return a * self._swish

    def backward(self, grad):
        d_a = grad * self._swish
        d_swish = grad * self._a
        # d/db [b * σ(b)] = swish(b) + σ(b)(1 − swish(b))
        d_b = d_swish * (self._swish + self._sig * (1.0 - self._swish))
        d_z = np.concatenate([d_a, d_b], axis=-1)

        inp = self._input
        n = inp.shape[0]
        self._dW = inp.reshape(-1, inp.shape[-1]).T @ d_z.reshape(-1, 2 * self.units) / n
        if self.use_bias:
            self._db = d_z.reshape(-1, 2 * self.units).sum(axis=0) / n

        return d_z @ self.W.T

    @property
    def params(self):
        if self.W is None:
            return {}
        p = {"W": self.W}
        if self.use_bias:
            p["b"] = self.b
        return p

    @property
    def grads(self):
        if self._dW is None:
            return {}
        g = {"W": self._dW}
        if self.use_bias and self._db is not None:
            g["b"] = self._db
        return g

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"units": self.units, "use_bias": self.use_bias})
        return cfg
