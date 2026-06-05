import numpy as np
from .base import Layer
from ..activations import get as get_activation
from ..initializers import get as get_initializer


class Dense(Layer):
    """
    Fully-connected (dense) layer.

    Parameters
    ----------
    units : int
        Number of output neurons.
    activation : str or Activation, optional
        Activation function applied after the linear transform.
    use_bias : bool
        Whether to include a bias vector.
    kernel_initializer : str or callable
        Initializer for the weight matrix W.
    bias_initializer : str or callable
        Initializer for the bias vector b.
    """

    def __init__(self, units, activation=None, use_bias=True,
                 kernel_initializer="glorot_uniform",
                 bias_initializer="zeros",
                 name=None):
        super().__init__(name=name)
        self.units = units
        self.activation = get_activation(activation)
        self.use_bias = use_bias
        self.kernel_initializer = get_initializer(kernel_initializer)
        self.bias_initializer = get_initializer(bias_initializer)

        self.W = None
        self.b = None
        self._dW = None
        self._db = None
        self._input = None

    def build(self, input_shape):
        n_in = input_shape[-1]
        self.W = self.kernel_initializer((n_in, self.units))
        if self.use_bias:
            self.b = self.bias_initializer((1, self.units))
        self._built = True

    def forward(self, x, training=False):
        if not self._built:
            self.build(x.shape)
        self._input = x
        z = x @ self.W
        if self.use_bias:
            z = z + self.b
        return self.activation.forward(z)

    def backward(self, grad):
        grad = self.activation.backward(grad)
        n = self._input.shape[0]
        # Support both 2D (batch, features) and 3D (batch, seq, features) inputs.
        # Flatten all leading dims before computing weight gradients, then restore.
        inp_flat = self._input.reshape(-1, self._input.shape[-1])
        grad_flat = grad.reshape(-1, grad.shape[-1])
        self._dW = inp_flat.T @ grad_flat / n
        if self.use_bias:
            self._db = np.sum(grad_flat, axis=0, keepdims=True) / n
        dx = grad @ self.W.T          # preserves original shape (2D or 3D)
        return dx

    @property
    def params(self):
        p = {"W": self.W}
        if self.use_bias:
            p["b"] = self.b
        return p

    @property
    def grads(self):
        g = {"W": self._dW}
        if self.use_bias and self._db is not None:
            g["b"] = self._db
        return g

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "units": self.units,
            "use_bias": self.use_bias,
        })
        return cfg
