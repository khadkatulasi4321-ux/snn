import numpy as np
from .base import Layer


class BatchNormalization(Layer):
    """
    Batch Normalization layer.

    Normalizes activations over the batch dimension. During training uses
    batch statistics; during inference uses running (exponential moving
    average) statistics.

    Parameters
    ----------
    axis : int
        Axis to normalize (usually the features/channels axis).
        For Dense output (N, F) use axis=-1.
        For Conv2D output (N, H, W, C) use axis=-1.
    momentum : float
        Momentum for the moving average (typically 0.99).
    epsilon : float
        Small constant for numerical stability.
    center : bool
        If True, add learnable beta offset.
    scale : bool
        If True, multiply by learnable gamma scale.
    """

    def __init__(self, axis=-1, momentum=0.99, epsilon=1e-5,
                 center=True, scale=True, name=None):
        super().__init__(name=name)
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale

        self.gamma = None
        self.beta = None
        self._running_mean = None
        self._running_var = None

        self._cache = None
        self._dgamma = None
        self._dbeta = None

    def build(self, input_shape):
        size = input_shape[self.axis]
        self.gamma = np.ones(size, dtype=np.float64)
        self.beta = np.zeros(size, dtype=np.float64)
        self._running_mean = np.zeros(size, dtype=np.float64)
        self._running_var = np.ones(size, dtype=np.float64)
        self._built = True

    def forward(self, x, training=False):
        if not self._built:
            self.build(x.shape)

        axis = self.axis if self.axis != -1 else len(x.shape) - 1
        reduce_axes = tuple(i for i in range(len(x.shape)) if i != axis)

        if training:
            mu = np.mean(x, axis=reduce_axes)
            var = np.var(x, axis=reduce_axes)
            self._running_mean = (self.momentum * self._running_mean +
                                  (1.0 - self.momentum) * mu)
            self._running_var = (self.momentum * self._running_var +
                                 (1.0 - self.momentum) * var)
        else:
            mu = self._running_mean
            var = self._running_var

        shape = [1] * len(x.shape)
        shape[axis] = -1

        mu_b = mu.reshape(shape)
        var_b = var.reshape(shape)

        x_norm = (x - mu_b) / np.sqrt(var_b + self.epsilon)

        gamma_b = self.gamma.reshape(shape) if self.scale else 1.0
        beta_b = self.beta.reshape(shape) if self.center else 0.0

        out = gamma_b * x_norm + beta_b

        if training:
            n = np.prod([x.shape[i] for i in reduce_axes])
            self._cache = (x, x_norm, mu, var, n, axis, reduce_axes, shape)

        return out

    def backward(self, grad):
        x, x_norm, mu, var, n, axis, reduce_axes, shape = self._cache

        gamma_b = self.gamma.reshape(shape) if self.scale else 1.0

        if self.scale:
            self._dgamma = np.sum(grad * x_norm, axis=reduce_axes)
        if self.center:
            self._dbeta = np.sum(grad, axis=reduce_axes)

        dx_norm = grad * gamma_b
        std_inv = 1.0 / np.sqrt(var.reshape(shape) + self.epsilon)

        dx = (1.0 / n) * std_inv * (
            n * dx_norm
            - np.sum(dx_norm, axis=reduce_axes, keepdims=True)
            - x_norm * np.sum(dx_norm * x_norm, axis=reduce_axes, keepdims=True)
        )
        return dx

    @property
    def params(self):
        p = {}
        if self.scale:
            p["gamma"] = self.gamma
        if self.center:
            p["beta"] = self.beta
        return p

    @property
    def grads(self):
        g = {}
        if self.scale and self._dgamma is not None:
            g["gamma"] = self._dgamma
        if self.center and self._dbeta is not None:
            g["beta"] = self._dbeta
        return g

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "axis": self.axis,
            "momentum": self.momentum,
            "epsilon": self.epsilon,
            "center": self.center,
            "scale": self.scale,
        })
        return cfg


class LayerNormalization(Layer):
    """
    Layer Normalization — normalizes over the last axis (feature axis).
    """

    def __init__(self, epsilon=1e-6, center=True, scale=True, name=None):
        super().__init__(name=name)
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.gamma = None
        self.beta = None
        self._cache = None
        self._dgamma = None
        self._dbeta = None

    def build(self, input_shape):
        size = input_shape[-1]
        self.gamma = np.ones(size, dtype=np.float64)
        self.beta = np.zeros(size, dtype=np.float64)
        self._built = True

    def forward(self, x, training=False):
        if not self._built:
            self.build(x.shape)
        mu = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mu) / np.sqrt(var + self.epsilon)
        out = x_norm
        if self.scale:
            out = out * self.gamma
        if self.center:
            out = out + self.beta
        self._cache = (x_norm, var)
        return out

    def backward(self, grad):
        x_norm, var = self._cache
        n = x_norm.shape[-1]
        if self.scale:
            self._dgamma = np.sum(grad * x_norm, axis=tuple(range(len(grad.shape) - 1)))
        if self.center:
            self._dbeta = np.sum(grad, axis=tuple(range(len(grad.shape) - 1)))
        gamma = self.gamma if self.scale else 1.0
        dx_norm = grad * gamma
        std_inv = 1.0 / np.sqrt(var + self.epsilon)
        dx = std_inv / n * (
            n * dx_norm
            - np.sum(dx_norm, axis=-1, keepdims=True)
            - x_norm * np.sum(dx_norm * x_norm, axis=-1, keepdims=True)
        )
        return dx

    @property
    def params(self):
        p = {}
        if self.scale:
            p["gamma"] = self.gamma
        if self.center:
            p["beta"] = self.beta
        return p

    @property
    def grads(self):
        g = {}
        if self.scale and self._dgamma is not None:
            g["gamma"] = self._dgamma
        if self.center and self._dbeta is not None:
            g["beta"] = self._dbeta
        return g
