import numpy as np
from .base import Layer


class Flatten(Layer):
    """
    Flattens input to (N, -1). Keeps the batch dimension.
    """

    def forward(self, x, training=False):
        self._input_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, grad):
        return grad.reshape(self._input_shape)


class Reshape(Layer):
    """
    Reshapes input (excluding batch dim) to a target shape.

    Parameters
    ----------
    target_shape : tuple
        New shape for each sample (not including batch size).
    """

    def __init__(self, target_shape, name=None):
        super().__init__(name=name)
        self.target_shape = tuple(target_shape)

    def forward(self, x, training=False):
        self._input_shape = x.shape
        return x.reshape((x.shape[0],) + self.target_shape)

    def backward(self, grad):
        return grad.reshape(self._input_shape)

    def get_config(self):
        cfg = super().get_config()
        cfg["target_shape"] = self.target_shape
        return cfg
