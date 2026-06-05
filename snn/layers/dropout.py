import numpy as np
from .base import Layer


class Dropout(Layer):
    """
    Dropout regularization layer.

    During training, randomly sets a fraction `rate` of inputs to zero
    and scales the remaining values by 1 / (1 - rate) (inverted dropout).
    During inference, the layer is a pass-through.

    Parameters
    ----------
    rate : float
        Fraction of units to drop (0 <= rate < 1).
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(self, rate=0.5, seed=None, name=None):
        super().__init__(name=name)
        if not 0.0 <= rate < 1.0:
            raise ValueError(f"Dropout rate must be in [0, 1). Got {rate}.")
        self.rate = rate
        self._rng = np.random.default_rng(seed)
        self._mask = None

    def forward(self, x, training=False):
        if not training or self.rate == 0.0:
            return x
        keep_prob = 1.0 - self.rate
        self._mask = (self._rng.random(x.shape) < keep_prob) / keep_prob
        return x * self._mask

    def backward(self, grad):
        if self._mask is None:
            return grad
        return grad * self._mask

    def get_config(self):
        cfg = super().get_config()
        cfg["rate"] = self.rate
        return cfg


class SpatialDropout2D(Layer):
    """
    Spatial Dropout for 2D feature maps (N, H, W, C).

    Drops entire feature maps (channels) instead of individual elements.

    Parameters
    ----------
    rate : float
        Fraction of feature maps to drop.
    seed : int, optional
    """

    def __init__(self, rate=0.5, seed=None, name=None):
        super().__init__(name=name)
        self.rate = rate
        self._rng = np.random.default_rng(seed)
        self._mask = None

    def forward(self, x, training=False):
        if not training or self.rate == 0.0:
            return x
        n, h, w, c = x.shape
        keep_prob = 1.0 - self.rate
        channel_mask = (self._rng.random((n, 1, 1, c)) < keep_prob) / keep_prob
        self._mask = channel_mask
        return x * channel_mask

    def backward(self, grad):
        if self._mask is None:
            return grad
        return grad * self._mask

    def get_config(self):
        cfg = super().get_config()
        cfg["rate"] = self.rate
        return cfg
