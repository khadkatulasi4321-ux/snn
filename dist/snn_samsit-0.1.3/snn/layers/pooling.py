import numpy as np
from .base import Layer


class MaxPooling2D(Layer):
    """
    2D Max Pooling layer (channels-last: N, H, W, C).
    """

    def __init__(self, pool_size=2, stride=None, name=None):
        super().__init__(name=name)
        self.pool_size = (pool_size, pool_size) if isinstance(pool_size, int) else tuple(pool_size)
        self.stride = stride if stride is not None else self.pool_size

    def forward(self, x, training=False):
        n, h, w, c = x.shape
        ph, pw = self.pool_size
        sh, sw = (self.stride, self.stride) if isinstance(self.stride, int) else self.stride
        out_h = (h - ph) // sh + 1
        out_w = (w - pw) // sw + 1

        self._input = x
        self._input_shape = x.shape
        out = np.zeros((n, out_h, out_w, c))
        self._max_mask = np.zeros_like(x, dtype=bool)

        for i in range(out_h):
            for j in range(out_w):
                patch = x[:, i*sh:i*sh+ph, j*sw:j*sw+pw, :]
                max_val = np.max(patch, axis=(1, 2), keepdims=True)
                out[:, i, j, :] = max_val[:, 0, 0, :]
                mask = (patch == max_val)
                self._max_mask[:, i*sh:i*sh+ph, j*sw:j*sw+pw, :] |= mask

        return out

    def backward(self, grad):
        n, h, w, c = self._input_shape
        ph, pw = self.pool_size
        sh, sw = (self.stride, self.stride) if isinstance(self.stride, int) else self.stride
        out_h, out_w = grad.shape[1], grad.shape[2]
        dx = np.zeros(self._input_shape)

        for i in range(out_h):
            for j in range(out_w):
                patch_mask = self._max_mask[:, i*sh:i*sh+ph, j*sw:j*sw+pw, :]
                g = grad[:, i, j, :][:, np.newaxis, np.newaxis, :]
                dx[:, i*sh:i*sh+ph, j*sw:j*sw+pw, :] += patch_mask * g

        return dx


class AveragePooling2D(Layer):
    """
    2D Average Pooling layer (channels-last: N, H, W, C).
    """

    def __init__(self, pool_size=2, stride=None, name=None):
        super().__init__(name=name)
        self.pool_size = (pool_size, pool_size) if isinstance(pool_size, int) else tuple(pool_size)
        self.stride = stride if stride is not None else self.pool_size

    def forward(self, x, training=False):
        n, h, w, c = x.shape
        ph, pw = self.pool_size
        sh, sw = (self.stride, self.stride) if isinstance(self.stride, int) else self.stride
        out_h = (h - ph) // sh + 1
        out_w = (w - pw) // sw + 1

        self._input_shape = x.shape
        out = np.zeros((n, out_h, out_w, c))

        for i in range(out_h):
            for j in range(out_w):
                out[:, i, j, :] = np.mean(x[:, i*sh:i*sh+ph, j*sw:j*sw+pw, :], axis=(1, 2))

        return out

    def backward(self, grad):
        ph, pw = self.pool_size
        sh, sw = (self.stride, self.stride) if isinstance(self.stride, int) else self.stride
        out_h, out_w = grad.shape[1], grad.shape[2]
        dx = np.zeros(self._input_shape)
        pool_area = ph * pw

        for i in range(out_h):
            for j in range(out_w):
                g = grad[:, i, j, :][:, np.newaxis, np.newaxis, :]
                dx[:, i*sh:i*sh+ph, j*sw:j*sw+pw, :] += g / pool_area

        return dx


class GlobalAveragePooling2D(Layer):
    """
    Global Average Pooling — reduces (N, H, W, C) to (N, C).
    """

    def forward(self, x, training=False):
        self._input_shape = x.shape
        return np.mean(x, axis=(1, 2))

    def backward(self, grad):
        n, h, w, c = self._input_shape
        return np.broadcast_to(
            grad[:, np.newaxis, np.newaxis, :] / (h * w),
            self._input_shape
        ).copy()


class GlobalMaxPooling2D(Layer):
    """
    Global Max Pooling — reduces (N, H, W, C) to (N, C).
    """

    def forward(self, x, training=False):
        self._input_shape = x.shape
        self._input = x
        out = np.max(x, axis=(1, 2))
        self._max_mask = (x == out[:, np.newaxis, np.newaxis, :])
        return out

    def backward(self, grad):
        return self._max_mask * grad[:, np.newaxis, np.newaxis, :]


class GlobalAveragePooling1D(Layer):
    """
    Global Average Pooling over the time/sequence axis.

    Reduces ``(batch, seq_len, features)`` to ``(batch, features)`` by
    averaging over ``seq_len``.  Standard way to collapse a Transformer or
    LSTM output into a fixed-size representation for classification.

    Examples
    --------
    >>> gap = GlobalAveragePooling1D()
    >>> x = np.random.randn(8, 20, 64)   # (batch, seq, features)
    >>> out = gap.forward(x)             # (8, 64)
    """

    def forward(self, x, training=False):
        self._input_shape = x.shape
        return np.mean(x, axis=1)           # (batch, features)

    def backward(self, grad):
        _, seq, _ = self._input_shape
        return np.broadcast_to(
            grad[:, np.newaxis, :] / seq, self._input_shape
        ).copy()


class GlobalMaxPooling1D(Layer):
    """
    Global Max Pooling over the time/sequence axis.

    Reduces ``(batch, seq_len, features)`` to ``(batch, features)`` by
    taking the maximum over ``seq_len``.

    Examples
    --------
    >>> gmp = GlobalMaxPooling1D()
    >>> x = np.random.randn(8, 20, 64)
    >>> out = gmp.forward(x)    # (8, 64)
    """

    def forward(self, x, training=False):
        self._input_shape = x.shape
        self._input = x
        out = np.max(x, axis=1)              # (batch, features)
        self._max_mask = (x == out[:, np.newaxis, :])
        return out

    def backward(self, grad):
        return self._max_mask * grad[:, np.newaxis, :]
