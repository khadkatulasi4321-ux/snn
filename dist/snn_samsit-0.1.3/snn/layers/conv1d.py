"""
conv1d.py — 1-D Convolutional layer and 1-D pooling.
"""
import numpy as np
from .base import Layer
from ..activations import get as get_activation
from ..initializers import get as get_initializer


class Conv1D(Layer):
    """
    1-D Convolution layer (channels-last: batch, length, channels).

    Slides a bank of ``filters`` kernels of width ``kernel_size`` along
    the time/sequence axis.  Useful for local pattern detection in
    sequences (text, audio, time-series).

    Parameters
    ----------
    filters : int
        Number of output channels (kernels).
    kernel_size : int
        Width of each convolutional kernel.
    stride : int
        Step between successive windows (default 1).
    padding : str
        ``"valid"`` — no padding, output shorter than input.
        ``"same"``  — zero-pad so output length equals input length
        (only when ``stride=1``).
    activation : str or Activation
        Applied after the convolution.
    use_bias : bool
        Whether to include a bias term.
    kernel_initializer : str
        Weight initialiser (default ``"glorot_uniform"``).

    Input / output shape
    --------------------
    ``(batch, length, in_channels)``
    → ``(batch, out_length, filters)``

    where ``out_length = (length + 2*pad - kernel_size) // stride + 1``.

    Examples
    --------
    >>> conv = Conv1D(filters=32, kernel_size=3, activation='relu')
    >>> x = np.random.randn(4, 50, 8)   # (batch, seq, channels)
    >>> out = conv.forward(x)           # (4, 48, 32) with valid padding
    """

    def __init__(self, filters, kernel_size, stride=1, padding="valid",
                 activation=None, use_bias=True,
                 kernel_initializer="glorot_uniform", name=None):
        super().__init__(name=name)
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding.lower()
        self.activation = get_activation(activation)
        self.use_bias = use_bias
        self._init = get_initializer(kernel_initializer)

        # W: (kernel_size, in_channels, filters)
        self.W = None
        self.b = None
        self._dW = None
        self._db = None
        self._input = None
        self._padded = None
        self._pad = 0

    def build(self, input_shape):
        in_ch = input_shape[-1]
        self.W = self._init((self.kernel_size, in_ch, self.filters))
        if self.use_bias:
            self.b = np.zeros(self.filters)
        self._built = True

    def _get_pad(self, length):
        if self.padding == "same":
            out = -(-length // self.stride)   # ceil div
            pad = max(0, (out - 1) * self.stride + self.kernel_size - length)
            return pad // 2, pad - pad // 2
        return 0, 0

    def forward(self, x, training=False):
        if not self._built:
            self.build(x.shape)
        self._input = x
        batch, length, in_ch = x.shape
        k = self.kernel_size

        pad_l, pad_r = self._get_pad(length)
        self._pad = (pad_l, pad_r)
        if pad_l > 0 or pad_r > 0:
            x = np.pad(x, [(0, 0), (pad_l, pad_r), (0, 0)])
        self._padded = x

        out_len = (x.shape[1] - k) // self.stride + 1

        # Vectorised im2col: collect all patches at once
        # patches: (batch, out_len, k, in_ch)
        patches = np.stack(
            [x[:, i * self.stride: i * self.stride + k, :]
             for i in range(out_len)],
            axis=1
        )
        self._patches = patches  # cache for backward

        # (batch, out_len, k*in_ch) @ (k*in_ch, filters) → (batch, out_len, filters)
        W_flat = self.W.reshape(-1, self.filters)
        out = patches.reshape(batch, out_len, -1) @ W_flat
        if self.use_bias:
            out = out + self.b
        return self.activation.forward(out)

    def backward(self, grad):
        grad = self.activation.backward(grad)
        batch, out_len, filters = grad.shape
        k = self.kernel_size
        in_ch = self.W.shape[1]

        W_flat = self.W.reshape(-1, filters)        # (k*in_ch, filters)
        patches_flat = self._patches.reshape(batch, out_len, -1)  # (batch, out_len, k*in_ch)

        # dW: (k*in_ch, filters)
        dW_flat = np.einsum('boi,bof->if', patches_flat, grad) / batch
        self._dW = dW_flat.reshape(self.W.shape)

        if self.use_bias:
            self._db = grad.sum(axis=(0, 1)) / batch

        # dx_padded: (batch, padded_len, in_ch)
        padded_len = self._padded.shape[1]
        dx_padded = np.zeros((batch, padded_len, in_ch))

        # Scatter gradients back
        dx_patches = grad @ W_flat.T                # (batch, out_len, k*in_ch)
        dx_patches = dx_patches.reshape(batch, out_len, k, in_ch)

        for i in range(out_len):
            dx_padded[:, i * self.stride: i * self.stride + k, :] += dx_patches[:, i, :, :]

        # Remove padding
        pl, pr = self._pad
        if pl > 0 or pr > 0:
            dx = dx_padded[:, pl: padded_len - pr, :]
        else:
            dx = dx_padded

        return dx

    @property
    def params(self):
        p = {"W": self.W}
        if self.use_bias:
            p["b"] = self.b
        return p if self.W is not None else {}

    @property
    def grads(self):
        g = {"W": self._dW}
        if self.use_bias and self._db is not None:
            g["b"] = self._db
        return g if self._dW is not None else {}

    def get_config(self):
        cfg = super().get_config()
        cfg.update(dict(filters=self.filters, kernel_size=self.kernel_size,
                        stride=self.stride, padding=self.padding))
        return cfg


class MaxPooling1D(Layer):
    """
    1-D Max Pooling over the time/sequence axis.

    Parameters
    ----------
    pool_size : int
        Size of the pooling window (default 2).
    stride : int
        Step between windows (default equals ``pool_size``).

    Input / output shape
    --------------------
    ``(batch, length, channels)`` → ``(batch, out_length, channels)``
    """

    def __init__(self, pool_size=2, stride=None, name=None):
        super().__init__(name=name)
        self.pool_size = pool_size
        self.stride = stride if stride is not None else pool_size

    def forward(self, x, training=False):
        self._input = x
        batch, length, ch = x.shape
        p, s = self.pool_size, self.stride
        out_len = (length - p) // s + 1

        out = np.zeros((batch, out_len, ch))
        self._mask = np.zeros_like(x, dtype=bool)

        for i in range(out_len):
            patch = x[:, i * s: i * s + p, :]
            max_val = patch.max(axis=1, keepdims=True)
            out[:, i, :] = max_val[:, 0, :]
            self._mask[:, i * s: i * s + p, :] |= (patch == max_val)

        return out

    def backward(self, grad):
        dx = np.zeros_like(self._input)
        p, s = self.pool_size, self.stride
        out_len = grad.shape[1]

        for i in range(out_len):
            mask_patch = self._mask[:, i * s: i * s + p, :]
            g = grad[:, i, :][:, np.newaxis, :]
            dx[:, i * s: i * s + p, :] += mask_patch * g

        return dx


class AveragePooling1D(Layer):
    """
    1-D Average Pooling over the time/sequence axis.

    Parameters
    ----------
    pool_size : int
        Width of each pooling window (default 2).
    stride : int
        Step between windows (default equals ``pool_size``).
    """

    def __init__(self, pool_size=2, stride=None, name=None):
        super().__init__(name=name)
        self.pool_size = pool_size
        self.stride = stride if stride is not None else pool_size

    def forward(self, x, training=False):
        self._input_shape = x.shape
        batch, length, ch = x.shape
        p, s = self.pool_size, self.stride
        out_len = (length - p) // s + 1

        out = np.zeros((batch, out_len, ch))
        for i in range(out_len):
            out[:, i, :] = x[:, i * s: i * s + p, :].mean(axis=1)
        return out

    def backward(self, grad):
        dx = np.zeros(self._input_shape)
        p, s = self.pool_size, self.stride
        out_len = grad.shape[1]

        for i in range(out_len):
            g = grad[:, i, :][:, np.newaxis, :]
            dx[:, i * s: i * s + p, :] += g / p

        return dx
