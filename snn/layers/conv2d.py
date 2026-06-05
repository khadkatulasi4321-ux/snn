import numpy as np
from .base import Layer
from ..activations import get as get_activation
from ..initializers import get as get_initializer


def _im2col(x, fh, fw, stride, pad):
    n, h, w, c = x.shape
    out_h = (h + 2 * pad - fh) // stride + 1
    out_w = (w + 2 * pad - fw) // stride + 1

    x_pad = np.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode="constant")
    col = np.zeros((n, out_h, out_w, fh, fw, c), dtype=x.dtype)

    for i in range(fh):
        i_max = i + stride * out_h
        for j in range(fw):
            j_max = j + stride * out_w
            col[:, :, :, i, j, :] = x_pad[:, i:i_max:stride, j:j_max:stride, :]

    col = col.reshape(n * out_h * out_w, fh * fw * c)
    return col, out_h, out_w


def _col2im(col, x_shape, fh, fw, stride, pad):
    n, h, w, c = x_shape
    out_h = (h + 2 * pad - fh) // stride + 1
    out_w = (w + 2 * pad - fw) // stride + 1

    col = col.reshape(n, out_h, out_w, fh, fw, c)
    x_pad = np.zeros((n, h + 2 * pad, w + 2 * pad, c), dtype=col.dtype)

    for i in range(fh):
        i_max = i + stride * out_h
        for j in range(fw):
            j_max = j + stride * out_w
            x_pad[:, i:i_max:stride, j:j_max:stride, :] += col[:, :, :, i, j, :]

    if pad == 0:
        return x_pad
    return x_pad[:, pad:-pad, pad:-pad, :]


class Conv2D(Layer):
    """
    2D Convolutional layer.

    Expects input shape: (N, H, W, C_in)  — channels-last convention.

    Parameters
    ----------
    filters : int
        Number of output filters (channels).
    kernel_size : int or (int, int)
        Size of the convolution kernel.
    stride : int
        Stride of the convolution.
    padding : str or int
        'same', 'valid', or an explicit integer pad.
    activation : str or Activation
        Applied after convolution + bias.
    use_bias : bool
    kernel_initializer : str or callable
    bias_initializer : str or callable
    """

    def __init__(self, filters, kernel_size=3, stride=1, padding="valid",
                 activation=None, use_bias=True,
                 kernel_initializer="he_normal",
                 bias_initializer="zeros",
                 name=None):
        super().__init__(name=name)
        self.filters = filters
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        self.stride = stride
        self.padding = padding
        self.activation = get_activation(activation)
        self.use_bias = use_bias
        self.kernel_initializer = get_initializer(kernel_initializer)
        self.bias_initializer = get_initializer(bias_initializer)

        self.W = None
        self.b = None
        self._dW = None
        self._db = None
        self._col = None
        self._input_shape = None

    def _get_pad(self, h, w):
        if self.padding == "valid":
            return 0
        if self.padding == "same":
            fh, fw = self.kernel_size
            pad_h = max((h - 1) * self.stride + fh - h, 0) // 2
            pad_w = max((w - 1) * self.stride + fw - w, 0) // 2
            return max(pad_h, pad_w)
        return int(self.padding)

    def build(self, input_shape):
        _, _, _, c_in = input_shape
        fh, fw = self.kernel_size
        self.W = self.kernel_initializer((fh, fw, c_in, self.filters))
        if self.use_bias:
            self.b = self.bias_initializer((1, self.filters))
        self._built = True

    def forward(self, x, training=False):
        if not self._built:
            self.build(x.shape)
        n, h, w, c = x.shape
        fh, fw = self.kernel_size
        pad = self._get_pad(h, w)
        self._input_shape = x.shape
        self._pad = pad

        col, out_h, out_w = _im2col(x, fh, fw, self.stride, pad)
        self._col = col

        W_col = self.W.reshape(-1, self.filters)
        out = col @ W_col
        if self.use_bias:
            out = out + self.b
        out = out.reshape(n, out_h, out_w, self.filters)
        return self.activation.forward(out)

    def backward(self, grad):
        n, out_h, out_w, _ = grad.shape
        fh, fw = self.kernel_size
        grad = self.activation.backward(grad)
        grad_col = grad.reshape(-1, self.filters)

        W_col = self.W.reshape(-1, self.filters)
        self._dW = (self._col.T @ grad_col).reshape(self.W.shape) / n
        if self.use_bias:
            self._db = np.sum(grad_col, axis=0, keepdims=True) / n

        dx_col = grad_col @ W_col.T
        dx = _col2im(dx_col, self._input_shape, fh, fw, self.stride, self._pad)
        return dx

    @property
    def params(self):
        p = {"W": self.W}
        if self.use_bias:
            p["b"] = self.b
        return p

    @property
    def grads(self):
        g = {}
        if self._dW is not None:
            g["W"] = self._dW
        if self.use_bias and self._db is not None:
            g["b"] = self._db
        return g

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "padding": self.padding,
            "use_bias": self.use_bias,
        })
        return cfg
