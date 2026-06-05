import numpy as np


class Activation:
    """Base class for all activation functions."""

    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError

    def __call__(self, x):
        return self.forward(x)


# ---------------------------------------------------------------------------
# Existing activations
# ---------------------------------------------------------------------------

class Linear(Activation):
    """Identity / no-op activation. f(x) = x."""

    def forward(self, x):
        self._input = x
        return x

    def backward(self, grad):
        return grad


class ReLU(Activation):
    """Rectified Linear Unit. f(x) = max(0, x)."""

    def forward(self, x):
        self._mask = x > 0
        return np.where(self._mask, x, 0.0)

    def backward(self, grad):
        return grad * self._mask


class LeakyReLU(Activation):
    """Leaky ReLU. f(x) = x if x > 0 else alpha*x.

    Parameters
    ----------
    alpha : float
        Slope for negative inputs (default 0.01).
    """

    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def forward(self, x):
        self._input = x
        return np.where(x > 0, x, self.alpha * x)

    def backward(self, grad):
        dx = np.where(self._input > 0, 1.0, self.alpha)
        return grad * dx


class ELU(Activation):
    """Exponential Linear Unit. f(x) = x if x > 0 else alpha*(exp(x)-1).

    Parameters
    ----------
    alpha : float
        Scale for negative saturation region (default 1.0).
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def forward(self, x):
        self._input = x
        return np.where(x > 0, x, self.alpha * (np.exp(np.clip(x, -500, 0)) - 1.0))

    def backward(self, grad):
        out = np.where(self._input > 0, 1.0,
                       self.alpha * np.exp(np.clip(self._input, -500, 0)))
        return grad * out


class SELU(Activation):
    """Scaled ELU — self-normalising activation for deep networks."""

    _ALPHA = 1.6732632423543772
    _SCALE = 1.0507009873554805

    def forward(self, x):
        self._input = x
        return self._SCALE * np.where(x > 0, x, self._ALPHA * (np.exp(np.clip(x, -500, 0)) - 1.0))

    def backward(self, grad):
        dx = self._SCALE * np.where(self._input > 0, 1.0,
                                     self._ALPHA * np.exp(np.clip(self._input, -500, 0)))
        return grad * dx


class Sigmoid(Activation):
    """Logistic sigmoid. f(x) = 1 / (1 + exp(-x))."""

    def forward(self, x):
        self._out = 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        return self._out

    def backward(self, grad):
        return grad * self._out * (1.0 - self._out)


class Tanh(Activation):
    """Hyperbolic tangent. f(x) = tanh(x)."""

    def forward(self, x):
        self._out = np.tanh(x)
        return self._out

    def backward(self, grad):
        return grad * (1.0 - self._out ** 2)


class Softmax(Activation):
    """Softmax over the last axis. Typically used on the output layer for
    multi-class classification."""

    def forward(self, x):
        shifted = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(shifted)
        self._out = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        return self._out

    def backward(self, grad):
        batch_size = self._out.shape[0]
        dx = np.empty_like(grad)
        for i in range(batch_size):
            s = self._out[i].reshape(-1, 1)
            jacobian = np.diagflat(s) - s @ s.T
            dx[i] = jacobian @ grad[i]
        return dx


class Softplus(Activation):
    """Smooth approximation of ReLU. f(x) = log(1 + exp(x))."""

    def forward(self, x):
        self._input = x
        return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

    def backward(self, grad):
        return grad / (1.0 + np.exp(-self._input))


class Swish(Activation):
    """Swish / SiLU. f(x) = x * sigmoid(x). Smooth, non-monotonic."""

    def forward(self, x):
        self._input = x
        self._sig = 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        return x * self._sig

    def backward(self, grad):
        swish = self._input * self._sig
        return grad * (swish + self._sig * (1.0 - swish))


class Mish(Activation):
    """Mish. f(x) = x * tanh(softplus(x)). Self-regularising, smooth."""

    def forward(self, x):
        self._input = x
        sp = np.log1p(np.exp(np.clip(x, -500, 500)))
        self._tanh_sp = np.tanh(sp)
        return x * self._tanh_sp

    def backward(self, grad):
        sp = np.log1p(np.exp(np.clip(self._input, -500, 500)))
        sig = 1.0 / (1.0 + np.exp(-np.clip(self._input, -500, 500)))
        tanh_sp = self._tanh_sp
        dtanh = 1.0 - tanh_sp ** 2
        dsp = sig
        return grad * (tanh_sp + self._input * dtanh * dsp)


# ---------------------------------------------------------------------------
# New activations — better for smooth / non-linear function approximation
# ---------------------------------------------------------------------------

class GELU(Activation):
    """Gaussian Error Linear Unit.

    Uses the tanh approximation:
    f(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))

    The standard choice in modern Transformers (BERT, GPT). Smooth and
    non-monotonic; outperforms ReLU/ELU on many regression and NLP tasks.
    """

    _K = np.sqrt(2.0 / np.pi)
    _C = 0.044715

    def forward(self, x):
        self._input = x
        inner = self._K * (x + self._C * x ** 3)
        self._tanh_inner = np.tanh(inner)
        self._cdf = 0.5 * (1.0 + self._tanh_inner)
        return x * self._cdf

    def backward(self, grad):
        x = self._input
        inner = self._K * (x + self._C * x ** 3)
        d_inner = self._K * (1.0 + 3.0 * self._C * x ** 2)
        d_tanh = 1.0 - self._tanh_inner ** 2
        dcdf = 0.5 * d_tanh * d_inner
        return grad * (self._cdf + x * dcdf)


class PReLU(Activation):
    """Parametric ReLU with a **learnable** negative slope.

    f(x) = x          if x ≥ 0
    f(x) = alpha * x  if x < 0

    ``alpha`` is updated by the optimiser just like any weight — pass the
    layer that contains this activation through a ``Sequential`` model and
    ``alpha`` will be trained automatically.

    Parameters
    ----------
    alpha : float
        Initial slope for the negative region (default 0.25).
    """

    def __init__(self, alpha=0.25):
        self.alpha = np.array(alpha, dtype=np.float64)
        self._d_alpha = None
        self._input = None

    def forward(self, x):
        self._input = x
        return np.where(x >= 0, x, self.alpha * x)

    def backward(self, grad):
        x = self._input
        neg_mask = x < 0
        self._d_alpha = np.array(np.sum(grad * np.where(neg_mask, x, 0.0)))
        dx = np.where(x >= 0, 1.0, self.alpha) * grad
        return dx

    @property
    def params(self):
        return {"alpha": self.alpha}

    @property
    def grads(self):
        if self._d_alpha is None:
            return {}
        return {"alpha": self._d_alpha}


class Sine(Activation):
    """Sine activation for **SIREN** (Sinusoidal Representation Networks).

    f(x) = sin(x)

    Designed for learning smooth, periodic, and polynomial-like functions.
    A network with sine activations can exactly represent derivatives of
    itself and naturally approximates functions like y = x², y = sin(x),
    or any smooth manifold.

    **Initialisation note:** use ``kernel_initializer="siren_uniform"``
    (or scale weights by 1/fan_in for hidden layers) for best convergence.
    As a quick start, ``glorot_uniform`` works for shallow networks.
    """

    def forward(self, x):
        self._input = x
        return np.sin(x)

    def backward(self, grad):
        return grad * np.cos(self._input)


class Hardswish(Activation):
    """Hard Swish activation (MobileNetV3).

    f(x) = x * ReLU6(x + 3) / 6

    Piecewise-smooth approximation of Swish. Efficient for deployment while
    retaining the non-monotonic character of Swish.
    """

    def forward(self, x):
        self._input = x
        return x * np.clip(x + 3.0, 0.0, 6.0) / 6.0

    def backward(self, grad):
        x = self._input
        relu6 = np.clip(x + 3.0, 0.0, 6.0)
        d_relu6 = ((x + 3.0 > 0) & (x + 3.0 < 6.0)).astype(float)
        return grad * (relu6 + x * d_relu6) / 6.0


class BentIdentity(Activation):
    """Bent Identity activation.

    f(x) = (√(x² + 1) − 1) / 2 + x

    Smooth everywhere, non-saturating, and non-linear at every point.
    The derivative is always positive (≥ 1) so it avoids the dying-neuron
    problem while still introducing the curvature needed to fit polynomial
    or quadratic data.  Good default for regression tasks.
    """

    def forward(self, x):
        self._input = x
        self._sq = np.sqrt(x ** 2 + 1.0)
        return (self._sq - 1.0) / 2.0 + x

    def backward(self, grad):
        return grad * (self._input / (2.0 * self._sq) + 1.0)


class Squareplus(Activation):
    """Squareplus — smooth, monotonic ReLU alternative.

    f(x) = (x + √(x² + b)) / 2

    Always positive and differentiable, with quadratic behaviour near the
    origin controlled by ``b``. Approaches ReLU as b → 0.

    Parameters
    ----------
    b : float
        Curvature at origin (default 4.0). Larger values = smoother transition.
    """

    def __init__(self, b=4.0):
        self.b = b

    def forward(self, x):
        self._input = x
        self._sq = np.sqrt(x ** 2 + self.b)
        return (x + self._sq) / 2.0

    def backward(self, grad):
        return grad * (1.0 + self._input / self._sq) / 2.0


# ---------------------------------------------------------------------------
# Registry & factory
# ---------------------------------------------------------------------------

class ReLU6(Activation):
    """ReLU capped at 6. f(x) = min(max(0, x), 6).

    Used in MobileNet and other efficient architectures.  The cap at 6
    improves fixed-point quantisation precision.
    """

    def forward(self, x):
        self._input = x
        return np.clip(x, 0.0, 6.0)

    def backward(self, grad):
        mask = (self._input > 0) & (self._input < 6)
        return grad * mask


class Hardsigmoid(Activation):
    """Piecewise-linear sigmoid approximation.

    f(x) = clip((x + 3) / 6, 0, 1)

    Computationally cheaper than the exact sigmoid; commonly used in
    quantisation-friendly networks (MobileNetV3).
    """

    def forward(self, x):
        self._input = x
        return np.clip((x + 3.0) / 6.0, 0.0, 1.0)

    def backward(self, grad):
        mask = (self._input > -3.0) & (self._input < 3.0)
        return grad * mask / 6.0


class LogSoftmax(Activation):
    """Log-softmax. f(x) = x − log(Σ exp(x)).

    Numerically stable; used together with negative log-likelihood loss
    (equivalent to ``CrossEntropy = NLLLoss(LogSoftmax(x))``).
    """

    def forward(self, x):
        shifted = x - np.max(x, axis=-1, keepdims=True)
        log_sum = np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True) + 1e-9)
        self._log_sm = shifted - log_sum
        return self._log_sm

    def backward(self, grad):
        sm = np.exp(self._log_sm)
        return grad - sm * grad.sum(axis=-1, keepdims=True)


class Sparsemax(Activation):
    """Sparsemax — differentiable sparse softmax (Martins & Astudillo 2016).

    Like softmax but returns a *sparse* probability distribution: many
    outputs are exactly 0.  Useful for attention mechanisms where hard,
    interpretable focus is desired.

    f(x) = max(0, x − τ(x))  where τ is the threshold that makes outputs
    sum to 1.
    """

    def forward(self, x):
        z = x - np.max(x, axis=-1, keepdims=True)
        z_sorted = np.sort(z, axis=-1)[:, ::-1]
        k = np.arange(1, x.shape[-1] + 1)
        cumsum = np.cumsum(z_sorted, axis=-1)
        support = z_sorted > (cumsum - 1.0) / k
        k_max = support.sum(axis=-1, keepdims=True)
        tau = (np.take_along_axis(cumsum, k_max - 1, axis=-1) - 1.0) / k_max
        self._out = np.maximum(0.0, z - tau)
        return self._out

    def backward(self, grad):
        support = self._out > 0
        n_support = support.sum(axis=-1, keepdims=True).astype(float)
        mean_grad = (grad * support).sum(axis=-1, keepdims=True) / np.maximum(n_support, 1.0)
        return support * (grad - mean_grad)


class CELU(Activation):
    """Continuously Differentiable ELU (Barron 2017).

    ``f(x) = x`` if ``x >= 0`` else ``alpha*(exp(x/alpha) - 1)``

    Unlike ELU, CELU is continuously differentiable at ``x=0`` for any
    ``alpha``.

    Parameters
    ----------
    alpha : float
        Scale for the negative branch (default 1.0).
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def forward(self, x):
        self._x = x
        return np.where(x >= 0, x, self.alpha * (np.exp(x / self.alpha) - 1.0))

    def backward(self, grad):
        dx = np.where(self._x >= 0, 1.0,
                      np.exp(self._x / self.alpha))
        return grad * dx


class Softsign(Activation):
    """Softsign activation. ``f(x) = x / (1 + |x|)``.

    A smooth, bounded alternative to Tanh with heavier tails.
    Saturates more slowly, which can help with gradient flow.
    """

    def forward(self, x):
        self._denom = 1.0 + np.abs(x)
        return x / self._denom

    def backward(self, grad):
        return grad / (self._denom ** 2)


class Tanhshrink(Activation):
    """Tanhshrink activation. ``f(x) = x - tanh(x)``.

    A soft-thresholding function that shrinks values toward zero.
    Used in sparse autoencoders and shrinkage estimators.
    """

    def forward(self, x):
        self._tanh = np.tanh(x)
        return x - self._tanh

    def backward(self, grad):
        return grad * (self._tanh ** 2)   # 1 - sech²(x) = tanh²(x)


_REGISTRY = {
    "linear": Linear,
    "relu": ReLU,
    "leaky_relu": LeakyReLU,
    "elu": ELU,
    "selu": SELU,
    "sigmoid": Sigmoid,
    "tanh": Tanh,
    "softmax": Softmax,
    "softplus": Softplus,
    "swish": Swish,
    "mish": Mish,
    "gelu": GELU,
    "prelu": PReLU,
    "sine": Sine,
    "hardswish": Hardswish,
    "bent_identity": BentIdentity,
    "squareplus": Squareplus,
    "relu6": ReLU6,
    "hardsigmoid": Hardsigmoid,
    "log_softmax": LogSoftmax,
    "sparsemax": Sparsemax,
    "celu": CELU,
    "softsign": Softsign,
    "tanhshrink": Tanhshrink,
}


def get(identifier):
    """Return an :class:`Activation` instance from a string or object.

    Parameters
    ----------
    identifier : str, Activation, or None
        ``None`` or ``"linear"`` → :class:`Linear`.

    Raises
    ------
    ValueError
        If the string key is not recognised.
    """
    if identifier is None:
        return Linear()
    if isinstance(identifier, Activation):
        return identifier
    if isinstance(identifier, str):
        key = identifier.lower()
        if key in _REGISTRY:
            return _REGISTRY[key]()
        raise ValueError(
            f"Unknown activation: '{identifier}'. "
            f"Available: {list(_REGISTRY)}"
        )
    raise TypeError(f"Could not interpret activation: {identifier}")
