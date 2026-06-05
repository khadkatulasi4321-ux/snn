import numpy as np


class Loss:
    def forward(self, y_pred, y_true):
        raise NotImplementedError

    def backward(self, y_pred, y_true):
        raise NotImplementedError

    def __call__(self, y_pred, y_true):
        return self.forward(y_pred, y_true)


class MeanSquaredError(Loss):
    def forward(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)

    def backward(self, y_pred, y_true):
        n = y_pred.shape[0]
        return 2.0 * (y_pred - y_true) / n


class MeanAbsoluteError(Loss):
    def forward(self, y_pred, y_true):
        return np.mean(np.abs(y_pred - y_true))

    def backward(self, y_pred, y_true):
        n = y_pred.shape[0]
        return np.sign(y_pred - y_true) / n


class HuberLoss(Loss):
    def __init__(self, delta=1.0):
        self.delta = delta

    def forward(self, y_pred, y_true):
        diff = np.abs(y_pred - y_true)
        quadratic = np.minimum(diff, self.delta)
        linear = diff - quadratic
        return np.mean(0.5 * quadratic ** 2 + self.delta * linear)

    def backward(self, y_pred, y_true):
        n = y_pred.shape[0]
        diff = y_pred - y_true
        abs_diff = np.abs(diff)
        return np.where(abs_diff <= self.delta, diff, self.delta * np.sign(diff)) / n


class BinaryCrossentropy(Loss):
    def __init__(self, from_logits=False, eps=1e-7):
        self.from_logits = from_logits
        self.eps = eps

    def forward(self, y_pred, y_true):
        if self.from_logits:
            y_pred = 1.0 / (1.0 + np.exp(-np.clip(y_pred, -500, 500)))
        y_pred = np.clip(y_pred, self.eps, 1.0 - self.eps)
        return -np.mean(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred))

    def backward(self, y_pred, y_true):
        n = y_pred.shape[0]
        if self.from_logits:
            y_pred_sig = 1.0 / (1.0 + np.exp(-np.clip(y_pred, -500, 500)))
            return (y_pred_sig - y_true) / n
        y_pred = np.clip(y_pred, self.eps, 1.0 - self.eps)
        return (-(y_true / y_pred) + (1.0 - y_true) / (1.0 - y_pred)) / n


class CategoricalCrossentropy(Loss):
    def __init__(self, from_logits=False, eps=1e-7):
        self.from_logits = from_logits
        self.eps = eps

    def _softmax(self, x):
        shifted = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(shifted)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def forward(self, y_pred, y_true):
        if self.from_logits:
            y_pred = self._softmax(y_pred)
        y_pred = np.clip(y_pred, self.eps, 1.0 - self.eps)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=-1))

    def backward(self, y_pred, y_true):
        n = y_pred.shape[0]
        if self.from_logits:
            probs = self._softmax(y_pred)
            return (probs - y_true) / n
        y_pred = np.clip(y_pred, self.eps, 1.0 - self.eps)
        return -(y_true / y_pred) / n


class SparseCategoricalCrossentropy(Loss):
    def __init__(self, from_logits=False, eps=1e-7):
        self.from_logits = from_logits
        self.eps = eps

    def _softmax(self, x):
        shifted = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(shifted)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def _to_one_hot(self, y_true, n_classes):
        n = y_true.shape[0]
        one_hot = np.zeros((n, n_classes))
        one_hot[np.arange(n), y_true.astype(int)] = 1.0
        return one_hot

    def forward(self, y_pred, y_true):
        if self.from_logits:
            y_pred = self._softmax(y_pred)
        y_pred = np.clip(y_pred, self.eps, 1.0 - self.eps)
        n = y_pred.shape[0]
        idx = y_true.astype(int).flatten()
        return -np.mean(np.log(y_pred[np.arange(n), idx]))

    def backward(self, y_pred, y_true):
        n = y_pred.shape[0]
        if self.from_logits:
            probs = self._softmax(y_pred)
            one_hot = self._to_one_hot(y_true.flatten(), y_pred.shape[-1])
            return (probs - one_hot) / n
        y_pred = np.clip(y_pred, self.eps, 1.0 - self.eps)
        one_hot = self._to_one_hot(y_true.flatten(), y_pred.shape[-1])
        return -(one_hot / y_pred) / n


class KLDivergence(Loss):
    def __init__(self, eps=1e-7):
        self.eps = eps

    def forward(self, y_pred, y_true):
        y_pred = np.clip(y_pred, self.eps, 1.0)
        y_true = np.clip(y_true, self.eps, 1.0)
        return np.mean(np.sum(y_true * np.log(y_true / y_pred), axis=-1))

    def backward(self, y_pred, y_true):
        n = y_pred.shape[0]
        y_pred = np.clip(y_pred, self.eps, 1.0)
        y_true = np.clip(y_true, self.eps, 1.0)
        return -(y_true / y_pred) / n


_REGISTRY = {
    "mse": MeanSquaredError,
    "mean_squared_error": MeanSquaredError,
    "mae": MeanAbsoluteError,
    "mean_absolute_error": MeanAbsoluteError,
    "huber": HuberLoss,
    "binary_crossentropy": BinaryCrossentropy,
    "categorical_crossentropy": CategoricalCrossentropy,
    "sparse_categorical_crossentropy": SparseCategoricalCrossentropy,
    "kl_divergence": KLDivergence,
}


def get(identifier):
    """Return a :class:`Loss` instance from a string, instance, or dict.

    Parameters
    ----------
    identifier : str, Loss, or dict
        * String key — ``"mse"``, ``"categorical_crossentropy"``, …
        * Loss instance — returned unchanged.
        * Config dict — ``{"name": "huber", "delta": 2.0}``. Any key accepted
          by the loss constructor may appear; ``"name"`` (or ``"class"``)
          selects the class.

    Raises
    ------
    ValueError
        Unknown string or dict name.
    """
    if isinstance(identifier, Loss):
        return identifier
    if isinstance(identifier, dict):
        cfg = dict(identifier)
        name = cfg.pop("name", cfg.pop("class", "mse")).lower()
        if name not in _REGISTRY:
            raise ValueError(
                f"Unknown loss '{name}'. Available: {list(_REGISTRY)}"
            )
        return _REGISTRY[name](**cfg)
    if isinstance(identifier, str):
        key = identifier.lower()
        if key in _REGISTRY:
            return _REGISTRY[key]()
        raise ValueError(
            f"Unknown loss: '{identifier}'. Available: {list(_REGISTRY)}"
        )
    raise TypeError(f"Could not interpret loss: {identifier}")
