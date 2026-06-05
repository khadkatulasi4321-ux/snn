import numpy as np


def zeros(shape):
    return np.zeros(shape, dtype=np.float64)


def ones(shape):
    return np.ones(shape, dtype=np.float64)


def random_normal(shape, mean=0.0, stddev=0.05, seed=None):
    rng = np.random.default_rng(seed)
    return rng.normal(mean, stddev, shape).astype(np.float64)


def random_uniform(shape, minval=-0.05, maxval=0.05, seed=None):
    rng = np.random.default_rng(seed)
    return rng.uniform(minval, maxval, shape).astype(np.float64)


def glorot_uniform(shape, seed=None):
    fan_in, fan_out = _compute_fans(shape)
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    rng = np.random.default_rng(seed)
    return rng.uniform(-limit, limit, shape).astype(np.float64)


def glorot_normal(shape, seed=None):
    fan_in, fan_out = _compute_fans(shape)
    stddev = np.sqrt(2.0 / (fan_in + fan_out))
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, stddev, shape).astype(np.float64)


def he_uniform(shape, seed=None):
    fan_in, _ = _compute_fans(shape)
    limit = np.sqrt(6.0 / fan_in)
    rng = np.random.default_rng(seed)
    return rng.uniform(-limit, limit, shape).astype(np.float64)


def he_normal(shape, seed=None):
    fan_in, _ = _compute_fans(shape)
    stddev = np.sqrt(2.0 / fan_in)
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, stddev, shape).astype(np.float64)


def lecun_uniform(shape, seed=None):
    fan_in, _ = _compute_fans(shape)
    limit = np.sqrt(3.0 / fan_in)
    rng = np.random.default_rng(seed)
    return rng.uniform(-limit, limit, shape).astype(np.float64)


def lecun_normal(shape, seed=None):
    fan_in, _ = _compute_fans(shape)
    stddev = np.sqrt(1.0 / fan_in)
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, stddev, shape).astype(np.float64)


def _compute_fans(shape):
    if len(shape) == 1:
        fan_in = fan_out = shape[0]
    elif len(shape) == 2:
        fan_in, fan_out = shape
    elif len(shape) == 4:
        receptive = shape[0] * shape[1]
        fan_in = receptive * shape[2]
        fan_out = receptive * shape[3]
    else:
        fan_in = fan_out = int(np.sqrt(np.prod(shape)))
    return fan_in, fan_out


_REGISTRY = {
    "zeros": zeros,
    "ones": ones,
    "random_normal": random_normal,
    "random_uniform": random_uniform,
    "glorot_uniform": glorot_uniform,
    "glorot_normal": glorot_normal,
    "he_uniform": he_uniform,
    "he_normal": he_normal,
    "lecun_uniform": lecun_uniform,
    "lecun_normal": lecun_normal,
}


def get(identifier):
    if callable(identifier):
        return identifier
    if isinstance(identifier, str):
        key = identifier.lower()
        if key in _REGISTRY:
            return _REGISTRY[key]
        raise ValueError(f"Unknown initializer: '{identifier}'. Available: {list(_REGISTRY)}")
    raise TypeError(f"Could not interpret initializer: {identifier}")
