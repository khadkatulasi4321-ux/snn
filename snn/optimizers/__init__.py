from .base import Optimizer
from .sgd import SGD
from .adam import Adam, AdamW
from .rmsprop import RMSprop
from .adagrad import Adagrad, Adadelta
from .nadam import Nadam
from .radam import RAdam
from .lion import Lion
from .lamb import LAMB
from .lookahead import Lookahead
from .adan import Adan

_REGISTRY = {
    "sgd": SGD,
    "adam": Adam,
    "adamw": AdamW,
    "rmsprop": RMSprop,
    "adagrad": Adagrad,
    "adadelta": Adadelta,
    "nadam": Nadam,
    "radam": RAdam,
    "lion": Lion,
    "lamb": LAMB,
    "adan": Adan,
    # Lookahead is a wrapper — not in string registry (wrap manually)
}


def get(identifier):
    """Return an :class:`Optimizer` instance from a string, instance, or dict.

    Parameters
    ----------
    identifier : str, Optimizer, or dict
        * String key — ``"adam"``, ``"sgd"``, …
        * Optimizer instance — returned unchanged.
        * Config dict — ``{"name": "adam", "learning_rate": 1e-3}``. Any key
          accepted by the optimizer constructor may appear; ``"name"`` (or
          ``"class"``) selects the class.

    Raises
    ------
    ValueError
        Unknown string or dict name.
    """
    if isinstance(identifier, Optimizer):
        return identifier
    if isinstance(identifier, dict):
        cfg = dict(identifier)
        name = cfg.pop("name", cfg.pop("class", "adam")).lower()
        if name not in _REGISTRY:
            raise ValueError(
                f"Unknown optimizer '{name}'. Available: {list(_REGISTRY)}"
            )
        return _REGISTRY[name](**cfg)
    if isinstance(identifier, str):
        key = identifier.lower()
        if key in _REGISTRY:
            return _REGISTRY[key]()
        raise ValueError(
            f"Unknown optimizer: '{identifier}'. Available: {list(_REGISTRY)}"
        )
    raise TypeError(f"Could not interpret optimizer: {identifier}")


__all__ = [
    "Optimizer",
    "SGD",
    "Adam",
    "AdamW",
    "RMSprop",
    "Adagrad",
    "Adadelta",
    "Nadam",
    "RAdam",
    "Lion",
    "LAMB",
    "Lookahead",
    "Adan",
    "get",
]
