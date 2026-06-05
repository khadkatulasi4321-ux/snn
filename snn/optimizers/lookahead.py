import numpy as np
from .base import Optimizer


class Lookahead(Optimizer):
    """
    Lookahead optimizer wrapper (Zhang et al. 2019).

    Wraps any base optimizer and adds a *slow-weights* outer loop:

    * **Inner loop** — the base optimizer updates "fast weights" for ``k``
      steps as usual.
    * **Outer update** — after every ``k`` inner steps, the slow weights
      interpolate toward the fast weights::

          θ_slow ← θ_slow + α · (θ_fast − θ_slow)
          θ_fast ← θ_slow

    This stabilises training across a wide range of learning rates and
    often improves generalisation with minimal overhead.

    Parameters
    ----------
    optimizer : Optimizer
        Any snn optimizer instance (Adam, SGD, Nadam, …).
    k : int
        Number of inner (fast) steps before each slow update (default 5).
    alpha : float
        Slow-weights interpolation coefficient (default 0.5).

    Examples
    --------
    >>> from snn.optimizers import Adam, Lookahead
    >>> opt = Lookahead(Adam(learning_rate=1e-3), k=5, alpha=0.5)
    >>> model.compile(opt, "categorical_crossentropy")
    """

    def __init__(self, optimizer, k=5, alpha=0.5):
        # Set _inner BEFORE calling super().__init__ because the base-class
        # constructor calls `self.learning_rate = ...` which triggers our
        # property setter (which reads self._inner).
        self._inner = optimizer
        super().__init__(optimizer.learning_rate)
        self.k = k
        self.alpha = alpha
        self._slow = {}
        self._step = 0

    # Proxy learning_rate to the inner optimizer so compile() can set it
    @property
    def learning_rate(self):
        return self._inner.learning_rate

    @learning_rate.setter
    def learning_rate(self, val):
        self._inner.learning_rate = val

    @property
    def weight_decay(self):
        return getattr(self._inner, "weight_decay", 0.0)

    @weight_decay.setter
    def weight_decay(self, val):
        if hasattr(self._inner, "weight_decay"):
            self._inner.weight_decay = val

    def apply_gradients(self, params, grads):
        # ── inner optimizer step ──
        fast = self._inner.apply_gradients(params, grads)
        self._step += 1

        # Initialise slow weights on first call
        for key, val in fast.items():
            if key not in self._slow:
                self._slow[key] = val.copy()

        # ── outer slow-weights update every k steps ──
        if self._step % self.k == 0:
            for key in fast:
                self._slow[key] = (self._slow[key]
                                   + self.alpha * (fast[key] - self._slow[key]))
                fast[key] = self._slow[key].copy()

        return fast

    def get_config(self):
        return {
            "optimizer": type(self._inner).__name__,
            "learning_rate": self.learning_rate,
            "k": self.k,
            "alpha": self.alpha,
        }
