import numpy as np
from .base import Optimizer


class Lion(Optimizer):
    """
    Lion — EvoLved Sign Momentum.

    Uses only the *sign* of the update vector, making every parameter
    receive an update of exactly ±learning_rate per step.  This gives
    Lion 3–10× lower memory usage than Adam (one momentum buffer instead
    of two) and competitive or better performance on image and language
    tasks.

    Update rule::

        c_t  = β₁ · m_{t-1} + (1 − β₁) · g_t
        θ_t  = θ_{t-1} − lr · (sign(c_t) + λ · θ_{t-1})
        m_t  = β₂ · m_{t-1} + (1 − β₂) · g_t

    Note: Lion typically needs a **3–10× smaller learning rate** and
    **3–10× larger weight decay** than Adam.  A good starting point is
    ``lr=1e-4, weight_decay=1e-2``.

    Reference: Chen et al. (2023) "Symbolic Discovery of Optimization
    Algorithms". https://arxiv.org/abs/2302.06675

    Parameters
    ----------
    learning_rate : float
        Step size (default 1e-4).
    beta_1 : float
        Coefficient for computing the update direction (default 0.9).
    beta_2 : float
        Coefficient for maintaining the momentum buffer (default 0.99).
    weight_decay : float
        Decoupled weight decay (default 0.0).
    """

    def __init__(self, learning_rate=1e-4, beta_1=0.9, beta_2=0.99,
                 weight_decay=0.0):
        super().__init__(learning_rate)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.weight_decay = weight_decay
        self._m = {}

    def apply_gradients(self, params, grads):
        self._iterations += 1
        updates = {}

        for key, grad in grads.items():
            param = params[key]
            m = self._m.get(key, np.zeros_like(param))

            # Interpolated direction
            c = self.beta_1 * m + (1.0 - self.beta_1) * grad

            # Weight update: sign of c + weight decay
            delta = np.sign(c) + self.weight_decay * param
            new_param = param - self.learning_rate * delta

            # Update momentum with β₂ (separate from β₁ used above)
            self._m[key] = self.beta_2 * m + (1.0 - self.beta_2) * grad

            updates[key] = new_param

        return updates

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "beta_1": self.beta_1,
            "beta_2": self.beta_2,
            "weight_decay": self.weight_decay,
        })
        return cfg
