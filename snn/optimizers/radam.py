import numpy as np
from .base import Optimizer


class RAdam(Optimizer):
    """
    RAdam — Rectified Adam.

    Fixes the variance issue in Adam's early training phase by computing
    the effective sample size of the second moment estimate.  When the
    approximation is not reliable (small t), it falls back to an SGD-with-
    momentum step; otherwise it applies the full variance-corrected Adam
    update.  This removes the need for a warmup schedule.

    Reference: Liu et al. (2019) "On the Variance of the Adaptive Learning
    Rate and Beyond". https://arxiv.org/abs/1908.03265

    Parameters
    ----------
    learning_rate : float
        Step size (default 1e-3).
    beta_1 : float
        First-moment decay (default 0.9).
    beta_2 : float
        Second-moment decay (default 0.999).
    epsilon : float
        Numerical stability constant (default 1e-8).
    weight_decay : float
        L2 regularisation coefficient (default 0.0).
    """

    def __init__(self, learning_rate=1e-3, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-8, weight_decay=0.0):
        super().__init__(learning_rate)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self._m = {}
        self._v = {}
        # Maximum ρ (SMA length upper bound)
        self._rho_max = 2.0 / (1.0 - beta_2) - 1.0

    def apply_gradients(self, params, grads):
        self._iterations += 1
        t = self._iterations
        updates = {}

        b1t = self.beta_1 ** t
        b2t = self.beta_2 ** t

        # SMA approximation at step t
        rho_t = self._rho_max - 2.0 * t * b2t / (1.0 - b2t)

        for key, grad in grads.items():
            param = params[key]
            g = grad + self.weight_decay * param if self.weight_decay else grad

            m = self._m.get(key, np.zeros_like(param))
            v = self._v.get(key, np.zeros_like(param))

            m_new = self.beta_1 * m + (1.0 - self.beta_1) * g
            v_new = self.beta_2 * v + (1.0 - self.beta_2) * g ** 2

            self._m[key] = m_new
            self._v[key] = v_new

            m_hat = m_new / (1.0 - b1t)

            if rho_t > 4:
                # Variance is tractable — use full Adam + rectification
                r = np.sqrt(
                    (rho_t - 4) * (rho_t - 2) * self._rho_max
                    / ((self._rho_max - 4) * (self._rho_max - 2) * rho_t)
                )
                v_hat = np.sqrt(v_new / (1.0 - b2t))
                updates[key] = param - self.learning_rate * r * m_hat / (v_hat + self.epsilon)
            else:
                # Variance is intractable — SGD + momentum step
                updates[key] = param - self.learning_rate * m_hat

        return updates

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "beta_1": self.beta_1,
            "beta_2": self.beta_2,
            "epsilon": self.epsilon,
            "weight_decay": self.weight_decay,
        })
        return cfg
