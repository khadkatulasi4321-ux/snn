import numpy as np
from .base import Optimizer


class Nadam(Optimizer):
    """
    Nadam — Nesterov-accelerated Adaptive Moment Estimation.

    Combines Adam's adaptive learning rates with Nesterov momentum,
    giving faster convergence than standard Adam in many settings.

    f(θ) update rule::

        m_t  = β₁ · m_{t-1} + (1 − β₁) · g_t
        v_t  = β₂ · v_{t-1} + (1 − β₂) · g_t²
        m̂    = β₁ · m_t / (1−β₁^{t+1}) + (1−β₁) · g_t / (1−β₁^t)
        θ_t  = θ_{t-1} − lr · m̂ / (√v̂ + ε)

    Parameters
    ----------
    learning_rate : float
        Step size (default 2e-3).
    beta_1 : float
        Exponential decay for first moment (default 0.9).
    beta_2 : float
        Exponential decay for second moment (default 0.999).
    epsilon : float
        Numerical stability constant (default 1e-8).
    weight_decay : float
        L2 regularisation coefficient (default 0.0).
    """

    def __init__(self, learning_rate=2e-3, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-8, weight_decay=0.0):
        super().__init__(learning_rate)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self._m = {}
        self._v = {}

    def apply_gradients(self, params, grads):
        self._iterations += 1
        t = self._iterations
        updates = {}

        b1t = 1.0 - self.beta_1 ** t
        b2t = 1.0 - self.beta_2 ** t

        for key, grad in grads.items():
            param = params[key]
            g = grad + self.weight_decay * param if self.weight_decay else grad

            m = self._m.get(key, np.zeros_like(param))
            v = self._v.get(key, np.zeros_like(param))

            m_new = self.beta_1 * m + (1.0 - self.beta_1) * g
            v_new = self.beta_2 * v + (1.0 - self.beta_2) * g ** 2

            self._m[key] = m_new
            self._v[key] = v_new

            # Nesterov look-ahead estimate of m̂
            m_hat = (self.beta_1 * m_new / (1.0 - self.beta_1 ** (t + 1))
                     + (1.0 - self.beta_1) * g / b1t)
            v_hat = v_new / b2t

            updates[key] = param - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

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
