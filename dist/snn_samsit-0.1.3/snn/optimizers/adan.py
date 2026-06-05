import numpy as np
from .base import Optimizer


class Adan(Optimizer):
    """
    Adan — Adaptive Nesterov Momentum Algorithm (Xie et al. 2022).

    Uses three exponential moving averages — of the gradient, the gradient
    *difference*, and a combined Nesterov-like term — to get fast convergence
    on non-convex problems.  Achieves competitive or better results than
    Adam on image classification and NLP benchmarks.

    Update rule::

        dk  = gk − g_{k-1}           (gradient difference)
        m1  = β₁·m1 + (1−β₁)·gk
        m2  = β₂·m2 + (1−β₂)·dk
        m3  = β₃·m3 + (1−β₃)·(gk + (1−β₂)·dk)²
        η   = lr / (√m3 + ε)
        θ   = (1 + λ·lr)⁻¹ · (θ − η · (m1 + (1−β₂)·m2))

    Parameters
    ----------
    learning_rate : float
        Step size (default 1e-3).
    beta_1 : float
        Decay for first moment (default 0.98).
    beta_2 : float
        Decay for gradient difference (default 0.92).
    beta_3 : float
        Decay for second-order moment (default 0.99).
    epsilon : float
        Numerical stability (default 1e-8).
    weight_decay : float
        Decoupled weight decay λ (default 0.02).
    """

    def __init__(self, learning_rate=1e-3, beta_1=0.98, beta_2=0.92,
                 beta_3=0.99, epsilon=1e-8, weight_decay=0.02):
        super().__init__(learning_rate)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.beta_3 = beta_3
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self._m1 = {}
        self._m2 = {}
        self._m3 = {}
        self._prev_g = {}

    def apply_gradients(self, params, grads):
        self._iterations += 1
        updates = {}

        for key, g in grads.items():
            param = params[key]
            prev_g = self._prev_g.get(key, np.zeros_like(param))
            m1 = self._m1.get(key, np.zeros_like(param))
            m2 = self._m2.get(key, np.zeros_like(param))
            m3 = self._m3.get(key, np.zeros_like(param))

            dk = g - prev_g
            m1_new = self.beta_1 * m1 + (1.0 - self.beta_1) * g
            m2_new = self.beta_2 * m2 + (1.0 - self.beta_2) * dk
            nesterov = g + (1.0 - self.beta_2) * dk
            m3_new = self.beta_3 * m3 + (1.0 - self.beta_3) * nesterov ** 2

            self._m1[key] = m1_new
            self._m2[key] = m2_new
            self._m3[key] = m3_new
            self._prev_g[key] = g.copy()

            eta = self.learning_rate / (np.sqrt(m3_new) + self.epsilon)
            step = eta * (m1_new + (1.0 - self.beta_2) * m2_new)

            # Decoupled weight decay
            updates[key] = (param - step) / (1.0 + self.weight_decay * self.learning_rate)

        return updates

    def get_config(self):
        cfg = super().get_config()
        cfg.update(dict(beta_1=self.beta_1, beta_2=self.beta_2,
                        beta_3=self.beta_3, epsilon=self.epsilon,
                        weight_decay=self.weight_decay))
        return cfg
