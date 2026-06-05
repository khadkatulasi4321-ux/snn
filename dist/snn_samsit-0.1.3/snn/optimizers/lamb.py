import numpy as np
from .base import Optimizer


class LAMB(Optimizer):
    """
    LAMB — Layer-wise Adaptive Moments optimizer (Ginsburg et al. 2019).

    Extends Adam with a *layer-wise trust ratio* that scales the update
    by ``‖θ‖ / ‖adam_update‖``.  This allows training with very large
    batch sizes (e.g. 65 536) without learning-rate tuning, and is the
    standard optimizer for BERT-style pretraining.

    Update rule::

        m_t = β₁·m_{t-1} + (1−β₁)·g_t
        v_t = β₂·v_{t-1} + (1−β₂)·g_t²
        m̂   = m_t/(1−β₁^t)   v̂ = v_t/(1−β₂^t)
        u   = m̂/(√v̂ + ε) + λ·θ
        r   = clip(‖θ‖/‖u‖, 0, clip_ratio)
        θ   = θ − lr·r·u

    Note
    ----
    LAMB is designed for **large batch training**.  For small batches
    (< 512) Adam or AdamW will generally perform equally well.

    Parameters
    ----------
    learning_rate : float
        Base step size (default 1e-3).
    beta_1, beta_2 : float
        Moment decay rates (default 0.9, 0.999).
    epsilon : float
        Numerical stability (default 1e-6).
    weight_decay : float
        L2 regularisation coefficient λ (default 0.0).
    clip_ratio : float
        Upper bound for the trust ratio (default 10.0).
    """

    def __init__(self, learning_rate=1e-3, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-6, weight_decay=0.0, clip_ratio=10.0):
        super().__init__(learning_rate)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.clip_ratio = clip_ratio
        self._m = {}
        self._v = {}

    def apply_gradients(self, params, grads):
        self._iterations += 1
        t = self._iterations
        b1t = 1.0 - self.beta_1 ** t
        b2t = 1.0 - self.beta_2 ** t
        updates = {}

        for key, grad in grads.items():
            param = params[key]
            m = self._m.get(key, np.zeros_like(param))
            v = self._v.get(key, np.zeros_like(param))

            m_new = self.beta_1 * m + (1.0 - self.beta_1) * grad
            v_new = self.beta_2 * v + (1.0 - self.beta_2) * grad ** 2
            self._m[key] = m_new
            self._v[key] = v_new

            m_hat = m_new / b1t
            v_hat = v_new / b2t

            # Adam update with weight decay
            u = m_hat / (np.sqrt(v_hat) + self.epsilon)
            if self.weight_decay:
                u = u + self.weight_decay * param

            # Layer-wise trust ratio
            w_norm = np.linalg.norm(param)
            u_norm = np.linalg.norm(u)
            if w_norm > 0 and u_norm > 0:
                r = np.clip(w_norm / u_norm, 0.0, self.clip_ratio)
            else:
                r = 1.0

            updates[key] = param - self.learning_rate * r * u

        return updates

    def get_config(self):
        cfg = super().get_config()
        cfg.update(dict(beta_1=self.beta_1, beta_2=self.beta_2,
                        epsilon=self.epsilon, weight_decay=self.weight_decay,
                        clip_ratio=self.clip_ratio))
        return cfg
