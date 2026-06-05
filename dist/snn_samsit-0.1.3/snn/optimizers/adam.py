import numpy as np
from .base import Optimizer


class Adam(Optimizer):
    """
    Adam optimizer (Adaptive Moment Estimation).
    """

    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-8, weight_decay=0.0, amsgrad=False):
        super().__init__(learning_rate)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        self._m = {}
        self._v = {}
        self._v_hat_max = {}

    def apply_gradients(self, params, grads):
        self._iterations += 1
        t = self._iterations
        updates = {}
        lr_t = self.learning_rate * (np.sqrt(1.0 - self.beta_2 ** t) /
                                      (1.0 - self.beta_1 ** t))

        for key, grad in grads.items():
            param = params[key]
            g = grad + self.weight_decay * param if self.weight_decay else grad

            m = self._m.get(key, np.zeros_like(param))
            v = self._v.get(key, np.zeros_like(param))

            m_new = self.beta_1 * m + (1.0 - self.beta_1) * g
            v_new = self.beta_2 * v + (1.0 - self.beta_2) * g ** 2

            self._m[key] = m_new
            self._v[key] = v_new

            if self.amsgrad:
                v_hat_max = self._v_hat_max.get(key, np.zeros_like(param))
                v_hat_max_new = np.maximum(v_hat_max, v_new)
                self._v_hat_max[key] = v_hat_max_new
                denom = np.sqrt(v_hat_max_new) + self.epsilon
            else:
                denom = np.sqrt(v_new) + self.epsilon

            updates[key] = param - lr_t * m_new / denom

        return updates

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "beta_1": self.beta_1,
            "beta_2": self.beta_2,
            "epsilon": self.epsilon,
            "weight_decay": self.weight_decay,
            "amsgrad": self.amsgrad,
        })
        return cfg


class AdamW(Adam):
    """
    AdamW — Adam with decoupled weight decay regularization.
    """

    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-8, weight_decay=0.01):
        super().__init__(learning_rate, beta_1, beta_2, epsilon, weight_decay=0.0)
        self._wd = weight_decay

    def apply_gradients(self, params, grads):
        self._iterations += 1
        t = self._iterations
        updates = {}
        lr_t = self.learning_rate * (np.sqrt(1.0 - self.beta_2 ** t) /
                                      (1.0 - self.beta_1 ** t))

        for key, grad in grads.items():
            param = params[key]
            m = self._m.get(key, np.zeros_like(param))
            v = self._v.get(key, np.zeros_like(param))

            m_new = self.beta_1 * m + (1.0 - self.beta_1) * grad
            v_new = self.beta_2 * v + (1.0 - self.beta_2) * grad ** 2

            self._m[key] = m_new
            self._v[key] = v_new

            denom = np.sqrt(v_new) + self.epsilon
            adam_update = lr_t * m_new / denom
            wd_update = self.learning_rate * self._wd * param

            updates[key] = param - adam_update - wd_update

        return updates
