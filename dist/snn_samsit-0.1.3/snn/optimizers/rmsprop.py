import numpy as np
from .base import Optimizer


class RMSprop(Optimizer):
    """
    RMSprop optimizer.
    """

    def __init__(self, learning_rate=0.001, rho=0.9, epsilon=1e-8,
                 momentum=0.0, weight_decay=0.0, centered=False):
        super().__init__(learning_rate)
        self.rho = rho
        self.epsilon = epsilon
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.centered = centered
        self._mean_sq = {}
        self._mean_grad = {}
        self._mom = {}

    def apply_gradients(self, params, grads):
        self._iterations += 1
        updates = {}

        for key, grad in grads.items():
            param = params[key]
            g = grad + self.weight_decay * param if self.weight_decay else grad

            mean_sq = self._mean_sq.get(key, np.zeros_like(param))
            mean_sq_new = self.rho * mean_sq + (1.0 - self.rho) * g ** 2
            self._mean_sq[key] = mean_sq_new

            if self.centered:
                mean_grad = self._mean_grad.get(key, np.zeros_like(param))
                mean_grad_new = self.rho * mean_grad + (1.0 - self.rho) * g
                self._mean_grad[key] = mean_grad_new
                denom = np.sqrt(mean_sq_new - mean_grad_new ** 2 + self.epsilon)
            else:
                denom = np.sqrt(mean_sq_new + self.epsilon)

            if self.momentum > 0:
                mom = self._mom.get(key, np.zeros_like(param))
                mom_new = self.momentum * mom + self.learning_rate * g / denom
                self._mom[key] = mom_new
                updates[key] = param - mom_new
            else:
                updates[key] = param - self.learning_rate * g / denom

        return updates

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "rho": self.rho,
            "epsilon": self.epsilon,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay,
            "centered": self.centered,
        })
        return cfg
