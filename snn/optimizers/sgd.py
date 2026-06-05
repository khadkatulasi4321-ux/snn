import numpy as np
from .base import Optimizer


class SGD(Optimizer):
    """
    Stochastic Gradient Descent with optional momentum and Nesterov acceleration.
    """

    def __init__(self, learning_rate=0.01, momentum=0.0, nesterov=False,
                 weight_decay=0.0):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.nesterov = nesterov
        self.weight_decay = weight_decay
        self._velocities = {}

    def apply_gradients(self, params, grads):
        self._iterations += 1
        updates = {}
        for key, grad in grads.items():
            param = params[key]
            g = grad + self.weight_decay * param if self.weight_decay else grad

            if self.momentum > 0:
                v = self._velocities.get(key, np.zeros_like(param))
                v_new = self.momentum * v - self.learning_rate * g
                self._velocities[key] = v_new
                if self.nesterov:
                    updates[key] = param + self.momentum * v_new - self.learning_rate * g
                else:
                    updates[key] = param + v_new
            else:
                updates[key] = param - self.learning_rate * g

        return updates

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "momentum": self.momentum,
            "nesterov": self.nesterov,
            "weight_decay": self.weight_decay,
        })
        return cfg
