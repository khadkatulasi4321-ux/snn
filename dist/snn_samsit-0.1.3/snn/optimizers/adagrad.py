import numpy as np
from .base import Optimizer


class Adagrad(Optimizer):
    """
    Adagrad optimizer — adapts learning rates using accumulated squared gradients.
    """

    def __init__(self, learning_rate=0.01, epsilon=1e-8,
                 initial_accumulator_value=0.1, weight_decay=0.0):
        super().__init__(learning_rate)
        self.epsilon = epsilon
        self.initial_accumulator_value = initial_accumulator_value
        self.weight_decay = weight_decay
        self._accumulator = {}

    def apply_gradients(self, params, grads):
        self._iterations += 1
        updates = {}

        for key, grad in grads.items():
            param = params[key]
            g = grad + self.weight_decay * param if self.weight_decay else grad

            acc = self._accumulator.get(
                key,
                np.full_like(param, self.initial_accumulator_value)
            )
            acc_new = acc + g ** 2
            self._accumulator[key] = acc_new

            updates[key] = param - self.learning_rate * g / (np.sqrt(acc_new) + self.epsilon)

        return updates

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "epsilon": self.epsilon,
            "initial_accumulator_value": self.initial_accumulator_value,
            "weight_decay": self.weight_decay,
        })
        return cfg


class Adadelta(Optimizer):
    """
    Adadelta optimizer — adapts learning rates without a global learning rate.
    """

    def __init__(self, learning_rate=1.0, rho=0.95, epsilon=1e-7,
                 weight_decay=0.0):
        super().__init__(learning_rate)
        self.rho = rho
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self._E_g2 = {}
        self._E_dx2 = {}

    def apply_gradients(self, params, grads):
        self._iterations += 1
        updates = {}

        for key, grad in grads.items():
            param = params[key]
            g = grad + self.weight_decay * param if self.weight_decay else grad

            E_g2 = self._E_g2.get(key, np.zeros_like(param))
            E_dx2 = self._E_dx2.get(key, np.zeros_like(param))

            E_g2_new = self.rho * E_g2 + (1.0 - self.rho) * g ** 2
            dx = -(np.sqrt(E_dx2 + self.epsilon) /
                   np.sqrt(E_g2_new + self.epsilon)) * g
            E_dx2_new = self.rho * E_dx2 + (1.0 - self.rho) * dx ** 2

            self._E_g2[key] = E_g2_new
            self._E_dx2[key] = E_dx2_new

            updates[key] = param + self.learning_rate * dx

        return updates

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "rho": self.rho,
            "epsilon": self.epsilon,
            "weight_decay": self.weight_decay,
        })
        return cfg
