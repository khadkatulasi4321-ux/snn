class Optimizer:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self._iterations = 0

    def apply_gradients(self, params, grads):
        raise NotImplementedError

    def get_config(self):
        return {"learning_rate": self.learning_rate}
