class Layer:
    def __init__(self, trainable=True, name=None):
        self.trainable = trainable
        self.name = name
        self._built = False

    def build(self, input_shape):
        self._built = True

    def forward(self, x, training=False):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError

    def __call__(self, x, training=False):
        # Graph-mode detection: when used with the functional API
        # (GraphModel + Input), x will be a _Tensor or a list of _Tensor.
        # We use duck typing so base.py doesn't import from model.py.
        if getattr(x, "_is_graph_tensor", False):
            from ..model import _Tensor
            return _Tensor(layer=self, parents=[x])
        if (isinstance(x, list) and x
                and getattr(x[0], "_is_graph_tensor", False)):
            from ..model import _Tensor
            return _Tensor(layer=self, parents=x)
        return self.forward(x, training=training)

    @property
    def params(self):
        return {}

    @property
    def grads(self):
        return {}

    def get_config(self):
        return {"name": self.name, "trainable": self.trainable}

    def count_params(self):
        total = 0
        for p in self.params.values():
            total += p.size
        return total

    def set_param(self, key, val):
        """Set a single parameter by name.

        Override in composite layers (e.g. TransformerBlock) to route
        prefixed keys to the correct sub-layer.  The default implementation
        calls ``setattr(self, key, val)`` when the attribute exists.
        """
        if hasattr(self, key):
            setattr(self, key, val)
