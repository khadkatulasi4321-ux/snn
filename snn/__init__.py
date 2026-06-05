"""
snn — A neural network / deep learning library built purely on NumPy.

Quickstart
----------
>>> from snn.model import Sequential
>>> from snn.layers import Dense, Dropout, BatchNormalization
>>> from snn.activations import ReLU, Softmax
>>>
>>> model = Sequential([
...     Dense(128, activation='relu'),
...     BatchNormalization(),
...     Dropout(0.3),
...     Dense(10, activation='softmax'),
... ])
>>> model.compile(optimizer='adam', loss='categorical_crossentropy',
...               metrics=['accuracy'])
>>> model.fit(X_train, y_train, epochs=20, batch_size=64,
...           validation_data=(X_val, y_val))
"""

from .model import Sequential, Model, GraphModel, Input
from .trainer import Trainer, Checkpoint
from . import layers
from . import activations
from . import losses
from . import optimizers
from . import initializers
from . import metrics
from . import utils
from . import nn
from . import docs

__version__ = "0.1.1"
__author__ = "samsit-phew"

__all__ = [
    "Sequential",
    "Model",
    "GraphModel",
    "Input",
    "Trainer",
    "Checkpoint",
    "layers",
    "activations",
    "losses",
    "optimizers",
    "initializers",
    "metrics",
    "utils",
    "nn",
    "docs",
    "__version__",
]
