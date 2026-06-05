"""
snn.nn — Everything in one flat namespace.

Inspired by the "Neural Networks from Scratch" (NNFS) book style, this
module collects every building block into a single importable namespace so
you never have to remember which sub-module something lives in.

Quick start::

    from snn.nn import Dense, ReLU, Softmax, Adam, CCE, Sequential

    model = Sequential([
        Dense(128),
        ReLU(),       # ← same class you'd use standalone!
        Dense(64),
        ReLU(),
        Dense(10),
        Softmax(),
    ])
    model.compile(Adam(learning_rate=1e-3), CCE())
    model.fit(X_train, y_train, epochs=20)

NNFS-style aliases are also provided so code from the book ports directly::

    from snn.nn import Layer_Dense, Activation_ReLU, Loss_CCE, Optimizer_Adam

    dense   = Layer_Dense(128, 64)
    relu    = Activation_ReLU()
    loss_fn = Loss_CCE()
    opt     = Optimizer_Adam(learning_rate=0.05)

    # Manual training loop (NNFS style)
    for epoch in range(100):
        out   = dense.forward(X)
        out   = relu.forward(out)
        loss  = loss_fn.forward(out, y)
        grad  = loss_fn.backward(out, y)
        grad  = relu.backward(grad)
        dense.backward(grad)
        opt.apply_gradients(dense.params, dense.grads)
"""

# ── Activations ───────────────────────────────────────────────────────────────
from .activations import (
    Activation,
    Linear,
    ReLU,
    LeakyReLU,
    ELU,
    SELU,
    Sigmoid,
    Tanh,
    Softmax,
    Softplus,
    Swish,
    Mish,
    GELU,
    PReLU,
    Sine,
    Hardswish,
    BentIdentity,
    Squareplus,
    ReLU6,
    Hardsigmoid,
    LogSoftmax,
    Sparsemax,
    CELU,
    Softsign,
    Tanhshrink,
    get as get_activation,
)

# ── Layers ────────────────────────────────────────────────────────────────────
from .layers import (
    Layer,
    Dense,
    Conv1D,
    Conv2D,
    MaxPooling1D,
    AveragePooling1D,
    MaxPooling2D,
    AveragePooling2D,
    GlobalAveragePooling1D,
    GlobalMaxPooling1D,
    GlobalAveragePooling2D,
    GlobalMaxPooling2D,
    Dropout,
    SpatialDropout2D,
    BatchNormalization,
    LayerNormalization,
    Flatten,
    Reshape,
    SimpleRNN,
    LSTM,
    GRU,
    Embedding,
    PositionalEncoding,
    MultiHeadAttention,
    TransformerBlock,
    GLU,
    SwiGLU,
    Add,
    Concatenate,
    Residual,
    TimeDistributed,
)

# ── Losses ────────────────────────────────────────────────────────────────────
from .losses import (
    Loss,
    MeanSquaredError,
    MeanAbsoluteError,
    HuberLoss,
    BinaryCrossentropy,
    CategoricalCrossentropy,
    SparseCategoricalCrossentropy,
    KLDivergence,
    get as get_loss,
)

# Short aliases — these mirror what NNFS calls its loss classes
MSE      = MeanSquaredError
MAE      = MeanAbsoluteError
Huber    = HuberLoss
BCE      = BinaryCrossentropy
CCE      = CategoricalCrossentropy
SparseCCE = SparseCategoricalCrossentropy
KLD      = KLDivergence

# ── Optimizers ────────────────────────────────────────────────────────────────
from .optimizers import (
    Optimizer,
    SGD,
    Adam,
    AdamW,
    RMSprop,
    Adagrad,
    Adadelta,
    Nadam,
    RAdam,
    Lion,
    LAMB,
    Lookahead,
    Adan,
    get as get_optimizer,
)

# ── Models ────────────────────────────────────────────────────────────────────
from .model import Sequential, Model, GraphModel, Input

# ── Utilities ─────────────────────────────────────────────────────────────────
from .utils import (
    to_categorical,
    train_test_split,
    EarlyStopping,
    ReduceLROnPlateau,
)
from .trainer import Trainer, Checkpoint
from .metrics import (
    accuracy,
    binary_accuracy,
    categorical_accuracy,
    precision,
    recall,
    f1_score,
    r2_score,
    confusion_matrix,
    get as get_metric,
)

# ═════════════════════════════════════════════════════════════════════════════
# NNFS-style class aliases
#
# The "Neural Networks from Scratch" book (Kinsley & Kukiela) prefixes every
# class with its category: Layer_*, Activation_*, Loss_*, Optimizer_*.  These
# aliases let you port NNFS code directly without renaming anything.
# ═════════════════════════════════════════════════════════════════════════════

# Layers
Layer_Dense              = Dense
Layer_Dropout            = Dropout
Layer_Flatten            = Flatten
Layer_Reshape            = Reshape
Layer_Conv2D             = Conv2D
Layer_Conv1D             = Conv1D
Layer_MaxPooling2D       = MaxPooling2D
Layer_AveragePooling2D   = AveragePooling2D
Layer_MaxPooling1D       = MaxPooling1D
Layer_AveragePooling1D   = AveragePooling1D
Layer_LSTM               = LSTM
Layer_GRU                = GRU
Layer_SimpleRNN          = SimpleRNN
Layer_Embedding          = Embedding
Layer_BatchNorm          = BatchNormalization
Layer_LayerNorm          = LayerNormalization
Layer_Residual           = Residual
Layer_TimeDistributed    = TimeDistributed
Layer_TransformerBlock   = TransformerBlock

# Activations
Activation_Linear        = Linear
Activation_ReLU          = ReLU
Activation_LeakyReLU     = LeakyReLU
Activation_ELU           = ELU
Activation_SELU          = SELU
Activation_Sigmoid       = Sigmoid
Activation_Tanh          = Tanh
Activation_Softmax       = Softmax
Activation_Softplus      = Softplus
Activation_Swish         = Swish
Activation_Mish          = Mish
Activation_GELU          = GELU
Activation_PReLU         = PReLU
Activation_Sine          = Sine
Activation_Hardswish     = Hardswish
Activation_ReLU6         = ReLU6
Activation_Hardsigmoid   = Hardsigmoid
Activation_LogSoftmax    = LogSoftmax
Activation_Sparsemax     = Sparsemax
Activation_CELU          = CELU
Activation_Softsign      = Softsign
Activation_Tanhshrink    = Tanhshrink

# Losses
Loss_MSE      = MeanSquaredError
Loss_MAE      = MeanAbsoluteError
Loss_Huber    = HuberLoss
Loss_BCE      = BinaryCrossentropy
Loss_CCE      = CategoricalCrossentropy
Loss_SparseCCE = SparseCategoricalCrossentropy
Loss_KLD      = KLDivergence

# Optimizers
Optimizer_SGD       = SGD
Optimizer_Adam      = Adam
Optimizer_AdamW     = AdamW
Optimizer_RMSprop   = RMSprop
Optimizer_Adagrad   = Adagrad
Optimizer_Adadelta  = Adadelta
Optimizer_Nadam     = Nadam
Optimizer_RAdam     = RAdam
Optimizer_Lion      = Lion
Optimizer_LAMB      = LAMB
Optimizer_Lookahead = Lookahead
Optimizer_Adan      = Adan

# ── __all__ ───────────────────────────────────────────────────────────────────

__all__ = [
    # ── Activations ──
    "Activation", "Linear",
    "ReLU", "LeakyReLU", "ELU", "SELU",
    "Sigmoid", "Tanh", "Softmax", "Softplus",
    "Swish", "Mish", "GELU", "PReLU", "Sine",
    "Hardswish", "BentIdentity", "Squareplus",
    "ReLU6", "Hardsigmoid", "LogSoftmax", "Sparsemax",
    "CELU", "Softsign", "Tanhshrink",
    "get_activation",
    # ── Layers ──
    "Layer",
    "Dense", "Conv1D", "Conv2D",
    "MaxPooling1D", "AveragePooling1D",
    "MaxPooling2D", "AveragePooling2D",
    "GlobalAveragePooling1D", "GlobalMaxPooling1D",
    "GlobalAveragePooling2D", "GlobalMaxPooling2D",
    "Dropout", "SpatialDropout2D",
    "BatchNormalization", "LayerNormalization",
    "Flatten", "Reshape",
    "SimpleRNN", "LSTM", "GRU",
    "Embedding", "PositionalEncoding",
    "MultiHeadAttention", "TransformerBlock",
    "GLU", "SwiGLU",
    "Add", "Concatenate", "Residual", "TimeDistributed",
    # ── Losses ──
    "Loss",
    "MeanSquaredError", "MeanAbsoluteError", "HuberLoss",
    "BinaryCrossentropy", "CategoricalCrossentropy",
    "SparseCategoricalCrossentropy", "KLDivergence",
    "MSE", "MAE", "Huber", "BCE", "CCE", "SparseCCE", "KLD",
    "get_loss",
    # ── Optimizers ──
    "Optimizer",
    "SGD", "Adam", "AdamW", "RMSprop", "Adagrad", "Adadelta",
    "Nadam", "RAdam", "Lion", "LAMB", "Lookahead", "Adan",
    "get_optimizer",
    # ── Models ──
    "Sequential", "Model", "GraphModel", "Input",
    # ── Utilities ──
    "to_categorical", "train_test_split",
    "EarlyStopping", "ReduceLROnPlateau",
    "Trainer", "Checkpoint",
    "accuracy", "binary_accuracy", "categorical_accuracy",
    "precision", "recall", "f1_score", "r2_score",
    "confusion_matrix", "get_metric",
    # ── NNFS aliases ──
    "Layer_Dense", "Layer_Dropout", "Layer_Flatten", "Layer_Reshape",
    "Layer_Conv2D", "Layer_Conv1D",
    "Layer_MaxPooling2D", "Layer_AveragePooling2D",
    "Layer_MaxPooling1D", "Layer_AveragePooling1D",
    "Layer_LSTM", "Layer_GRU", "Layer_SimpleRNN",
    "Layer_Embedding", "Layer_BatchNorm", "Layer_LayerNorm",
    "Layer_Residual", "Layer_TimeDistributed", "Layer_TransformerBlock",
    "Activation_Linear", "Activation_ReLU", "Activation_LeakyReLU",
    "Activation_ELU", "Activation_SELU",
    "Activation_Sigmoid", "Activation_Tanh", "Activation_Softmax",
    "Activation_Softplus", "Activation_Swish", "Activation_Mish",
    "Activation_GELU", "Activation_PReLU", "Activation_Sine",
    "Activation_Hardswish", "Activation_ReLU6", "Activation_Hardsigmoid",
    "Activation_LogSoftmax", "Activation_Sparsemax",
    "Activation_CELU", "Activation_Softsign", "Activation_Tanhshrink",
    "Loss_MSE", "Loss_MAE", "Loss_Huber",
    "Loss_BCE", "Loss_CCE", "Loss_SparseCCE", "Loss_KLD",
    "Optimizer_SGD", "Optimizer_Adam", "Optimizer_AdamW",
    "Optimizer_RMSprop", "Optimizer_Adagrad", "Optimizer_Adadelta",
    "Optimizer_Nadam", "Optimizer_RAdam", "Optimizer_Lion",
    "Optimizer_LAMB", "Optimizer_Lookahead", "Optimizer_Adan",
]
