"""
attention.py — Multi-Head Self-Attention and Transformer block.

These layers are the building blocks of Transformer architectures and
are suitable for NLP, sequence modelling, and any task where long-range
dependencies matter.
"""
import numpy as np
from .base import Layer
from .batchnorm import LayerNormalization
from .dense import Dense
from .dropout import Dropout


class MultiHeadAttention(Layer):
    """
    Multi-Head Self-Attention (Vaswani et al. 2017).

    Computes scaled dot-product attention in parallel over ``num_heads``
    heads, then projects the concatenated result back to ``embed_dim``.

    Attention(Q, K, V) = softmax(Q·Kᵀ / √d_k) · V

    Each head operates on a ``head_dim = embed_dim // num_heads`` subspace.

    Parameters
    ----------
    embed_dim : int
        Total embedding dimensionality (must be divisible by ``num_heads``).
    num_heads : int
        Number of parallel attention heads.
    dropout_rate : float
        Dropout applied to the attention weight matrix during training.

    Input shape
    -----------
    ``(batch, seq_len, embed_dim)``

    Output shape
    ------------
    ``(batch, seq_len, embed_dim)``

    Examples
    --------
    >>> attn = MultiHeadAttention(embed_dim=64, num_heads=4)
    >>> x = np.random.randn(2, 10, 64)
    >>> out = attn.forward(x)   # (2, 10, 64)
    """

    def __init__(self, embed_dim, num_heads, dropout_rate=0.0, name=None):
        super().__init__(name=name)
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout_rate = dropout_rate
        self.scale = np.sqrt(self.head_dim)

        # Projection weights — lazily built
        self.W_q = self.W_k = self.W_v = self.W_o = None
        self.b_q = self.b_k = self.b_v = self.b_o = None
        self._dW_q = self._dW_k = self._dW_v = self._dW_o = None
        self._db_q = self._db_k = self._db_v = self._db_o = None
        self._cache = None
        self._drop_mask = None

    def build(self, input_shape):
        d = self.embed_dim
        scale = 1.0 / np.sqrt(d)
        rng = np.random.default_rng()
        self.W_q = rng.normal(0, scale, (d, d))
        self.W_k = rng.normal(0, scale, (d, d))
        self.W_v = rng.normal(0, scale, (d, d))
        self.W_o = rng.normal(0, scale, (d, d))
        self.b_q = np.zeros((1, 1, d))
        self.b_k = np.zeros((1, 1, d))
        self.b_v = np.zeros((1, 1, d))
        self.b_o = np.zeros((1, 1, d))
        self._built = True

    # ── head reshape helpers ─────────────────────────────────────────────

    def _split_heads(self, x):
        """(batch, seq, d) → (batch, heads, seq, head_dim)"""
        b, s, _ = x.shape
        return x.reshape(b, s, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

    def _merge_heads(self, x):
        """(batch, heads, seq, head_dim) → (batch, seq, d)"""
        b, h, s, hd = x.shape
        return x.transpose(0, 2, 1, 3).reshape(b, s, self.embed_dim)

    @staticmethod
    def _softmax(x):
        shifted = x - np.max(x, axis=-1, keepdims=True)
        e = np.exp(shifted)
        return e / (e.sum(axis=-1, keepdims=True) + 1e-9)

    # ── forward ─────────────────────────────────────────────────────────

    def forward(self, x, training=False, mask=None):
        if not self._built:
            self.build(x.shape)

        batch = x.shape[0]

        Q = x @ self.W_q + self.b_q       # (batch, seq, d)
        K = x @ self.W_k + self.b_k
        V = x @ self.W_v + self.b_v

        Q_h = self._split_heads(Q)         # (batch, heads, seq, head_dim)
        K_h = self._split_heads(K)
        V_h = self._split_heads(V)

        # Scaled dot-product scores
        scores = Q_h @ K_h.transpose(0, 1, 3, 2) / self.scale   # (batch, heads, seq, seq)
        if mask is not None:
            scores = scores + mask * -1e9

        attn_w = self._softmax(scores)     # (batch, heads, seq, seq)

        # Optional attention dropout
        if training and self.dropout_rate > 0.0:
            self._drop_mask = (
                np.random.random(attn_w.shape) > self.dropout_rate
            ).astype(np.float64)
            attn = attn_w * self._drop_mask / (1.0 - self.dropout_rate)
        else:
            self._drop_mask = None
            attn = attn_w

        ctx_h = attn @ V_h                 # (batch, heads, seq, head_dim)
        ctx = self._merge_heads(ctx_h)     # (batch, seq, d)

        out = ctx @ self.W_o + self.b_o

        self._cache = (x, Q_h, K_h, V_h, scores, attn_w, attn, ctx_h, ctx)
        return out

    # ── backward ─────────────────────────────────────────────────────────

    def backward(self, grad):
        x, Q_h, K_h, V_h, scores, attn_w, attn, ctx_h, ctx = self._cache
        batch = x.shape[0]
        d = self.embed_dim

        # ── output projection ──
        self._dW_o = ctx.reshape(-1, d).T @ grad.reshape(-1, d) / batch
        self._db_o = grad.sum(axis=(0, 1), keepdims=True) / batch
        d_ctx = grad @ self.W_o.T                         # (batch, seq, d)

        d_ctx_h = self._split_heads(d_ctx)                # (batch, heads, seq, head_dim)

        # Un-apply dropout
        if self._drop_mask is not None:
            d_ctx_h = d_ctx_h / (1.0 - self.dropout_rate)

        # ── attention context: ctx = attn @ V ──
        # d_attn: (batch, heads, seq, seq)
        d_attn = d_ctx_h @ V_h.transpose(0, 1, 3, 2)
        # d_V:   (batch, heads, seq, head_dim)
        d_V_h = attn.transpose(0, 1, 3, 2) @ d_ctx_h

        # ── softmax backward ──
        d_scores = attn_w * (d_attn - (d_attn * attn_w).sum(axis=-1, keepdims=True))
        d_scores = d_scores / self.scale

        # ── Q·Kᵀ backward ──
        d_Q_h = d_scores @ K_h                            # (batch, heads, seq, head_dim)
        d_K_h = d_scores.transpose(0, 1, 3, 2) @ Q_h     # (batch, heads, seq, head_dim)

        # ── merge heads back ──
        d_Q = self._merge_heads(d_Q_h)    # (batch, seq, d)
        d_K = self._merge_heads(d_K_h)
        d_V = self._merge_heads(d_V_h)

        # ── input projections ──
        x_flat = x.reshape(-1, d)
        self._dW_q = x_flat.T @ d_Q.reshape(-1, d) / batch
        self._dW_k = x_flat.T @ d_K.reshape(-1, d) / batch
        self._dW_v = x_flat.T @ d_V.reshape(-1, d) / batch
        self._db_q = d_Q.sum(axis=(0, 1), keepdims=True) / batch
        self._db_k = d_K.sum(axis=(0, 1), keepdims=True) / batch
        self._db_v = d_V.sum(axis=(0, 1), keepdims=True) / batch

        dx = d_Q @ self.W_q.T + d_K @ self.W_k.T + d_V @ self.W_v.T
        return dx

    # ── params / grads ───────────────────────────────────────────────────

    @property
    def params(self):
        if not self._built:
            return {}
        return dict(
            W_q=self.W_q, W_k=self.W_k, W_v=self.W_v, W_o=self.W_o,
            b_q=self.b_q, b_k=self.b_k, b_v=self.b_v, b_o=self.b_o,
        )

    @property
    def grads(self):
        if self._dW_q is None:
            return {}
        return dict(
            W_q=self._dW_q, W_k=self._dW_k, W_v=self._dW_v, W_o=self._dW_o,
            b_q=self._db_q, b_k=self._db_k, b_v=self._db_v, b_o=self._db_o,
        )


class TransformerBlock(Layer):
    """
    Transformer encoder block (Pre-LayerNorm variant).

    Architecture per block::

        x → LayerNorm → MultiHeadAttention → Dropout → + x (residual)
          → LayerNorm → FFN(ff_dim) → Dropout → + x (residual)

    The FFN consists of two Dense layers:
    ``Dense(ff_dim, activation) → Dense(embed_dim)``.

    Pre-LN (applied before each sub-layer) is more numerically stable
    than Post-LN and typically requires no learning-rate warmup.

    Parameters
    ----------
    embed_dim : int
        Model dimensionality.
    num_heads : int
        Number of attention heads.
    ff_dim : int
        Hidden size of the feed-forward network (usually 2–4 × embed_dim).
    dropout_rate : float
        Dropout applied inside attention and after each sub-layer.
    activation : str
        Activation used in the FFN hidden layer (default ``"gelu"``).

    Input / output shape
    --------------------
    ``(batch, seq_len, embed_dim)`` → same shape.

    Examples
    --------
    Simple language classifier::

        model = Sequential([
            Embedding(vocab_size=5000, embed_dim=64),
            PositionalEncoding(),
            TransformerBlock(embed_dim=64, num_heads=4, ff_dim=128),
            GlobalAveragePooling1D(),
            Dense(2, activation="softmax"),
        ])
    """

    def __init__(self, embed_dim, num_heads, ff_dim,
                 dropout_rate=0.1, activation="gelu", name=None):
        super().__init__(name=name)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate

        self._ln1 = LayerNormalization()
        self._attn = MultiHeadAttention(embed_dim, num_heads, dropout_rate=dropout_rate)
        self._drop1 = Dropout(dropout_rate)
        self._ln2 = LayerNormalization()
        self._ff1 = Dense(ff_dim, activation=activation)
        self._ff2 = Dense(embed_dim)
        self._drop2 = Dropout(dropout_rate)

    # ── forward ─────────────────────────────────────────────────────────

    def forward(self, x, training=False):
        # ── attention sub-layer (Pre-LN) ──
        residual = x
        x = self._ln1.forward(x, training=training)
        x = self._attn.forward(x, training=training)
        x = self._drop1.forward(x, training=training)
        x = x + residual

        # ── FFN sub-layer (Pre-LN) ──
        residual = x
        x = self._ln2.forward(x, training=training)
        x = self._ff1.forward(x, training=training)
        x = self._ff2.forward(x, training=training)
        x = self._drop2.forward(x, training=training)
        x = x + residual

        return x

    # ── backward ─────────────────────────────────────────────────────────

    def backward(self, grad):
        # ── FFN sub-layer backward ──
        d = self._drop2.backward(grad)
        d = self._ff2.backward(d)
        d = self._ff1.backward(d)
        d = self._ln2.backward(d)
        grad = grad + d             # residual

        # ── attention sub-layer backward ──
        d = self._drop1.backward(grad)
        d = self._attn.backward(d)
        d = self._ln1.backward(d)
        grad = grad + d             # residual

        return grad

    # ── params / grads — flat namespace with sub-layer prefix ────────────

    @property
    def params(self):
        p = {}
        p.update({f"ln1_{k}": v for k, v in self._ln1.params.items()})
        p.update({f"attn_{k}": v for k, v in self._attn.params.items()})
        p.update({f"ln2_{k}": v for k, v in self._ln2.params.items()})
        p.update({f"ff1_{k}": v for k, v in self._ff1.params.items()})
        p.update({f"ff2_{k}": v for k, v in self._ff2.params.items()})
        return p

    @property
    def grads(self):
        g = {}
        g.update({f"ln1_{k}": v for k, v in self._ln1.grads.items()})
        g.update({f"attn_{k}": v for k, v in self._attn.grads.items()})
        g.update({f"ln2_{k}": v for k, v in self._ln2.grads.items()})
        g.update({f"ff1_{k}": v for k, v in self._ff1.grads.items()})
        g.update({f"ff2_{k}": v for k, v in self._ff2.grads.items()})
        return g

    def set_param(self, key, val):
        """Route a prefixed parameter update to the correct sub-layer."""
        _SUB = {
            "ln1": self._ln1,
            "attn": self._attn,
            "ln2": self._ln2,
            "ff1": self._ff1,
            "ff2": self._ff2,
        }
        for prefix, sub in _SUB.items():
            pfx = prefix + "_"
            if key.startswith(pfx):
                sub_key = key[len(pfx):]
                sub.set_param(sub_key, val)
                return
        # fallback
        if hasattr(self, key):
            setattr(self, key, val)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout_rate": self.dropout_rate,
        })
        return cfg
