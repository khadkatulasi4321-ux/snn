import numpy as np
from .base import Layer
from ..initializers import get as get_initializer


class Embedding(Layer):
    """
    Token embedding layer.

    Maps integer token indices to dense float vectors.  This is the
    standard first layer for any NLP or sequence model.

    Parameters
    ----------
    vocab_size : int
        Number of unique tokens (size of the vocabulary).
    embed_dim : int
        Dimensionality of the embedding vectors.
    embeddings_initializer : str
        Initialiser for the embedding matrix (default ``"random_normal"``).

    Input shape
    -----------
    ``(batch, seq_len)`` — integer token indices in ``[0, vocab_size)``.

    Output shape
    ------------
    ``(batch, seq_len, embed_dim)`` — dense embedding vectors.

    Examples
    --------
    >>> emb = Embedding(vocab_size=1000, embed_dim=64)
    >>> x = np.array([[1, 5, 23, 0], [4, 2, 9, 7]])   # (2, 4) integer tokens
    >>> out = emb.forward(x)   # (2, 4, 64)
    """

    def __init__(self, vocab_size, embed_dim,
                 embeddings_initializer="random_normal", name=None):
        super().__init__(name=name)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self._init = get_initializer(embeddings_initializer)
        self.W = None
        self._dW = None
        self._input = None

    def build(self, input_shape):
        self.W = self._init((self.vocab_size, self.embed_dim)) * 0.02
        self._built = True

    def forward(self, x, training=False):
        if not self._built:
            self.build(x.shape)
        self._input = x.astype(int)
        return self.W[self._input]      # (batch, seq_len, embed_dim)

    def backward(self, grad):
        # grad: (batch, seq_len, embed_dim)
        # Scatter-add gradient into the embedding rows that were looked up.
        self._dW = np.zeros_like(self.W)
        np.add.at(self._dW, self._input, grad)
        self._dW /= grad.shape[0]       # normalise by batch size
        # No meaningful gradient flows back to integer indices
        return np.zeros_like(self._input, dtype=np.float64)

    @property
    def params(self):
        return {"W": self.W} if self.W is not None else {}

    @property
    def grads(self):
        return {"W": self._dW} if self._dW is not None else {}

    def count_params(self):
        return self.vocab_size * self.embed_dim

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"vocab_size": self.vocab_size, "embed_dim": self.embed_dim})
        return cfg


class PositionalEncoding(Layer):
    """
    Sinusoidal Positional Encoding (Vaswani et al. 2017).

    Adds a fixed, non-learned position signal to a sequence embedding so
    that the model can distinguish token order.  The encoding is defined
    by::

        PE[pos, 2i]   = sin(pos / 10000^(2i/d))
        PE[pos, 2i+1] = cos(pos / 10000^(2i/d))

    This layer has **no trainable parameters** — it just adds the PE
    matrix to the incoming embedding and passes the gradient straight
    through.

    Parameters
    ----------
    max_seq_len : int
        Maximum sequence length to pre-compute (default 512).

    Input / output shape
    --------------------
    ``(batch, seq_len, embed_dim)`` → same shape (PE added in-place).

    Examples
    --------
    >>> pe = PositionalEncoding(max_seq_len=128)
    >>> x = np.random.randn(4, 20, 64)    # (batch, seq, embed_dim)
    >>> out = pe.forward(x)               # (4, 20, 64) — PE added
    """

    def __init__(self, max_seq_len=512, name=None):
        super().__init__(trainable=False, name=name)
        self.max_seq_len = max_seq_len
        self._pe_cache = {}

    def _get_pe(self, embed_dim):
        if embed_dim not in self._pe_cache:
            pe = np.zeros((self.max_seq_len, embed_dim))
            pos = np.arange(self.max_seq_len)[:, np.newaxis]
            i = np.arange(0, embed_dim, 2)
            div = np.exp(i * (-np.log(10000.0) / embed_dim))
            pe[:, 0::2] = np.sin(pos * div)
            # handle odd embed_dim
            n_cos = len(np.arange(1, embed_dim, 2))
            pe[:, 1::2] = np.cos(pos * div[:n_cos])
            self._pe_cache[embed_dim] = pe
        return self._pe_cache[embed_dim]

    def forward(self, x, training=False):
        # x: (batch, seq_len, embed_dim)
        seq_len, embed_dim = x.shape[1], x.shape[2]
        pe = self._get_pe(embed_dim)[:seq_len]   # (seq_len, embed_dim)
        return x + pe[np.newaxis, :, :]

    def backward(self, grad):
        # PE is fixed — gradient passes through unchanged
        return grad
