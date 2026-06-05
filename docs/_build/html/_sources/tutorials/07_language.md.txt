# Tutorial 7 — Language Models (Transformer)

This tutorial builds two models that learn from text sequences:

1. **Simple approach** — `Embedding + LSTM + Dense` (for when you want fast training)
2. **Transformer approach** — `Embedding + PositionalEncoding + TransformerBlock + Dense`

Both are built with the same `Sequential` API you already know.

---

## Key concepts

| Layer | What it does | Input → Output |
|---|---|---|
| `Embedding` | Maps integer token IDs → dense vectors | `(batch, seq)` → `(batch, seq, d)` |
| `PositionalEncoding` | Adds sinusoidal position info | `(batch, seq, d)` → same |
| `MultiHeadAttention` | Each token attends to all others | `(batch, seq, d)` → same |
| `TransformerBlock` | Attention + FFN + residuals | `(batch, seq, d)` → same |
| `GlobalAveragePooling1D` | Collapse sequence → single vector | `(batch, seq, d)` → `(batch, d)` |

---

## Step 1 — Text preprocessing (character-level)

```python
import numpy as np

# Toy corpus
texts = [
    "the cat sat on the mat",
    "the dog ran in the park",
    "cats and dogs are animals",
    "animals live in nature",
    "the quick brown fox jumps",
    "fox and rabbit in the field",
    "nature is beautiful",
    "the park is green",
]
labels = [0, 0, 1, 1, 0, 1, 1, 0]   # 0 = has "the", 1 = doesn't

# Build character vocabulary
all_chars = sorted(set("".join(texts)))
char2idx = {c: i + 1 for i, c in enumerate(all_chars)}  # 0 = padding
idx2char = {i: c for c, i in char2idx.items()}
vocab_size = len(char2idx) + 1   # +1 for padding token

MAX_LEN = max(len(t) for t in texts)

# Encode and pad
def encode(text, max_len):
    ids = [char2idx[c] for c in text]
    ids = ids[:max_len]
    return ids + [0] * (max_len - len(ids))

X = np.array([encode(t, MAX_LEN) for t in texts])  # (8, MAX_LEN) integers
y = np.array([[l] for l in labels], dtype=np.float64)

print(f"Vocabulary size: {vocab_size}")
print(f"Max sequence length: {MAX_LEN}")
print(f"X shape: {X.shape}")
```

---

## Approach 1 — LSTM classifier (simple)

```python
from snn.model import Sequential
from snn.layers import Embedding, LSTM, Dense, Dropout

model_lstm = Sequential([
    Embedding(vocab_size=vocab_size, embed_dim=16),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(1, activation="sigmoid"),
])

model_lstm.compile("adam", "binary_crossentropy",
                   metrics=["binary_accuracy"],
                   learning_rate=1e-3)
model_lstm.fit(X, y, epochs=100, batch_size=4, verbose=0)

preds = model_lstm.predict(X).round()
print(f"LSTM accuracy: {(preds.flatten() == y.flatten()).mean():.2f}")
```

---

## Approach 2 — Transformer classifier

```python
from snn.model import Sequential
from snn.layers import (
    Embedding, PositionalEncoding,
    TransformerBlock, GlobalAveragePooling1D,
    Dense, Dropout,
)

model_tfm = Sequential([
    Embedding(vocab_size=vocab_size, embed_dim=32),
    PositionalEncoding(max_seq_len=128),
    TransformerBlock(embed_dim=32, num_heads=4, ff_dim=64, dropout_rate=0.1),
    GlobalAveragePooling1D(),       # (batch, seq, 32) → (batch, 32)
    Dropout(0.2),
    Dense(1, activation="sigmoid"),
])

model_tfm.compile("adam", "binary_crossentropy",
                  metrics=["binary_accuracy"],
                  learning_rate=5e-4)
model_tfm.fit(X, y, epochs=200, batch_size=4, verbose=0)

preds = model_tfm.predict(X).round()
print(f"Transformer accuracy: {(preds.flatten() == y.flatten()).mean():.2f}")
```

---

## Word-level tokenisation

For real NLP tasks you'll tokenise by word instead of character:

```python
# Build vocabulary from a corpus
corpus = ["the cat sat", "the dog ran", "cats and dogs"]
all_words = sorted(set(w for s in corpus for w in s.split()))
word2idx = {w: i + 1 for i, w in enumerate(all_words)}   # 0 = pad
vocab_size = len(word2idx) + 1

def encode_words(text, max_len):
    ids = [word2idx.get(w, 0) for w in text.split()]
    ids = ids[:max_len]
    return ids + [0] * (max_len - len(ids))

MAX_LEN = 10
X = np.array([encode_words(s, MAX_LEN) for s in corpus])
```

---

## Multiple Transformer layers (deeper model)

Stack `TransformerBlock` layers for more capacity:

```python
model = Sequential([
    Embedding(vocab_size=5000, embed_dim=64),
    PositionalEncoding(max_seq_len=256),
    TransformerBlock(embed_dim=64, num_heads=4, ff_dim=128),
    TransformerBlock(embed_dim=64, num_heads=4, ff_dim=128),   # 2nd block
    GlobalAveragePooling1D(),
    Dense(64, activation="gelu"),
    Dense(num_classes, activation="softmax"),
])
```

Each `TransformerBlock` has:
- `embed_dim × embed_dim × 4` attention weights (Q, K, V, O projections)
- `embed_dim × ff_dim × 2` FFN weights

For `embed_dim=64, ff_dim=128`: **65 536 params per block**.

---

## GLU and SwiGLU as FFN replacements

The default FFN in `TransformerBlock` is `Dense(ff_dim, "gelu") → Dense(embed_dim)`.

You can manually build the sub-layers with `SwiGLU` (used in LLaMA/PaLM):

```python
from snn.layers import SwiGLU

# In a custom forward loop:
# ffn_out = Dense(embed_dim)(SwiGLU(units=ff_dim)(x))
# (SwiGLU internally projects to 2*ff_dim and gates)
```

Or as a standalone block for non-standard architectures:

```python
model = Sequential([
    Embedding(vocab_size=500, embed_dim=32),
    SwiGLU(units=64),    # 32 → 64 with gating
    Dense(32),           # 64 → 32 back down
    Dense(num_classes, activation="softmax"),
])
```

---

## Causal masking (decoder / autoregressive)

Pass a causal mask to `MultiHeadAttention` to prevent each position from
attending to future positions (needed for language generation, not just
classification):

```python
from snn.layers import MultiHeadAttention
import numpy as np

def causal_mask(seq_len):
    """Upper triangular mask — 1 = blocked, 0 = allowed."""
    m = np.triu(np.ones((seq_len, seq_len)), k=1)
    return m[np.newaxis, np.newaxis, :, :]   # (1, 1, seq, seq)

attn = MultiHeadAttention(embed_dim=64, num_heads=4)
x = np.random.randn(2, 10, 64)
mask = causal_mask(10)
out = attn.forward(x, training=False, mask=mask)
```

---

## Full working example — sentiment classifier

This trains a Transformer on a small word-level dataset and achieves 100%
training accuracy in under 2 seconds:

```python
import numpy as np
from snn.model import Sequential
from snn.layers import (
    Embedding, PositionalEncoding, TransformerBlock,
    GlobalAveragePooling1D, Dense, Dropout,
)

# ----- Data ----------------------------------------------------------------
positive = ["great movie", "loved it", "fantastic film", "amazing story",
            "brilliant acting", "wonderful experience", "highly recommend",
            "truly spectacular"]
negative = ["terrible movie", "hated it", "boring film", "awful story",
            "bad acting", "poor quality", "waste of time", "very disappointing"]

texts = positive + negative
labels = [1] * len(positive) + [0] * len(negative)

all_words = sorted(set(w for t in texts for w in t.split()))
word2idx = {w: i + 1 for i, w in enumerate(all_words)}
VOCAB  = len(word2idx) + 1
MAXLEN = 3

def enc(text):
    ids = [word2idx[w] for w in text.split()]
    return ids[:MAXLEN] + [0] * (MAXLEN - len(ids))

X = np.array([enc(t) for t in texts])
y = np.array([[l] for l in labels], dtype=np.float64)

# ----- Model ---------------------------------------------------------------
model = Sequential([
    Embedding(VOCAB, embed_dim=16),
    PositionalEncoding(max_seq_len=16),
    TransformerBlock(embed_dim=16, num_heads=2, ff_dim=32, dropout_rate=0.0),
    GlobalAveragePooling1D(),
    Dense(1, activation="sigmoid"),
])

model.compile("adam", "binary_crossentropy",
              metrics=["binary_accuracy"], learning_rate=5e-3)

history = model.fit(X, y, epochs=200, batch_size=8, verbose=0)
print(f"Final accuracy: {history['binary_accuracy'][-1]:.2f}")

# ----- Inference -----------------------------------------------------------
test = np.array([enc("great acting"), enc("boring story")])
pred = model.predict(test)
for t, p in zip(["great acting", "boring story"], pred):
    sentiment = "positive" if p[0] > 0.5 else "negative"
    print(f"  '{t}' → {sentiment} ({p[0]:.3f})")
```

---

## Tips for NLP with snn

- **Embed dim 16–64** for toy tasks, 128–512 for real datasets
- **1–4 TransformerBlocks** are usually enough; more → slower without gain on small data
- **Gradient clipping** helps: `model.compile(..., clip_norm=1.0)`
- **Small LR** for Transformers: `1e-3` to `3e-4`; use RAdam or Nadam for stability
- **Padding zeros** are treated like real tokens — mask them for better accuracy on longer sequences
- For generation (autoregressive), you need causal masking (see above)

---

## What next?

- **[Tutorial 5](05_sequences.md)** — LSTM/GRU for sequences if you want a lighter model
- **[Layers guide](../guide/layers.md)** — full API docs for every layer
