import numpy as np


def to_categorical(y, num_classes=None):
    """Convert integer class labels to one-hot vectors."""
    y = np.asarray(y, dtype=int).flatten()
    if num_classes is None:
        num_classes = np.max(y) + 1
    one_hot = np.zeros((len(y), num_classes), dtype=np.float64)
    one_hot[np.arange(len(y)), y] = 1.0
    return one_hot


def normalize(x, axis=None):
    """Min-max normalize to [0, 1]."""
    x_min = np.min(x, axis=axis, keepdims=True)
    x_max = np.max(x, axis=axis, keepdims=True)
    return (x - x_min) / (x_max - x_min + 1e-8)


def standardize(x, axis=None):
    """Standardize to zero mean and unit variance."""
    mu = np.mean(x, axis=axis, keepdims=True)
    sigma = np.std(x, axis=axis, keepdims=True)
    return (x - mu) / (sigma + 1e-8)


def train_test_split(x, y, test_size=0.2, shuffle=True, seed=None):
    """Split arrays into train and test sets."""
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    idx = rng.permutation(n) if shuffle else np.arange(n)
    n_test = int(n * test_size)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return x[train_idx], x[test_idx], y[train_idx], y[test_idx]


def batch_generator(x, y, batch_size=32, shuffle=True):
    """Yield (x_batch, y_batch) mini-batches."""
    n = x.shape[0]
    idx = np.random.permutation(n) if shuffle else np.arange(n)
    for start in range(0, n, batch_size):
        batch_idx = idx[start:start + batch_size]
        yield x[batch_idx], y[batch_idx]


def clip_gradients(grads, max_norm=5.0):
    """Global gradient norm clipping."""
    total_norm = np.sqrt(sum(np.sum(g ** 2) for g in grads.values()))
    scale = min(1.0, max_norm / (total_norm + 1e-8))
    return {k: v * scale for k, v in grads.items()}


def learning_rate_schedule(initial_lr, decay_rate=0.96, decay_steps=1000):
    """Return a callable exponential LR schedule."""
    def schedule(step):
        return initial_lr * (decay_rate ** (step / decay_steps))
    return schedule


def cosine_annealing(initial_lr, t_max, eta_min=0.0):
    """Return a callable cosine annealing LR schedule."""
    def schedule(step):
        return eta_min + 0.5 * (initial_lr - eta_min) * (
            1 + np.cos(np.pi * step / t_max)
        )
    return schedule


def warmup_schedule(initial_lr, warmup_steps, base_schedule=None):
    """Linear warmup followed by an optional base schedule."""
    def schedule(step):
        if step < warmup_steps:
            return initial_lr * (step + 1) / warmup_steps
        if base_schedule is not None:
            return base_schedule(step - warmup_steps)
        return initial_lr
    return schedule


class EarlyStopping:
    """
    Stop training when a monitored metric has stopped improving.

    Parameters
    ----------
    monitor : str
        Name of the metric to watch (e.g. 'val_loss').
    patience : int
        Number of epochs with no improvement before stopping.
    min_delta : float
        Minimum change to qualify as an improvement.
    mode : str
        'min' for loss-like metrics, 'max' for accuracy-like.
    restore_best_weights : bool
        If True, restore the best weights when training stops.
    """

    def __init__(self, monitor="val_loss", patience=5, min_delta=0.0,
                 mode="auto", restore_best_weights=False, verbose=1):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        self._best = None
        self._counter = 0
        self._best_weights = None
        self._stopped = False

        if mode == "auto":
            self._is_better = (lambda cur, best: cur < best - min_delta
                               if "loss" in monitor
                               else lambda cur, best: cur > best + min_delta)
        elif mode == "min":
            self._is_better = lambda cur, best: cur < best - min_delta
        else:
            self._is_better = lambda cur, best: cur > best + min_delta

    def __call__(self, epoch, history, model=None):
        if self.monitor not in history:
            return
        current = history[self.monitor][-1]
        if self._best is None or self._is_better(current, self._best):
            self._best = current
            self._counter = 0
            if self.restore_best_weights and model is not None:
                self._best_weights = model.get_weights()
        else:
            self._counter += 1
            if self.verbose:
                print(f"EarlyStopping: no improvement for {self._counter}/{self.patience} epochs")
            if self._counter >= self.patience:
                self._stopped = True
                if self.verbose:
                    print(f"EarlyStopping: stopping at epoch {epoch}.")
                if self.restore_best_weights and self._best_weights and model:
                    model.set_weights(self._best_weights)
                raise StopIteration

    @property
    def stopped(self):
        return self._stopped


class ReduceLROnPlateau:
    """
    Reduce learning rate when a metric has stopped improving.

    Parameters
    ----------
    monitor : str
    factor : float
        Factor by which the learning rate will be reduced.
    patience : int
    min_lr : float
        Lower bound for the learning rate.
    verbose : int
    """

    def __init__(self, monitor="val_loss", factor=0.5, patience=3,
                 min_lr=1e-6, verbose=1):
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.verbose = verbose
        self._best = None
        self._counter = 0

    def __call__(self, epoch, history, optimizer=None):
        if self.monitor not in history:
            return
        current = history[self.monitor][-1]
        if self._best is None or current < self._best:
            self._best = current
            self._counter = 0
        else:
            self._counter += 1
            if self._counter >= self.patience and optimizer is not None:
                new_lr = max(optimizer.learning_rate * self.factor, self.min_lr)
                if new_lr < optimizer.learning_rate:
                    optimizer.learning_rate = new_lr
                    if self.verbose:
                        print(f"ReduceLROnPlateau: lr reduced to {new_lr:.2e}")
                self._counter = 0
