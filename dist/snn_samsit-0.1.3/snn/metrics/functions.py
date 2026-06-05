import numpy as np


def accuracy(y_pred, y_true):
    """Classification accuracy."""
    if y_pred.ndim > 1 and y_pred.shape[-1] > 1:
        pred_labels = np.argmax(y_pred, axis=-1)
    else:
        pred_labels = (y_pred.flatten() >= 0.5).astype(int)

    if y_true.ndim > 1 and y_true.shape[-1] > 1:
        true_labels = np.argmax(y_true, axis=-1)
    else:
        true_labels = y_true.flatten().astype(int)

    return np.mean(pred_labels == true_labels)


def binary_accuracy(y_pred, y_true, threshold=0.5):
    return np.mean((y_pred.flatten() >= threshold) == y_true.flatten().astype(bool))


def categorical_accuracy(y_pred, y_true):
    return np.mean(np.argmax(y_pred, axis=-1) == np.argmax(y_true, axis=-1))


def sparse_categorical_accuracy(y_pred, y_true):
    return np.mean(np.argmax(y_pred, axis=-1) == y_true.flatten().astype(int))


def top_k_accuracy(y_pred, y_true, k=5):
    top_k = np.argsort(y_pred, axis=-1)[:, -k:]
    true_labels = np.argmax(y_true, axis=-1)
    return np.mean([true_labels[i] in top_k[i] for i in range(len(true_labels))])


def precision(y_pred, y_true, threshold=0.5):
    pred = (y_pred.flatten() >= threshold).astype(int)
    true = y_true.flatten().astype(int)
    tp = np.sum((pred == 1) & (true == 1))
    fp = np.sum((pred == 1) & (true == 0))
    return tp / (tp + fp + 1e-8)


def recall(y_pred, y_true, threshold=0.5):
    pred = (y_pred.flatten() >= threshold).astype(int)
    true = y_true.flatten().astype(int)
    tp = np.sum((pred == 1) & (true == 1))
    fn = np.sum((pred == 0) & (true == 1))
    return tp / (tp + fn + 1e-8)


def f1_score(y_pred, y_true, threshold=0.5):
    p = precision(y_pred, y_true, threshold)
    r = recall(y_pred, y_true, threshold)
    return 2 * p * r / (p + r + 1e-8)


def mean_squared_error(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)


def mean_absolute_error(y_pred, y_true):
    return np.mean(np.abs(y_pred - y_true))


def r2_score(y_pred, y_true):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / (ss_tot + 1e-8)


def confusion_matrix(y_pred, y_true, n_classes=None):
    if y_pred.ndim > 1 and y_pred.shape[-1] > 1:
        pred_labels = np.argmax(y_pred, axis=-1)
    else:
        pred_labels = (y_pred.flatten() >= 0.5).astype(int)

    if y_true.ndim > 1 and y_true.shape[-1] > 1:
        true_labels = np.argmax(y_true, axis=-1)
    else:
        true_labels = y_true.flatten().astype(int)

    if n_classes is None:
        n_classes = max(pred_labels.max(), true_labels.max()) + 1

    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(true_labels, pred_labels):
        cm[t, p] += 1
    return cm


_REGISTRY = {
    "accuracy": accuracy,
    "binary_accuracy": binary_accuracy,
    "categorical_accuracy": categorical_accuracy,
    "sparse_categorical_accuracy": sparse_categorical_accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1_score,
    "f1_score": f1_score,
    "mse": mean_squared_error,
    "mean_squared_error": mean_squared_error,
    "mae": mean_absolute_error,
    "mean_absolute_error": mean_absolute_error,
    "r2": r2_score,
    "r2_score": r2_score,
}


def get(identifier):
    if callable(identifier):
        return identifier
    if isinstance(identifier, str):
        key = identifier.lower()
        if key in _REGISTRY:
            return _REGISTRY[key]
        raise ValueError(f"Unknown metric: '{identifier}'. Available: {list(_REGISTRY)}")
    raise TypeError(f"Could not interpret metric: {identifier}")
