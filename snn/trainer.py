"""
Trainer — advanced training loop with gradient accumulation,
mixed-precision simulation, and multi-metric checkpointing.
"""

import numpy as np
import time
import os


class Checkpoint:
    """
    Saves and restores model weights when a monitored metric improves.

    Parameters
    ----------
    monitor : str
        Name of the metric to watch (e.g. ``'val_loss'``, ``'val_accuracy'``).
    mode : str
        ``'min'`` for loss-like metrics, ``'max'`` for accuracy-like metrics,
        ``'auto'`` to infer from the metric name.
    save_path : str, optional
        Path prefix for the ``.npz`` file. Defaults to ``'checkpoint'``.
    verbose : int
        Verbosity level (0 = silent, 1 = print on save).
    """

    def __init__(self, monitor="val_loss", mode="auto",
                 save_path="checkpoint", verbose=1):
        self.monitor = monitor
        self.save_path = save_path
        self.verbose = verbose
        self._best = None
        self._best_weights = None

        if mode == "auto":
            self._is_better = (lambda cur, best: cur < best
                               if "loss" in monitor
                               else lambda cur, best: cur > best)
        elif mode == "min":
            self._is_better = lambda cur, best: cur < best
        else:
            self._is_better = lambda cur, best: cur > best

    def update(self, current, model):
        if self._best is None or self._is_better(current, self._best):
            self._best = current
            self._best_weights = model.get_weights()
            if self.verbose:
                print(f"  ✓ Checkpoint saved  {self.monitor}={current:.6f}")
            return True
        return False

    def restore(self, model):
        if self._best_weights is not None:
            model.set_weights(self._best_weights)
            if self._best is not None:
                print(f"  ✓ Best weights restored  {self.monitor}={self._best:.6f}")

    def save_to_disk(self, model):
        model.save_weights(self.save_path)


class Trainer:
    """
    Advanced training loop for a ``Sequential`` model.

    Extends the built-in ``model.fit()`` with:

    - **Gradient accumulation** — accumulate gradients over N micro-batches
      before a weight update.  Useful for effective large batch sizes when
      memory is limited.
    - **Mixed-precision simulation** — casts activations/gradients to
      ``float16`` and immediately back to ``float64`` during the forward/
      backward pass, simulating the precision loss of hardware FP16 without
      requiring actual hardware support.
    - **Multi-metric checkpointing** — track any number of metrics and save
      the best weights for each independently.
    - **Callbacks** — plug in ``EarlyStopping``, ``ReduceLROnPlateau``, or
      any callable ``(epoch, history, model, optimizer) → None``.

    Parameters
    ----------
    model : Sequential
        Compiled ``Sequential`` model.
    gradient_accumulation_steps : int
        Number of micro-batches to accumulate before calling
        ``optimizer.apply_gradients``.  Effective batch size =
        ``batch_size × gradient_accumulation_steps``.
    mixed_precision : bool
        Simulate FP16 mixed precision.
    clip_grad_norm : float or None
        If set, clip the global gradient norm to this value before each
        optimizer step.
    checkpoints : list[Checkpoint] or None
        One or more ``Checkpoint`` objects.  Each monitors its own metric.
    verbose : int
        0 = silent, 1 = one line per epoch, 2 = one line per step.

    Examples
    --------
    >>> from snn.trainer import Trainer, Checkpoint
    >>> ckpt = Checkpoint(monitor='val_loss', save_path='best_model')
    >>> trainer = Trainer(
    ...     model,
    ...     gradient_accumulation_steps=4,
    ...     mixed_precision=True,
    ...     clip_grad_norm=1.0,
    ...     checkpoints=[ckpt],
    ... )
    >>> history = trainer.fit(
    ...     X_train, y_train,
    ...     epochs=50,
    ...     batch_size=32,
    ...     validation_data=(X_val, y_val),
    ... )
    >>> ckpt.restore(model)   # reload best weights
    """

    def __init__(self, model, gradient_accumulation_steps=1,
                 mixed_precision=False, clip_grad_norm=None,
                 checkpoints=None, verbose=1):
        self.model = model
        self.gradient_accumulation_steps = max(1, int(gradient_accumulation_steps))
        self.mixed_precision = mixed_precision
        self.clip_grad_norm = clip_grad_norm
        self.checkpoints = checkpoints or []
        self.verbose = verbose
        self.history = {}

    def _cast_fp16(self, x):
        """Simulate FP16 loss by casting down and back up."""
        return x.astype(np.float16).astype(np.float64)

    def _forward(self, x, training=True):
        out = x
        for layer in self.model._layers:
            out = layer.forward(out, training=training)
            if self.mixed_precision:
                out = self._cast_fp16(out)
        return out

    def _backward_accumulated(self, grad):
        """Run backward pass and scale grad by accumulation steps."""
        grad = grad / self.gradient_accumulation_steps
        if self.mixed_precision:
            grad = self._cast_fp16(grad)
        for layer in reversed(self.model._layers):
            grad = layer.backward(grad)
            if self.mixed_precision:
                grad = self._cast_fp16(grad)
        return grad

    def _accumulate_grads(self, accumulated, model):
        """Add current layer grads into the accumulation buffer."""
        for i, layer in enumerate(model._layers):
            if not layer.trainable:
                continue
            for k, g in layer.grads.items():
                key = f"L{i}_{k}"
                if key in accumulated:
                    accumulated[key] = accumulated[key] + g
                else:
                    accumulated[key] = g.copy()

    def _apply_accumulated_grads(self, accumulated, model):
        """Apply the accumulated + optionally clipped gradient dict."""
        if self.clip_grad_norm is not None:
            total_norm = np.sqrt(sum(np.sum(g ** 2) for g in accumulated.values()))
            scale = min(1.0, self.clip_grad_norm / (total_norm + 1e-8))
            accumulated = {k: v * scale for k, v in accumulated.items()}

        for i, layer in enumerate(model._layers):
            if not layer.trainable:
                continue
            p = layer.params
            if not p:
                continue
            pfx = f"L{i}_"
            layer_p = {pfx + k: v for k, v in p.items()}
            layer_g = {pfx + k: accumulated[pfx + k]
                       for k in p if pfx + k in accumulated}
            if not layer_g:
                continue
            updated = model._optimizer.apply_gradients(layer_p, layer_g)
            for k in p:
                new_val = updated.get(pfx + k)
                if new_val is not None and hasattr(layer, k):
                    setattr(layer, k, new_val)

    def fit(self, x, y, epochs=10, batch_size=32, validation_data=None,
            shuffle=True, callbacks=None):
        """
        Train the model.

        Parameters
        ----------
        x : ndarray
            Input features, shape ``(N, ...)``.
        y : ndarray
            Targets, shape ``(N, ...)``.
        epochs : int
            Total number of passes over the dataset.
        batch_size : int
            Micro-batch size (before accumulation).
        validation_data : tuple (x_val, y_val), optional
            If provided, compute validation metrics each epoch.
        shuffle : bool
            Shuffle training data each epoch.
        callbacks : list, optional
            List of callables invoked as ``cb(epoch, history, model, optimizer)``.

        Returns
        -------
        dict
            Training history.
        """
        if not self.model._compiled:
            raise RuntimeError("Call model.compile() before trainer.fit().")

        model = self.model
        opt = model._optimizer
        loss_fn = model._loss_fn
        metric_fns = model._metric_fns
        metric_names = model._metric_names
        n = x.shape[0]

        self.history = {k: [] for k in ["loss"] + metric_names}
        if validation_data is not None:
            self.history.update({f"val_{k}": [] for k in ["loss"] + metric_names})

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            if shuffle:
                idx = np.random.permutation(n)
                x, y = x[idx], y[idx]

            epoch_loss = 0.0
            epoch_preds, epoch_targets = [], []
            n_batches = 0
            accumulated_grads = {}
            acc_step = 0

            for start in range(0, n, batch_size):
                xb = x[start:start + batch_size]
                yb = y[start:start + batch_size]

                y_pred = self._forward(xb, training=True)
                loss = loss_fn.forward(y_pred, yb)
                grad = loss_fn.backward(y_pred, yb)

                self._backward_accumulated(grad)
                self._accumulate_grads(accumulated_grads, model)

                epoch_loss += loss
                epoch_preds.append(y_pred)
                epoch_targets.append(yb)
                n_batches += 1
                acc_step += 1

                if acc_step >= self.gradient_accumulation_steps or \
                        (start + batch_size) >= n:
                    self._apply_accumulated_grads(accumulated_grads, model)
                    accumulated_grads = {}
                    acc_step = 0

                    if self.verbose >= 2:
                        step_num = start // batch_size + 1
                        total_steps = (n + batch_size - 1) // batch_size
                        print(f"  step {step_num}/{total_steps} — loss: {loss:.4f}")

            epoch_loss /= n_batches
            self.history["loss"].append(epoch_loss)

            all_preds = np.concatenate(epoch_preds, axis=0)
            all_targets = np.concatenate(epoch_targets, axis=0)

            metric_vals = {}
            for name, fn in zip(metric_names, metric_fns):
                val = fn(all_preds, all_targets)
                self.history[name].append(val)
                metric_vals[name] = val

            val_str = ""
            if validation_data is not None:
                xv, yv = validation_data
                val_pred = self._forward(xv, training=False)
                val_loss = loss_fn.forward(val_pred, yv)
                self.history["val_loss"].append(val_loss)
                val_str = f" — val_loss: {val_loss:.4f}"

                for name, fn in zip(metric_names, metric_fns):
                    vval = fn(val_pred, yv)
                    self.history[f"val_{name}"].append(vval)
                    val_str += f" — val_{name}: {vval:.4f}"

                for ckpt in self.checkpoints:
                    key = ckpt.monitor
                    current = self.history.get(key, [None])[-1]
                    if current is not None:
                        ckpt.update(current, model)

            if self.verbose >= 1:
                elapsed = time.time() - t0
                mp_tag = " [fp16]" if self.mixed_precision else ""
                ga_tag = f" [ga×{self.gradient_accumulation_steps}]" \
                    if self.gradient_accumulation_steps > 1 else ""
                metric_str = "".join(
                    f" — {k}: {v:.4f}" for k, v in metric_vals.items()
                )
                print(f"Epoch {epoch}/{epochs}{mp_tag}{ga_tag} [{elapsed:.2f}s]"
                      f" — loss: {epoch_loss:.4f}{metric_str}{val_str}")

            if callbacks:
                try:
                    for cb in callbacks:
                        if callable(cb):
                            cb(epoch, self.history, model, opt)
                except StopIteration:
                    print(f"Training stopped early at epoch {epoch}.")
                    break

        return self.history
