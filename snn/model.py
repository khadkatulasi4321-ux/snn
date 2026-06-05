import numpy as np
import time
from .layers.base import Layer
from .losses import get as get_loss
from .optimizers import get as get_optimizer
from .metrics import get as get_metric


class Sequential:
    """
    Sequential model — a linear stack of layers.

    Build, compile, then train:

    >>> model = Sequential([
    ...     Dense(64, activation='relu'),
    ...     Dropout(0.3),
    ...     Dense(10, activation='softmax'),
    ... ])
    >>> model.compile(
    ...     optimizer='adam',
    ...     loss='categorical_crossentropy',
    ...     metrics=['accuracy'],
    ... )
    >>> history = model.fit(X_train, y_train, epochs=20, batch_size=32,
    ...                     validation_data=(X_val, y_val))
    >>> y_pred = model.predict(X_test)
    >>> results = model.evaluate(X_test, y_test)
    """

    def __init__(self, layers=None):
        self._layers = []
        self._loss_fn = None
        self._optimizer = None
        self._metric_fns = []
        self._metric_names = []
        self._compiled = False
        self._clip_norm = None
        self.history = {}

        if layers:
            for layer in layers:
                self.add(layer)

    def add(self, layer):
        """Append a layer to the model.

        Parameters
        ----------
        layer : Layer
            Any :class:`snn.layers.base.Layer` subclass.

        Returns
        -------
        self
            Allows chaining: ``model.add(Dense(64)).add(Dense(10))``.
        """
        if not isinstance(layer, Layer):
            raise TypeError(f"Expected a Layer, got {type(layer)}")
        self._layers.append(layer)
        return self

    # ------------------------------------------------------------------
    # compile
    # ------------------------------------------------------------------

    def compile(
        self,
        optimizer="adam",
        loss="mse",
        metrics=None,
        *,
        learning_rate=None,
        weight_decay=None,
        clip_norm=None,
        from_logits=False,
        verbose=False,
    ):
        """Configure the model for training.

        Parameters
        ----------
        optimizer : str, Optimizer, or dict
            Weight-update rule. Three forms accepted:

            * **String shorthand** — ``"adam"``, ``"sgd"``, ``"adamw"``, …
            * **Instance** — ``Adam(learning_rate=3e-4)``
            * **Config dict** — ``{"name": "adam", "learning_rate": 1e-3}``
              Any key recognised by the optimizer constructor can appear in
              the dict; the ``"name"`` (or ``"class"``) key selects the
              optimizer class.

        loss : str, Loss, or dict
            Objective function. Three forms accepted:

            * **String shorthand** — ``"mse"``, ``"categorical_crossentropy"``, …
            * **Instance** — ``HuberLoss(delta=2.0)``
            * **Config dict** — ``{"name": "huber", "delta": 2.0}``

        metrics : str or list, optional
            Metric(s) tracked per epoch. Accepts a single string or a list
            of strings / callables.

        learning_rate : float, optional
            Shortcut to override the optimizer's learning rate without
            constructing an instance manually. Applied after the optimizer is
            built from ``optimizer``.

        weight_decay : float, optional
            Shortcut to set ``weight_decay`` on the optimizer (if supported).

        clip_norm : float, optional
            Clip the global gradient L2 norm to this value before each
            optimizer step.  Equivalent to using
            ``Trainer(model, clip_grad_norm=clip_norm)`` but available
            directly on the model.

        from_logits : bool
            If ``True``, the loss receives raw (unnormalised) scores. Passed
            automatically to ``BinaryCrossentropy`` and
            ``CategoricalCrossentropy`` when constructed from a string or dict.
            Has no effect when you pass a ``Loss`` instance directly.

        verbose : bool
            If ``True``, prints a compile-summary table showing the chosen
            optimizer, loss, metrics, and any extra settings.

        Returns
        -------
        self

        Examples
        --------
        Minimal:

        >>> model.compile("adam", "mse")

        Learning-rate shortcut (no need to import Adam):

        >>> model.compile("adam", "categorical_crossentropy",
        ...               metrics=["accuracy"], learning_rate=3e-4)

        Dict config — all hyperparameters in one place:

        >>> model.compile(
        ...     optimizer={"name": "adamw", "learning_rate": 1e-3,
        ...                "weight_decay": 1e-4},
        ...     loss={"name": "huber", "delta": 2.0},
        ...     metrics=["r2"],
        ... )

        With gradient clipping (no Trainer needed):

        >>> model.compile("adam", "mse", clip_norm=1.0)

        Logit input to loss (last Dense has no activation):

        >>> model.compile("adam", "categorical_crossentropy",
        ...               from_logits=True)
        """
        # ── optimizer ────────────────────────────────────────────────────
        from .optimizers import _REGISTRY as _OPT_REG

        if isinstance(optimizer, dict):
            cfg = dict(optimizer)
            name = cfg.pop("name", cfg.pop("class", "adam")).lower()
            if learning_rate is not None:
                cfg["learning_rate"] = learning_rate
            if weight_decay is not None:
                cfg["weight_decay"] = weight_decay
            if name not in _OPT_REG:
                raise ValueError(
                    f"Unknown optimizer '{name}'. Available: {list(_OPT_REG)}"
                )
            self._optimizer = _OPT_REG[name](**cfg)
        else:
            self._optimizer = get_optimizer(optimizer)
            if learning_rate is not None:
                self._optimizer.learning_rate = learning_rate
            if weight_decay is not None and hasattr(self._optimizer, "weight_decay"):
                self._optimizer.weight_decay = weight_decay

        # ── loss ─────────────────────────────────────────────────────────
        from .losses import _REGISTRY as _LOSS_REG

        if isinstance(loss, dict):
            cfg = dict(loss)
            name = cfg.pop("name", cfg.pop("class", "mse")).lower()
            if from_logits and "crossentropy" in name and "from_logits" not in cfg:
                cfg["from_logits"] = True
            if name not in _LOSS_REG:
                raise ValueError(
                    f"Unknown loss '{name}'. Available: {list(_LOSS_REG)}"
                )
            self._loss_fn = _LOSS_REG[name](**cfg)
        elif from_logits and isinstance(loss, str) and "crossentropy" in loss:
            self._loss_fn = _LOSS_REG[loss.lower()](from_logits=True)
        else:
            self._loss_fn = get_loss(loss)

        # ── metrics ──────────────────────────────────────────────────────
        self._metric_fns = []
        self._metric_names = []
        if metrics is not None:
            if not isinstance(metrics, list):
                metrics = [metrics]
            for m in metrics:
                fn = get_metric(m)
                self._metric_fns.append(fn)
                name = m if isinstance(m, str) else getattr(m, "__name__", str(m))
                self._metric_names.append(name)

        # ── clip_norm ────────────────────────────────────────────────────
        self._clip_norm = clip_norm

        self._compiled = True

        if verbose:
            self._print_compile_summary()

        return self

    # ------------------------------------------------------------------
    # compile helpers
    # ------------------------------------------------------------------

    def _print_compile_summary(self):
        w = 52
        print("┌" + "─" * w + "┐")
        print(f"│{'  Compile Summary':^{w}}│")
        print("├" + "─" * w + "┤")
        opt_name = type(self._optimizer).__name__
        lr = getattr(self._optimizer, "learning_rate", "?")
        wd = getattr(self._optimizer, "weight_decay", None)
        print(f"│  {'Optimizer':<18} {opt_name} (lr={lr}{',' if wd else ''}"
              f"{' wd=' + str(wd) if wd else ''}){'':<{max(0,w-35-len(opt_name)-len(str(lr)))}}│")
        loss_name = type(self._loss_fn).__name__
        fl = getattr(self._loss_fn, "from_logits", False)
        print(f"│  {'Loss':<18} {loss_name}"
              f"{' (from_logits)' if fl else ''}"
              f"{'':<{max(0, w - 22 - len(loss_name) - (14 if fl else 0))}}│")
        met_str = ", ".join(self._metric_names) if self._metric_names else "none"
        print(f"│  {'Metrics':<18} {met_str:<{w-21}}│")
        clip_str = str(self._clip_norm) if self._clip_norm else "off"
        print(f"│  {'Clip norm':<18} {clip_str:<{w-21}}│")
        print("└" + "─" * w + "┘")

    def get_compile_config(self):
        """Return a plain-dict snapshot of the current compile settings.

        Useful for logging, serialising experiments, or re-compiling a new
        model with the same settings.

        Returns
        -------
        dict
            Keys: ``"optimizer"``, ``"loss"``, ``"metrics"``,
            ``"learning_rate"``, ``"weight_decay"``, ``"clip_norm"``.
        """
        if not self._compiled:
            return {}
        return {
            "optimizer": type(self._optimizer).__name__,
            "learning_rate": getattr(self._optimizer, "learning_rate", None),
            "weight_decay": getattr(self._optimizer, "weight_decay", None),
            "loss": type(self._loss_fn).__name__,
            "from_logits": getattr(self._loss_fn, "from_logits", False),
            "metrics": list(self._metric_names),
            "clip_norm": self._clip_norm,
        }

    # ------------------------------------------------------------------
    # forward / backward
    # ------------------------------------------------------------------

    def _forward(self, x, training=False):
        out = x
        for layer in self._layers:
            out = layer.forward(out, training=training)
        return out

    def _backward(self, grad):
        for layer in reversed(self._layers):
            grad = layer.backward(grad)
        return grad

    # ------------------------------------------------------------------
    # gradient clipping
    # ------------------------------------------------------------------

    def _clip_gradients(self):
        """Clip the global gradient L2 norm in-place across all layers."""
        sq_sum = 0.0
        for layer in self._layers:
            if not layer.trainable:
                continue
            for g in layer.grads.values():
                if g is not None:
                    sq_sum += np.sum(g ** 2)
            act = getattr(layer, "activation", None)
            if act is not None:
                ag = getattr(act, "grads", None)
                if ag:
                    for g in ag.values():
                        if g is not None:
                            sq_sum += np.sum(g ** 2)

        global_norm = np.sqrt(sq_sum) + 1e-8
        if global_norm <= self._clip_norm:
            return
        scale = self._clip_norm / global_norm

        for layer in self._layers:
            if not layer.trainable:
                continue
            g_dict = layer.grads
            for k, g in g_dict.items():
                if g is not None:
                    # modify the underlying stored gradient in-place via attr
                    grad_attr = f"_d{k}"
                    if hasattr(layer, grad_attr):
                        setattr(layer, grad_attr, g * scale)
            act = getattr(layer, "activation", None)
            if act is not None:
                ag = getattr(act, "grads", None)
                if ag:
                    if hasattr(act, "_d_alpha"):
                        act._d_alpha = act._d_alpha * scale

    # ------------------------------------------------------------------
    # optimizer step
    # ------------------------------------------------------------------

    def _apply_optimizer(self):
        if self._clip_norm is not None:
            self._clip_gradients()

        for i, layer in enumerate(self._layers):
            if not layer.trainable:
                continue
            p = layer.params
            g = layer.grads
            if p and g:
                # Prefix keys with layer index so optimizer state never collides
                # across layers that share parameter names (e.g. every Dense has "W")
                pfx = f"L{i}_"
                prefixed_p = {pfx + k: v for k, v in p.items()}
                prefixed_g = {pfx + k: v for k, v in g.items() if k in p}
                updated = self._optimizer.apply_gradients(prefixed_p, prefixed_g)
                for k in p:
                    new_val = updated[pfx + k]
                    layer.set_param(k, new_val)

            # Update learnable activation params (e.g. PReLU alpha).
            # Uses a distinct prefix "L{i}_act_" to avoid any collision with
            # layer params that share the same key name.
            act = getattr(layer, "activation", None)
            if act is not None:
                ap = getattr(act, "params", None)
                ag = getattr(act, "grads", None)
                if ap and ag:
                    pfx = f"L{i}_act_"
                    prefixed_p = {pfx + k: v for k, v in ap.items()}
                    prefixed_g = {pfx + k: v for k, v in ag.items() if k in ap}
                    updated = self._optimizer.apply_gradients(prefixed_p, prefixed_g)
                    for k in ap:
                        if hasattr(act, k):
                            setattr(act, k, updated[pfx + k])

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, x, y, epochs=10, batch_size=32, validation_data=None,
            shuffle=True, verbose=1, callbacks=None):
        """Train the model.

        Parameters
        ----------
        x : ndarray
            Input data of shape ``(N, ...)``.
        y : ndarray
            Target data of shape ``(N, ...)``.
        epochs : int
            Number of full passes over the dataset.
        batch_size : int
            Number of samples per gradient update.
        validation_data : tuple, optional
            ``(X_val, y_val)`` evaluated at the end of each epoch.
        shuffle : bool
            Whether to shuffle the training data each epoch.
        verbose : int
            ``0`` = silent, ``1`` = one line per epoch.
        callbacks : list, optional
            Callables invoked at the end of each epoch as
            ``callback(epoch, history)``.

        Returns
        -------
        dict
            Training history. Keys are ``"loss"``, each metric name, and
            ``"val_*"`` equivalents when ``validation_data`` is provided.
        """
        if not self._compiled:
            raise RuntimeError(
                "Model must be compiled before training. Call model.compile()."
            )

        n = x.shape[0]
        self.history = {k: [] for k in ["loss"] + self._metric_names}
        if validation_data is not None:
            self.history.update({f"val_{k}": [] for k in ["loss"] + self._metric_names})

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            if shuffle:
                idx = np.random.permutation(n)
                x, y = x[idx], y[idx]

            epoch_loss = 0.0
            epoch_preds = []
            epoch_targets = []
            n_batches = 0

            for start in range(0, n, batch_size):
                xb = x[start:start + batch_size]
                yb = y[start:start + batch_size]

                y_pred = self._forward(xb, training=True)
                loss = self._loss_fn.forward(y_pred, yb)
                grad = self._loss_fn.backward(y_pred, yb)

                self._backward(grad)
                self._apply_optimizer()

                epoch_loss += loss
                epoch_preds.append(y_pred)
                epoch_targets.append(yb)
                n_batches += 1

            epoch_loss /= n_batches
            self.history["loss"].append(epoch_loss)

            all_preds = np.concatenate(epoch_preds, axis=0)
            all_targets = np.concatenate(epoch_targets, axis=0)

            metric_vals = {}
            for name, fn in zip(self._metric_names, self._metric_fns):
                val = fn(all_preds, all_targets)
                self.history[name].append(val)
                metric_vals[name] = val

            val_str = ""
            if validation_data is not None:
                xv, yv = validation_data
                val_pred = self._forward(xv, training=False)
                val_loss = self._loss_fn.forward(val_pred, yv)
                self.history["val_loss"].append(val_loss)
                val_str = f" — val_loss: {val_loss:.4f}"
                for name, fn in zip(self._metric_names, self._metric_fns):
                    vval = fn(val_pred, yv)
                    self.history[f"val_{name}"].append(vval)
                    val_str += f" — val_{name}: {vval:.4f}"

            if verbose:
                elapsed = time.time() - t0
                metric_str = "".join(
                    f" — {k}: {v:.4f}" for k, v in metric_vals.items()
                )
                print(f"Epoch {epoch}/{epochs} [{elapsed:.2f}s]"
                      f" — loss: {epoch_loss:.4f}{metric_str}{val_str}")

            if callbacks:
                try:
                    for cb in callbacks:
                        cb(epoch, self.history)
                except StopIteration:
                    if verbose:
                        print(f"  Early stopping at epoch {epoch}.")
                    break

        return self.history

    # ------------------------------------------------------------------
    # predict / evaluate
    # ------------------------------------------------------------------

    def predict(self, x, batch_size=None):
        """Run inference on ``x``.

        Parameters
        ----------
        x : ndarray
        batch_size : int, optional
            If given, runs inference in mini-batches (saves memory for large
            datasets).

        Returns
        -------
        ndarray
            Predictions of shape ``(N, output_units)``.
        """
        if batch_size is None:
            return self._forward(x, training=False)

        results = []
        for start in range(0, x.shape[0], batch_size):
            xb = x[start:start + batch_size]
            results.append(self._forward(xb, training=False))
        return np.concatenate(results, axis=0)

    def evaluate(self, x, y, batch_size=None, verbose=1):
        """Compute loss and metrics on ``(x, y)``.

        Parameters
        ----------
        x : ndarray
        y : ndarray
        batch_size : int, optional
        verbose : int

        Returns
        -------
        dict
            ``{"loss": float, metric_name: float, …}``
        """
        y_pred = self.predict(x, batch_size=batch_size)
        loss = self._loss_fn.forward(y_pred, y)
        results = {"loss": loss}
        for name, fn in zip(self._metric_names, self._metric_fns):
            results[name] = fn(y_pred, y)

        if verbose:
            parts = (
                [f"loss: {loss:.4f}"]
                + [f"{k}: {v:.4f}" for k, v in results.items() if k != "loss"]
            )
            print(" — ".join(parts))

        return results

    # ------------------------------------------------------------------
    # introspection
    # ------------------------------------------------------------------

    def summary(self):
        """Print a table of layers, output shapes, and parameter counts."""
        print("=" * 65)
        print(f"{'Layer':<25} {'Output Shape':<20} {'Params':>10}")
        print("=" * 65)
        total = 0
        for i, layer in enumerate(self._layers):
            cls = type(layer).__name__
            name = layer.name or f"{cls.lower()}_{i}"
            n_p = layer.count_params()
            total += n_p
            print(f"{name:<25} {'?':<20} {n_p:>10,}")
        print("=" * 65)
        print(f"Total params: {total:,}")
        print("=" * 65)

    # ------------------------------------------------------------------
    # weight serialisation
    # ------------------------------------------------------------------

    def get_weights(self):
        """Return a list of per-layer weight dicts (copies)."""
        return [
            {k: v.copy() for k, v in layer.params.items()}
            for layer in self._layers
        ]

    def set_weights(self, weights):
        """Restore weights from the format returned by :meth:`get_weights`."""
        for layer, w_dict in zip(self._layers, weights):
            for k, v in w_dict.items():
                if hasattr(layer, k):
                    setattr(layer, k, v.copy())

    def save_weights(self, path):
        """Save all trainable weights to a NumPy ``.npz`` file.

        Parameters
        ----------
        path : str
            Filename prefix (the ``.npz`` extension is added automatically).
        """
        weights = {}
        for i, layer in enumerate(self._layers):
            for k, v in layer.params.items():
                weights[f"layer_{i}_{k}"] = v
        np.savez(path, **weights)
        print(f"Weights saved to {path}.npz")

    def load_weights(self, path):
        """Load weights from a ``.npz`` file written by :meth:`save_weights`.

        Parameters
        ----------
        path : str
            Path to the ``.npz`` file (with or without extension).
        """
        data = np.load(path if path.endswith(".npz") else path + ".npz")
        for i, layer in enumerate(self._layers):
            for k in layer.params:
                key = f"layer_{i}_{k}"
                if key in data:
                    setattr(layer, k, data[key])
        print(f"Weights loaded from {path}")

    def __repr__(self):
        lines = ["Sequential("]
        for i, layer in enumerate(self._layers):
            lines.append(f"  ({i}) {type(layer).__name__}")
        lines.append(")")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Shared compile / training helpers (used by Model and GraphModel)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_optimizer(optimizer, learning_rate, weight_decay):
    from .optimizers import get as get_opt
    from .optimizers import _REGISTRY as _OPT_REG

    if isinstance(optimizer, dict):
        cfg = dict(optimizer)
        name = cfg.pop("name", cfg.pop("class", "adam")).lower()
        if learning_rate is not None:
            cfg["learning_rate"] = learning_rate
        if weight_decay is not None:
            cfg["weight_decay"] = weight_decay
        if name not in _OPT_REG:
            raise ValueError(f"Unknown optimizer '{name}'. Available: {list(_OPT_REG)}")
        return _OPT_REG[name](**cfg)
    else:
        opt = get_opt(optimizer)
        if learning_rate is not None:
            opt.learning_rate = learning_rate
        if weight_decay is not None and hasattr(opt, "weight_decay"):
            opt.weight_decay = weight_decay
        return opt


def _build_loss(loss, from_logits):
    from .losses import get as get_loss
    from .losses import _REGISTRY as _LOSS_REG

    if isinstance(loss, dict):
        cfg = dict(loss)
        name = cfg.pop("name", cfg.pop("class", "mse")).lower()
        if from_logits and "crossentropy" in name and "from_logits" not in cfg:
            cfg["from_logits"] = True
        if name not in _LOSS_REG:
            raise ValueError(f"Unknown loss '{name}'. Available: {list(_LOSS_REG)}")
        return _LOSS_REG[name](**cfg)
    elif from_logits and isinstance(loss, str) and "crossentropy" in loss:
        from .losses import _REGISTRY as _LOSS_REG
        return _LOSS_REG[loss.lower()](from_logits=True)
    else:
        return get_loss(loss)


def _build_metrics(metrics):
    from .metrics import get as get_metric
    fns, names = [], []
    if metrics is None:
        return fns, names
    if not isinstance(metrics, list):
        metrics = [metrics]
    for m in metrics:
        fns.append(get_metric(m))
        names.append(m if isinstance(m, str) else getattr(m, "__name__", str(m)))
    return fns, names


def _apply_optimizer_to_layers(layers_list, optimizer, clip_norm=None):
    """Run one optimizer step over an ordered list of layers."""
    if clip_norm is not None:
        sq_sum = 0.0
        for layer in layers_list:
            if not layer.trainable:
                continue
            for g in layer.grads.values():
                if g is not None:
                    sq_sum += np.sum(g ** 2)
        global_norm = np.sqrt(sq_sum) + 1e-8
        if global_norm > clip_norm:
            scale = clip_norm / global_norm
            for layer in layers_list:
                if not layer.trainable:
                    continue
                for k, g in layer.grads.items():
                    attr = f"_d{k}"
                    if g is not None and hasattr(layer, attr):
                        setattr(layer, attr, g * scale)

    for i, layer in enumerate(layers_list):
        if not layer.trainable:
            continue
        p = layer.params
        g = layer.grads
        if p and g:
            pfx = f"L{i}_"
            prefixed_p = {pfx + k: v for k, v in p.items()}
            prefixed_g = {pfx + k: v for k, v in g.items() if k in p}
            if not prefixed_g:
                continue
            updated = optimizer.apply_gradients(prefixed_p, prefixed_g)
            for k in p:
                if pfx + k in updated:
                    layer.set_param(k, updated[pfx + k])

        act = getattr(layer, "activation", None)
        if act is not None:
            ap = getattr(act, "params", None)
            ag = getattr(act, "grads", None)
            if ap and ag:
                pfx = f"L{i}_act_"
                prefixed_p = {pfx + k: v for k, v in ap.items()}
                prefixed_g = {pfx + k: v for k, v in ag.items() if k in ap}
                if prefixed_g:
                    updated = optimizer.apply_gradients(prefixed_p, prefixed_g)
                    for k in ap:
                        if hasattr(act, k):
                            setattr(act, k, updated[pfx + k])


def _fit_loop(obj, x, y, epochs, batch_size, validation_data,
              shuffle, verbose, callbacks):
    """Shared mini-batch training loop for all model types."""
    n = x.shape[0]
    obj.history = {k: [] for k in ["loss"] + obj._metric_names}
    if validation_data is not None:
        obj.history.update({f"val_{k}": [] for k in ["loss"] + obj._metric_names})

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        if shuffle:
            idx = np.random.permutation(n)
            x, y = x[idx], y[idx]

        epoch_loss, epoch_preds, epoch_targets, n_batches = 0.0, [], [], 0
        for start in range(0, n, batch_size):
            xb = x[start: start + batch_size]
            yb = y[start: start + batch_size]
            y_pred = obj._forward(xb, training=True)
            loss = obj._loss_fn.forward(y_pred, yb)
            grad = obj._loss_fn.backward(y_pred, yb)
            obj._backward(grad)
            obj._optimizer_step()
            epoch_loss += loss
            epoch_preds.append(y_pred)
            epoch_targets.append(yb)
            n_batches += 1

        epoch_loss /= n_batches
        obj.history["loss"].append(epoch_loss)
        all_preds = np.concatenate(epoch_preds, axis=0)
        all_targets = np.concatenate(epoch_targets, axis=0)

        metric_vals = {}
        for name, fn in zip(obj._metric_names, obj._metric_fns):
            val = fn(all_preds, all_targets)
            obj.history[name].append(val)
            metric_vals[name] = val

        val_str = ""
        if validation_data is not None:
            xv, yv = validation_data
            vp = obj._forward(xv, training=False)
            vl = obj._loss_fn.forward(vp, yv)
            obj.history["val_loss"].append(vl)
            val_str = f" — val_loss: {vl:.4f}"
            for name, fn in zip(obj._metric_names, obj._metric_fns):
                vval = fn(vp, yv)
                obj.history[f"val_{name}"].append(vval)
                val_str += f" — val_{name}: {vval:.4f}"

        if verbose:
            elapsed = time.time() - t0
            m_str = "".join(f" — {k}: {v:.4f}" for k, v in metric_vals.items())
            print(f"Epoch {epoch}/{epochs} [{elapsed:.2f}s]"
                  f" — loss: {epoch_loss:.4f}{m_str}{val_str}")

        if callbacks:
            try:
                for cb in callbacks:
                    cb(epoch, obj.history)
            except StopIteration:
                if verbose:
                    print(f"  Early stopping at epoch {epoch}.")
                break

    return obj.history


# ═══════════════════════════════════════════════════════════════════════════════
# Computation graph nodes (used by GraphModel + functional API)
# ═══════════════════════════════════════════════════════════════════════════════

class _Tensor:
    """
    A node in the functional computation graph.

    Created automatically when you call a :class:`~snn.layers.base.Layer`
    with an :class:`Input` or another :class:`_Tensor`.  You should never
    need to instantiate this directly.
    """

    _is_graph_tensor = True   # duck-type marker used by Layer.__call__

    def __init__(self, layer, parents):
        self.layer = layer          # the Layer that produced this node (None for Input)
        self.parents = parents      # list[_Tensor]
        self.data = None            # numpy array filled during _forward
        self.grad = None            # numpy array filled during _backward


class Input(_Tensor):
    """
    Entry-point node for the :class:`GraphModel` functional API.

    Parameters
    ----------
    shape : tuple
        Shape of a *single sample* (without the batch dimension).

    Examples
    --------
    >>> x = Input(shape=(784,))
    >>> h = Dense(128, activation='relu')(x)
    >>> out = Dense(10, activation='softmax')(h)
    >>> model = GraphModel(inputs=x, outputs=out)
    """

    def __init__(self, shape):
        super().__init__(layer=None, parents=[])
        self.shape = shape


# ═══════════════════════════════════════════════════════════════════════════════
# Model — subclassable (PyTorch / Keras-style)
# ═══════════════════════════════════════════════════════════════════════════════

class Model:
    """
    Subclassable model class — define ``call()`` and ``backward()`` yourself.

    All :class:`~snn.layers.base.Layer` instances assigned as attributes
    (or stored in lists/tuples of attributes) are auto-discovered for
    parameter collection and optimizer updates.  The compile / fit /
    predict / evaluate API is identical to :class:`Sequential`.

    Examples
    --------
    **Basic usage** (functionally identical to Sequential):

    .. code-block:: python

        class MyNet(Model):
            def __init__(self):
                super().__init__()
                self.fc1 = Dense(128, activation='relu')
                self.bn   = BatchNormalization()
                self.fc2  = Dense(10, activation='softmax')

            def call(self, x, training=False):
                x = self.fc1(x, training=training)
                x = self.bn(x, training=training)
                return self.fc2(x, training=training)

            def backward(self, grad):
                grad = self.fc2.backward(grad)
                grad = self.bn.backward(grad)
                return self.fc1.backward(grad)

        model = MyNet()
        model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=20)

    **Residual block** (skip connections):

    .. code-block:: python

        class ResBlock(Model):
            def __init__(self, units):
                super().__init__()
                self.fc1 = Dense(units, activation='relu')
                self.fc2 = Dense(units)                 # no activation here
                self.proj = Dense(units)                # shortcut projection

            def call(self, x, training=False):
                shortcut = self.proj(x, training=training)
                h = self.fc1(x, training=training)
                h = self.fc2(h, training=training)
                self._pre_act = h + shortcut
                return np.maximum(0, self._pre_act)     # ReLU after residual

            def backward(self, grad):
                d = grad * (self._pre_act > 0)          # ReLU gate
                d_proj = self.proj.backward(d)
                d = self.fc2.backward(d)
                d = self.fc1.backward(d)
                return d + d_proj
    """

    def __init__(self):
        self._loss_fn = None
        self._optimizer = None
        self._metric_fns = []
        self._metric_names = []
        self._compiled = False
        self._clip_norm = None
        self.history = {}

    # ── override these ────────────────────────────────────────────────────────

    def call(self, x, training=False):
        """Forward pass.  Override in subclasses."""
        raise NotImplementedError("Subclasses must implement call()")

    def backward(self, grad):
        """Backward pass.  Override in subclasses."""
        raise NotImplementedError("Subclasses must implement backward()")

    # ── layer discovery ───────────────────────────────────────────────────────

    def _collect_layers(self):
        """Walk self.__dict__ and return all Layer instances in order."""
        layers_out, seen = [], set()

        def _visit(obj):
            if isinstance(obj, Layer) and id(obj) not in seen:
                layers_out.append(obj)
                seen.add(id(obj))
            elif isinstance(obj, (list, tuple)):
                for item in obj:
                    _visit(item)

        for val in self.__dict__.values():
            _visit(val)
        return layers_out

    # ── training engine ───────────────────────────────────────────────────────

    def _forward(self, x, training=False):
        return self.call(x, training=training)

    def _backward(self, grad):
        return self.backward(grad)

    def _optimizer_step(self):
        _apply_optimizer_to_layers(
            self._collect_layers(), self._optimizer, self._clip_norm
        )

    # ── public API ────────────────────────────────────────────────────────────

    def compile(self, optimizer="adam", loss="mse", metrics=None, *,
                learning_rate=None, weight_decay=None, clip_norm=None,
                from_logits=False, verbose=False):
        """Configure the model — identical interface to :class:`Sequential`.

        Parameters
        ----------
        optimizer : str, Optimizer, or dict
        loss : str, Loss, or dict
        metrics : str or list, optional
        learning_rate : float, optional
        weight_decay : float, optional
        clip_norm : float, optional
        from_logits : bool
        verbose : bool
        """
        self._optimizer = _build_optimizer(optimizer, learning_rate, weight_decay)
        self._loss_fn = _build_loss(loss, from_logits)
        self._metric_fns, self._metric_names = _build_metrics(metrics)
        self._clip_norm = clip_norm
        self._compiled = True
        if verbose:
            self._print_compile_summary()
        return self

    def _print_compile_summary(self):
        w = 52
        print("┌" + "─" * w + "┐")
        print(f"│{'  Compile Summary':^{w}}│")
        print("├" + "─" * w + "┤")
        opt_name = type(self._optimizer).__name__
        lr = getattr(self._optimizer, "learning_rate", "?")
        print(f"│  {'Optimizer':<18} {opt_name} (lr={lr}){'':<{max(0, w-30-len(opt_name)-len(str(lr)))}}│")
        loss_name = type(self._loss_fn).__name__
        print(f"│  {'Loss':<18} {loss_name:<{w-21}}│")
        met_str = ", ".join(self._metric_names) if self._metric_names else "none"
        print(f"│  {'Metrics':<18} {met_str:<{w-21}}│")
        print("└" + "─" * w + "┘")

    def fit(self, x, y, epochs=10, batch_size=32, validation_data=None,
            shuffle=True, verbose=1, callbacks=None):
        """Train the model.  Identical interface to :class:`Sequential.fit`."""
        if not self._compiled:
            raise RuntimeError("Call model.compile() before training.")
        return _fit_loop(self, x, y, epochs, batch_size, validation_data,
                         shuffle, verbose, callbacks)

    def predict(self, x, batch_size=None):
        """Run inference on ``x``.  Returns predictions as a NumPy array."""
        if batch_size is None:
            return self._forward(x, training=False)
        results = []
        for start in range(0, x.shape[0], batch_size):
            results.append(self._forward(x[start: start + batch_size], training=False))
        return np.concatenate(results, axis=0)

    def evaluate(self, x, y, batch_size=None, verbose=1):
        """Compute loss and metrics on ``(x, y)``."""
        y_pred = self.predict(x, batch_size=batch_size)
        loss = self._loss_fn.forward(y_pred, y)
        results = {"loss": loss}
        for name, fn in zip(self._metric_names, self._metric_fns):
            results[name] = fn(y_pred, y)
        if verbose:
            parts = [f"loss: {loss:.4f}"] + [
                f"{k}: {v:.4f}" for k, v in results.items() if k != "loss"
            ]
            print(" — ".join(parts))
        return results

    def summary(self):
        """Print all discovered Layer types and parameter counts."""
        print("=" * 55)
        print(f"{'Model':<30} {'Params':>10}")
        print("=" * 55)
        total = 0
        for i, layer in enumerate(self._collect_layers()):
            cls = type(layer).__name__
            n_p = layer.count_params()
            total += n_p
            print(f"  ({i}) {cls:<26} {n_p:>10,}")
        print("=" * 55)
        print(f"Total params: {total:,}")
        print("=" * 55)

    def save_weights(self, path):
        """Save all trainable weights to a ``.npz`` file."""
        weights = {}
        for i, layer in enumerate(self._collect_layers()):
            for k, v in layer.params.items():
                weights[f"layer_{i}_{k}"] = v
        np.savez(path, **weights)
        print(f"Weights saved to {path}.npz")

    def load_weights(self, path):
        """Load weights from a ``.npz`` file written by :meth:`save_weights`."""
        data = np.load(path if path.endswith(".npz") else path + ".npz")
        for i, layer in enumerate(self._collect_layers()):
            for k in layer.params:
                key = f"layer_{i}_{k}"
                if key in data:
                    layer.set_param(k, data[key])
        print(f"Weights loaded from {path}")

    def get_compile_config(self):
        if not self._compiled:
            return {}
        return {
            "optimizer": type(self._optimizer).__name__,
            "learning_rate": getattr(self._optimizer, "learning_rate", None),
            "loss": type(self._loss_fn).__name__,
            "metrics": list(self._metric_names),
            "clip_norm": self._clip_norm,
        }

    def __repr__(self):
        cls = type(self).__name__
        layers = self._collect_layers()
        lines = [f"{cls}("]
        for i, layer in enumerate(layers):
            lines.append(f"  ({i}) {type(layer).__name__}")
        lines.append(")")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# GraphModel — functional API (Input + layer chaining)
# ═══════════════════════════════════════════════════════════════════════════════

class GraphModel:
    """
    Functional-API model: build arbitrary computation graphs with skip
    connections, multi-input layers, and shared weights.

    Build a graph by connecting layers to :class:`Input` nodes, then pass
    the terminal node(s) to ``GraphModel``.  The model traces the DAG,
    sorts it topologically, and handles forward / backward automatically.

    Examples
    --------
    **Simple feedforward** (equivalent to Sequential):

    .. code-block:: python

        from snn.model import Input, GraphModel
        from snn.layers import Dense

        x   = Input(shape=(784,))
        h   = Dense(256, activation='relu')(x)
        h   = Dense(128, activation='relu')(h)
        out = Dense(10, activation='softmax')(h)

        model = GraphModel(inputs=x, outputs=out)
        model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=20)

    **Residual / skip connection**:

    .. code-block:: python

        x      = Input(shape=(128,))
        skip   = Dense(64)(x)                        # shortcut
        h      = Dense(64, activation='relu')(x)
        h      = Dense(64)(h)                        # same size as skip
        merged = Add()([h, skip])                    # element-wise add
        out    = Dense(10, activation='softmax')(merged)

        model = GraphModel(inputs=x, outputs=out)

    **Multi-branch** (text + metadata fusion):

    .. code-block:: python

        text_in  = Input(shape=(300,))              # e.g. mean-pooled GloVe
        meta_in  = Input(shape=(10,))               # numerical features

        h_text   = Dense(128, 'relu')(text_in)
        h_meta   = Dense(32, 'relu')(meta_in)
        merged   = Concatenate()([h_text, h_meta])  # (batch, 160)
        out      = Dense(5, 'softmax')(merged)

        model = GraphModel(inputs=[text_in, meta_in], outputs=out)
        model.fit([X_text, X_meta], y, epochs=10)

    Parameters
    ----------
    inputs : Input or list of Input
        Entry-point node(s).
    outputs : _Tensor or list of _Tensor
        Terminal node(s) whose values are returned by ``predict``.
    """

    def __init__(self, inputs, outputs):
        self._in_nodes = inputs if isinstance(inputs, list) else [inputs]
        self._out_nodes = outputs if isinstance(outputs, list) else [outputs]
        self._order = self._topo_sort()

        # Shared state (same interface as Sequential / Model)
        self._loss_fn = None
        self._optimizer = None
        self._metric_fns = []
        self._metric_names = []
        self._compiled = False
        self._clip_norm = None
        self.history = {}

    # ── graph traversal ───────────────────────────────────────────────────────

    def _topo_sort(self):
        """Return nodes in topological order (inputs before outputs)."""
        visited, order = set(), []

        def _visit(node):
            if id(node) in visited:
                return
            visited.add(id(node))
            for parent in node.parents:
                _visit(parent)
            order.append(node)

        for out_node in self._out_nodes:
            _visit(out_node)
        return order

    def _get_all_layers(self):
        """Collect layers from all non-input nodes (preserves topo order)."""
        layers_out, seen = [], set()
        for node in self._order:
            if node.layer is None:
                continue
            if id(node.layer) not in seen:
                layers_out.append(node.layer)
                seen.add(id(node.layer))
        return layers_out

    # ── forward / backward ───────────────────────────────────────────────────

    def _forward(self, x, training=False):
        # Reset node values
        for node in self._order:
            node.data = None
            node.grad = None

        # Feed inputs
        x_list = x if isinstance(x, list) else [x]
        for in_node, val in zip(self._in_nodes, x_list):
            in_node.data = val

        # Execute in topo order
        for node in self._order:
            if node.layer is None:          # Input node
                continue
            parent_data = [p.data for p in node.parents]
            inp = parent_data[0] if len(parent_data) == 1 else parent_data
            node.data = node.layer.forward(inp, training)

        # Return single output or list
        if len(self._out_nodes) == 1:
            return self._out_nodes[0].data
        return [n.data for n in self._out_nodes]

    def _backward(self, grad):
        # Seed the output gradient
        if len(self._out_nodes) == 1:
            self._out_nodes[0].grad = grad
        else:
            for out_node, g in zip(self._out_nodes, grad):
                out_node.grad = g

        # Propagate in reverse topo order
        for node in reversed(self._order):
            if node.layer is None or node.grad is None:
                continue
            dx = node.layer.backward(node.grad)

            if isinstance(dx, list):
                # Multi-input layer (Add, Concatenate, …)
                for parent, dxi in zip(node.parents, dx):
                    parent.grad = dxi if parent.grad is None else parent.grad + dxi
            else:
                assert len(node.parents) == 1, (
                    "Single dx returned for multi-parent node. "
                    "Multi-input layers must return a list from backward()."
                )
                parent = node.parents[0]
                parent.grad = dx if parent.grad is None else parent.grad + dx

    def _optimizer_step(self):
        _apply_optimizer_to_layers(
            self._get_all_layers(), self._optimizer, self._clip_norm
        )

    # ── public API ────────────────────────────────────────────────────────────

    def compile(self, optimizer="adam", loss="mse", metrics=None, *,
                learning_rate=None, weight_decay=None, clip_norm=None,
                from_logits=False, verbose=False):
        """Configure the model.  Identical interface to :class:`Sequential`.

        Parameters
        ----------
        optimizer : str, Optimizer, or dict
        loss : str, Loss, or dict
        metrics : str or list, optional
        learning_rate : float, optional
        weight_decay : float, optional
        clip_norm : float, optional
        from_logits : bool
        verbose : bool
        """
        self._optimizer = _build_optimizer(optimizer, learning_rate, weight_decay)
        self._loss_fn = _build_loss(loss, from_logits)
        self._metric_fns, self._metric_names = _build_metrics(metrics)
        self._clip_norm = clip_norm
        self._compiled = True
        if verbose:
            self._print_compile_summary()
        return self

    def _print_compile_summary(self):
        w = 52
        print("┌" + "─" * w + "┐")
        print(f"│{'  Compile Summary':^{w}}│")
        print("├" + "─" * w + "┤")
        opt_name = type(self._optimizer).__name__
        lr = getattr(self._optimizer, "learning_rate", "?")
        print(f"│  {'Optimizer':<18} {opt_name} (lr={lr}){'':<{max(0, w-30-len(opt_name)-len(str(lr)))}}│")
        loss_name = type(self._loss_fn).__name__
        print(f"│  {'Loss':<18} {loss_name:<{w-21}}│")
        met_str = ", ".join(self._metric_names) if self._metric_names else "none"
        print(f"│  {'Metrics':<18} {met_str:<{w-21}}│")
        print(f"│  {'Nodes':<18} {len(self._order):<{w-21}}│")
        print("└" + "─" * w + "┘")

    def fit(self, x, y, epochs=10, batch_size=32, validation_data=None,
            shuffle=True, verbose=1, callbacks=None):
        """Train the model.  Identical interface to :class:`Sequential.fit`."""
        if not self._compiled:
            raise RuntimeError("Call model.compile() before training.")
        return _fit_loop(self, x, y, epochs, batch_size, validation_data,
                         shuffle, verbose, callbacks)

    def predict(self, x, batch_size=None):
        """Run inference on ``x``."""
        if batch_size is None:
            return self._forward(x, training=False)
        results = []
        for start in range(0, (x[0] if isinstance(x, list) else x).shape[0], batch_size):
            xb = ([xi[start: start + batch_size] for xi in x]
                  if isinstance(x, list) else x[start: start + batch_size])
            results.append(self._forward(xb, training=False))
        return np.concatenate(results, axis=0)

    def evaluate(self, x, y, batch_size=None, verbose=1):
        """Compute loss and metrics on ``(x, y)``."""
        y_pred = self.predict(x, batch_size=batch_size)
        loss = self._loss_fn.forward(y_pred, y)
        results = {"loss": loss}
        for name, fn in zip(self._metric_names, self._metric_fns):
            results[name] = fn(y_pred, y)
        if verbose:
            parts = [f"loss: {loss:.4f}"] + [
                f"{k}: {v:.4f}" for k, v in results.items() if k != "loss"
            ]
            print(" — ".join(parts))
        return results

    def summary(self):
        """Print all graph nodes and parameter counts."""
        print("=" * 55)
        print(f"{'Node (Layer)':<30} {'Params':>10}")
        print("=" * 55)
        total = 0
        for i, node in enumerate(self._order):
            if node.layer is None:
                print(f"  ({i}) {'Input':<26} {'—':>10}")
            else:
                cls = type(node.layer).__name__
                n_p = node.layer.count_params()
                total += n_p
                print(f"  ({i}) {cls:<26} {n_p:>10,}")
        print("=" * 55)
        print(f"Total params: {total:,}")
        print("=" * 55)

    def save_weights(self, path):
        """Save all weights to a ``.npz`` file."""
        weights = {}
        for i, layer in enumerate(self._get_all_layers()):
            for k, v in layer.params.items():
                weights[f"layer_{i}_{k}"] = v
        np.savez(path, **weights)
        print(f"Weights saved to {path}.npz")

    def load_weights(self, path):
        """Load weights from a ``.npz`` file."""
        data = np.load(path if path.endswith(".npz") else path + ".npz")
        for i, layer in enumerate(self._get_all_layers()):
            for k in layer.params:
                key = f"layer_{i}_{k}"
                if key in data:
                    layer.set_param(k, data[key])
        print(f"Weights loaded from {path}")

    def __repr__(self):
        n_in = len(self._in_nodes)
        n_out = len(self._out_nodes)
        n_nodes = len(self._order)
        return f"GraphModel(inputs={n_in}, outputs={n_out}, nodes={n_nodes})"
