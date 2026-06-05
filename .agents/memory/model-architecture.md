---
name: Model architecture
description: Three model types in snn and key implementation decisions
---
Three model types all share compile/fit/predict/evaluate interface:
- Sequential: self._layers list, built-in forward/backward chain
- Model: subclassable, users implement call() and backward(), layers auto-discovered from __dict__
- GraphModel: functional API, Input() + Layer.__call__(tensor) builds a DAG, topo sort handles forward/backward automatically

**Why:** GraphModel needs Layer.__call__ to detect _Tensor nodes. We use duck typing (getattr(x, '_is_graph_tensor', False)) instead of isinstance to avoid circular imports (base.py → model.py would be circular).

**How to apply:** When adding new merge layers (Add, Concatenate etc.), their backward() must return a list when they have multiple parents. GraphModel._backward() handles both scalar and list returns.

Shared training helpers in model.py: _build_optimizer(), _build_loss(), _build_metrics(), _apply_optimizer_to_layers(), _fit_loop(). Sequential still has its own inline loop (don't touch it — it's stable).
