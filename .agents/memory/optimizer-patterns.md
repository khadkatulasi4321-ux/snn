---
name: Optimizer patterns
description: Known gotchas when implementing snn optimizers
---
Optimizer base class __init__ immediately calls self.learning_rate = learning_rate, which triggers any property setter.

**Why Lookahead broke:** Lookahead overrides learning_rate as a property that reads self._inner. If _inner isn't set before super().__init__(), the setter fires and AttributeError is raised.

**Fix:** Always assign self._inner = optimizer BEFORE calling super().__init__() in any wrapper optimizer that overrides learning_rate as a property.

Lookahead is not in _REGISTRY (it's a wrapper, not a standalone optimizer). Use: Lookahead(Adam(lr=1e-3), k=5).
