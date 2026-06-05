---
name: Dense 3D backward
description: Dense.backward handles both 2D and 3D input shapes
---
Dense.backward uses inp_flat = self._input.reshape(-1, last_dim) and grad_flat = grad.reshape(-1, last_dim) before computing dW. This handles both 2D (batch, features) and 3D (batch, seq, features) inputs.

**Why:** TransformerBlock feeds 3D tensors (batch, seq, embed_dim) into Dense layers via the FFN sub-network. The naive self._input.T @ grad only works for 2D.

**How to apply:** Any new layer that feeds Dense with 3D input will work correctly. Do NOT revert to the old 2D-only code.
