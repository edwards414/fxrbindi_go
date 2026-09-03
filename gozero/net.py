"""AlphaZero-style policy+value network with post-AlphaGo-Zero improvements.

Differences from the original AlphaGo Zero (2017) architecture:

* KataGo-style global pooling bias in alternating residual blocks: each such
  block computes per-channel (mean, max) over the whole board and injects the
  result back as a channel-wise bias.  Plain 3x3 convolution stacks need many
  layers before whole-board information (ko fights, overall territory balance)
  reaches every point; global pooling makes it available immediately.
* GroupNorm instead of BatchNorm: deterministic at any batch size (batch=1
  GTP play behaves identically to training) and requires no cross-device
  statistics synchronisation under pmap.
* The value head keeps AlphaGo Zero's tanh scalar.  Policy/value dense layers
  derive their shapes from the board, so the same trunk supports 9x9 and 19x19.
"""
from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp


class GlobalPoolBias(nn.Module):
    """KataGo-style global pooling -> channel bias."""

    channels: int
    compute_dtype: str = "float32"

    @nn.compact
    def __call__(self, x):
        dtype = jnp.bfloat16 if self.compute_dtype == "bfloat16" else jnp.float32
        # x: (B, H, W, C)
        g = jnp.concatenate([x.mean(axis=(1, 2)), x.max(axis=(1, 2))], axis=-1)
        g = nn.Dense(self.channels, dtype=dtype, param_dtype=jnp.float32)(
            nn.relu(nn.Dense(self.channels, dtype=dtype, param_dtype=jnp.float32)(g))
        )
        return x + g[:, None, None, :]


class ResBlock(nn.Module):
    channels: int
    use_gpool: bool
    compute_dtype: str = "float32"

    @nn.compact
    def __call__(self, x):
        dtype = jnp.bfloat16 if self.compute_dtype == "bfloat16" else jnp.float32
        h = nn.Conv(
            self.channels, (3, 3), use_bias=False,
            dtype=dtype, param_dtype=jnp.float32,
        )(x)
        h = nn.GroupNorm(num_groups=8, dtype=dtype, param_dtype=jnp.float32)(h)
        h = nn.relu(h)
        h = nn.Conv(
            self.channels, (3, 3), use_bias=False,
            dtype=dtype, param_dtype=jnp.float32,
        )(h)
        h = nn.GroupNorm(num_groups=8, dtype=dtype, param_dtype=jnp.float32)(h)
        if self.use_gpool:
            h = GlobalPoolBias(self.channels, self.compute_dtype)(h)
        return nn.relu(x + h)


class AZNet(nn.Module):
    """Policy + value network.

    Input observation: (B, H, W, C_in) as produced by pgx Go environments.
    Returns (policy_logits (B, num_actions), value (B,)).
    """

    num_actions: int
    channels: int = 128
    num_blocks: int = 8
    compute_dtype: str = "float32"

    @nn.compact
    def __call__(self, x):
        dtype = jnp.bfloat16 if self.compute_dtype == "bfloat16" else jnp.float32
        x = x.astype(dtype)
        h = nn.Conv(
            self.channels, (3, 3), use_bias=False,
            dtype=dtype, param_dtype=jnp.float32,
        )(x)
        h = nn.GroupNorm(num_groups=8, dtype=dtype, param_dtype=jnp.float32)(h)
        h = nn.relu(h)
        for i in range(self.num_blocks):
            h = ResBlock(
                self.channels,
                use_gpool=(i % 2 == 1),
                compute_dtype=self.compute_dtype,
            )(h)

        # Policy head
        p = nn.Conv(
            4, (1, 1), use_bias=False,
            dtype=dtype, param_dtype=jnp.float32,
        )(h)
        p = nn.GroupNorm(num_groups=1, dtype=dtype, param_dtype=jnp.float32)(p)
        p = nn.relu(p)
        p = p.reshape((p.shape[0], -1))
        logits = nn.Dense(
            self.num_actions, dtype=dtype, param_dtype=jnp.float32,
        )(p)

        # Value head
        v = nn.Conv(
            2, (1, 1), use_bias=False,
            dtype=dtype, param_dtype=jnp.float32,
        )(h)
        v = nn.GroupNorm(num_groups=1, dtype=dtype, param_dtype=jnp.float32)(v)
        v = nn.relu(v)
        v = v.reshape((v.shape[0], -1))
        v = nn.relu(nn.Dense(128, dtype=dtype, param_dtype=jnp.float32)(v))
        v = nn.tanh(nn.Dense(1, dtype=dtype, param_dtype=jnp.float32)(v))
        # MCTS and the losses are numerically safer in float32; the expensive
        # convolution/dense kernels above still use H100 bfloat16 tensor cores.
        return logits.astype(jnp.float32), v.squeeze(-1).astype(jnp.float32)
