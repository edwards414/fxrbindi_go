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
* The value head keeps AlphaGo Zero's tanh scalar, sized down for 9x9.
"""
from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp


class GlobalPoolBias(nn.Module):
    """KataGo-style global pooling -> channel bias."""

    channels: int

    @nn.compact
    def __call__(self, x):
        # x: (B, H, W, C)
        g = jnp.concatenate([x.mean(axis=(1, 2)), x.max(axis=(1, 2))], axis=-1)
        g = nn.Dense(self.channels)(nn.relu(nn.Dense(self.channels)(g)))
        return x + g[:, None, None, :]


class ResBlock(nn.Module):
    channels: int
    use_gpool: bool

    @nn.compact
    def __call__(self, x):
        h = nn.Conv(self.channels, (3, 3), use_bias=False)(x)
        h = nn.GroupNorm(num_groups=8)(h)
        h = nn.relu(h)
        h = nn.Conv(self.channels, (3, 3), use_bias=False)(h)
        h = nn.GroupNorm(num_groups=8)(h)
        if self.use_gpool:
            h = GlobalPoolBias(self.channels)(h)
        return nn.relu(x + h)


class AZNet(nn.Module):
    """Policy + value network.

    Input observation: (B, H, W, C_in) as produced by pgx Go environments.
    Returns (policy_logits (B, num_actions), value (B,)).
    """

    num_actions: int
    channels: int = 128
    num_blocks: int = 8

    @nn.compact
    def __call__(self, x):
        x = x.astype(jnp.float32)
        h = nn.Conv(self.channels, (3, 3), use_bias=False)(x)
        h = nn.GroupNorm(num_groups=8)(h)
        h = nn.relu(h)
        for i in range(self.num_blocks):
            h = ResBlock(self.channels, use_gpool=(i % 2 == 1))(h)

        # Policy head
        p = nn.Conv(4, (1, 1), use_bias=False)(h)
        p = nn.GroupNorm(num_groups=1)(p)
        p = nn.relu(p)
        p = p.reshape((p.shape[0], -1))
        logits = nn.Dense(self.num_actions)(p)

        # Value head
        v = nn.Conv(2, (1, 1), use_bias=False)(h)
        v = nn.GroupNorm(num_groups=1)(v)
        v = nn.relu(v)
        v = v.reshape((v.shape[0], -1))
        v = nn.relu(nn.Dense(128)(v))
        v = nn.tanh(nn.Dense(1)(v))
        return logits, v.squeeze(-1)
