"""Shared inference-time search utilities (used by evaluate.py and gtp.py).

train.py keeps its own jitted copy of this logic inside its pmap closures;
if you change the search semantics here, mirror it there.
"""
from __future__ import annotations

import pickle

import jax
import jax.numpy as jnp
import mctx
import pgx

from gozero.net import AZNet


def load_ckpt(path):
    """Returns (env, net, params, ckpt_dict)."""
    with open(path, "rb") as f:
        ck = pickle.load(f)
    cfg = ck["config"]
    env = pgx.make(cfg["env_id"])
    net = AZNet(
        num_actions=env.num_actions,
        channels=cfg["channels"],
        num_blocks=cfg["blocks"],
    )
    return env, net, ck["params"], ck


def make_search_fn(env, net, num_simulations: int, max_considered: int = 16):
    """Returns a jitted fn (params, key, batched_states) -> policy_output.

    With num_simulations == 0 the returned fn does a raw-policy argmax and
    fabricates a matching policy_output-like namespace (action only).
    """

    def forward(params, obs):
        return net.apply({"params": params}, obs)

    def recurrent_fn(params, rng_key, action, state):
        del rng_key
        current_player = state.current_player
        state = jax.vmap(env.step)(state, action)
        logits, value = forward(params, state.observation)
        logits = jnp.where(state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)
        reward = state.rewards[jnp.arange(state.rewards.shape[0]), current_player]
        value = jnp.where(state.terminated, 0.0, value)
        discount = jnp.where(state.terminated, 0.0, -1.0 * jnp.ones_like(value))
        return (
            mctx.RecurrentFnOutput(
                reward=reward, discount=discount, prior_logits=logits, value=value
            ),
            state,
        )

    @jax.jit
    def search(params, key, states):
        logits, value = forward(params, states.observation)
        masked = jnp.where(states.legal_action_mask, logits, jnp.finfo(logits.dtype).min)
        if num_simulations == 0:
            return masked.argmax(axis=-1), value
        root = mctx.RootFnOutput(prior_logits=masked, value=value, embedding=states)
        po = mctx.gumbel_muzero_policy(
            params=params,
            rng_key=key,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=num_simulations,
            invalid_actions=~states.legal_action_mask,
            qtransform=mctx.qtransform_completed_by_mix_value,
            max_num_considered_actions=max_considered,
            gumbel_scale=1.0,
        )
        return po.action, value

    return search


def batch_of_one(state):
    """Add a leading batch axis of 1 to an unbatched pgx state."""
    return jax.tree_util.tree_map(lambda x: x[None], state)


def unbatch(state):
    return jax.tree_util.tree_map(lambda x: x[0], state)
