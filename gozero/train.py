"""Gumbel-AlphaZero training for Go on pgx (fully on-GPU self-play).

Improvements over the original AlphaGo Zero training recipe:

* Gumbel root action selection (Danihelka et al. 2022, via mctx): an unbiased
  policy-improvement operator that works with ~32 simulations per move instead
  of AlphaGo Zero's 800, a 25x reduction in search cost per training game.
* Vectorised self-play: thousands of games run in lockstep on each GPU; the
  environment (pgx) and MCTS (mctx) are both jitted JAX code, so the pipeline
  never leaves the accelerator during an iteration.
* AdamW + warmup/cosine schedule instead of plain SGD with hand-tuned drops.

Run:
    CUDA_VISIBLE_DEVICES=0,3,4 python -m gozero.train --run-dir runs/v1
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import pickle
import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
import mctx
import numpy as np
import optax
import pgx

from gozero.net import AZNet


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", default="go_9x9")
    p.add_argument("--run-dir", default="runs/dev")
    p.add_argument("--resume", default=None, help="checkpoint .pkl to resume from")
    p.add_argument("--channels", type=int, default=128)
    p.add_argument("--blocks", type=int, default=8)
    p.add_argument("--selfplay-batch", type=int, default=1024, help="games per device per iteration")
    p.add_argument("--sims", type=int, default=32, help="MCTS simulations per move")
    p.add_argument("--max-considered", type=int, default=16, help="Gumbel max considered actions at root")
    p.add_argument("--max-steps", type=int, default=162, help="self-play scan length; 162 = pgx go_9x9 hard cap, so every game finishes in-window")
    p.add_argument("--pass-guard-ply", type=int, default=40,
                   help="selfplay-only: forbid pass at the search root before this ply "
                        "(unless no other legal move). Prevents the early pass-collapse "
                        "equilibrium where the losing side learns to concede instantly.")
    p.add_argument("--train-batch", type=int, default=2048, help="minibatch per device")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-iters", type=int, default=20)
    p.add_argument("--decay-iters", type=int, default=2000, help="cosine decay horizon in iterations")
    p.add_argument("--iters", type=int, default=100000)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--eval-batch", type=int, default=128, help="eval games per device")
    p.add_argument("--save-every", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Self-play
# ---------------------------------------------------------------------------
class SelfplayOut(NamedTuple):
    obs: jnp.ndarray            # (T, B, H, W, C)  observation before the move
    policy_tgt: jnp.ndarray     # (T, B, A)        Gumbel-improved policy target
    reward: jnp.ndarray         # (T, B)           reward for mover, seen after step
    valid: jnp.ndarray          # (T, B)           game was live when move was made
    done_after: jnp.ndarray     # (T, B)           game ended at/before end of this step
    is_pass: jnp.ndarray        # (T, B)           chosen move was a pass


class Sample(NamedTuple):
    obs: jnp.ndarray
    policy_tgt: jnp.ndarray
    value_tgt: jnp.ndarray
    mask: jnp.ndarray


def make_fns(env, net, args, num_devices):
    """Build the pmapped selfplay / target / train / eval functions."""

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
        out = mctx.RecurrentFnOutput(
            reward=reward, discount=discount, prior_logits=logits, value=value
        )
        return out, state

    batch_per_dev = args.selfplay_batch

    @functools.partial(jax.pmap, axis_name="d")
    def selfplay(params, rng_key) -> SelfplayOut:
        key, sub = jax.random.split(rng_key)
        states = jax.vmap(env.init)(jax.random.split(sub, batch_per_dev))

        def step_fn(carry, t):
            key, state = carry
            key, k_search = jax.random.split(key)
            obs = state.observation
            valid = ~(state.terminated | state.truncated)
            cur = state.current_player

            # Pass guard (selfplay exploration only): before ply `pass_guard_ply`
            # the root may not pass unless pass is the only legal move.  This
            # blocks the degenerate "concede when slightly behind" equilibrium.
            invalid = ~state.legal_action_mask
            forbid_pass = (t < args.pass_guard_ply) & state.legal_action_mask[:, :-1].any(-1)
            invalid = invalid.at[:, -1].set(invalid[:, -1] | forbid_pass)

            logits, value = forward(params, obs)
            root_logits = jnp.where(~invalid, logits, jnp.finfo(logits.dtype).min)
            root = mctx.RootFnOutput(
                prior_logits=root_logits, value=value, embedding=state
            )
            po = mctx.gumbel_muzero_policy(
                params=params,
                rng_key=k_search,
                root=root,
                recurrent_fn=recurrent_fn,
                num_simulations=args.sims,
                invalid_actions=invalid,
                qtransform=mctx.qtransform_completed_by_mix_value,
                max_num_considered_actions=args.max_considered,
                gumbel_scale=1.0,
            )
            next_state = jax.vmap(env.step)(state, po.action)
            reward = next_state.rewards[jnp.arange(batch_per_dev), cur]
            done_after = next_state.terminated | next_state.truncated
            out = SelfplayOut(
                obs=obs,
                policy_tgt=po.action_weights,
                reward=reward,
                valid=valid,
                done_after=done_after,
                is_pass=po.action == env.num_actions - 1,
            )
            return (key, next_state), out

        _, data = jax.lax.scan(
            step_fn, (key, states), jnp.arange(args.max_steps)
        )
        return data

    @functools.partial(jax.pmap, axis_name="d")
    def compute_targets(data: SelfplayOut) -> Sample:
        # Value target: outcome from the mover's perspective, sign flipping at
        # each ply.  v_t = r_t + gamma_t * v_{t+1}, gamma_t = 0 once the game
        # is over (post-terminal steps have reward 0, so they stay 0).
        def body(v_next, xs):
            reward, done_after = xs
            v = reward + jnp.where(done_after, 0.0, -1.0) * v_next
            return v, v

        _, value_tgt = jax.lax.scan(
            body,
            jnp.zeros_like(data.reward[0]),
            (data.reward, data.done_after),
            reverse=True,
        )
        # Only train on moves made in live games that actually finished within
        # the scan window (unfinished tails would carry a bogus 0 bootstrap).
        finished = jnp.flip(jnp.cumsum(jnp.flip(data.done_after, 0), 0), 0) >= 1
        mask = data.valid & finished
        return Sample(data.obs, data.policy_tgt, value_tgt, mask)

    def loss_fn(params, batch: Sample):
        logits, value = forward(params, batch.obs)
        n = jnp.maximum(batch.mask.sum(), 1.0)
        logp = jax.nn.log_softmax(logits, axis=-1)
        policy_loss = -(batch.policy_tgt * logp).sum(-1)
        policy_loss = (policy_loss * batch.mask).sum() / n
        value_loss = ((value - batch.value_tgt) ** 2 * batch.mask).sum() / n
        return policy_loss + value_loss, (policy_loss, value_loss)

    def make_optimizer():
        updates_per_iter = max(
            (args.max_steps * args.selfplay_batch) // args.train_batch, 1
        )
        sched = optax.warmup_cosine_decay_schedule(
            init_value=args.lr / 10,
            peak_value=args.lr,
            warmup_steps=args.warmup_iters * updates_per_iter,
            decay_steps=args.decay_iters * updates_per_iter,
            end_value=args.lr / 10,
        )
        return optax.adamw(sched, weight_decay=args.weight_decay)

    optimizer = make_optimizer()
    updates_per_iter = max((args.max_steps * args.selfplay_batch) // args.train_batch, 1)

    @functools.partial(jax.pmap, axis_name="d")
    def train_epoch(params, opt_state, samples: Sample, key):
        """One pass over this device's selfplay shard, fully on-device.

        (A host-side shuffle of pmap output silently gathers everything onto
        device 0; keeping the permutation + minibatch loop inside pmap avoids
        that round-trip entirely.)
        """
        n_frames = args.max_steps * args.selfplay_batch
        perm = jax.random.permutation(key, n_frames)

        def to_batches(x):
            x = x.reshape((n_frames,) + x.shape[2:])[perm]
            return x[: updates_per_iter * args.train_batch].reshape(
                (updates_per_iter, args.train_batch) + x.shape[1:]
            )

        batches = jax.tree_util.tree_map(to_batches, samples)

        def step(carry, batch):
            params, opt_state = carry
            (loss, (pl, vl)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                params, batch
            )
            grads = jax.lax.pmean(grads, axis_name="d")
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return (params, opt_state), (loss, pl, vl)

        (params, opt_state), (losses, pls, vls) = jax.lax.scan(
            step, (params, opt_state), batches
        )
        return params, opt_state, losses.mean(), pls.mean(), vls.mean()

    # ------------------------------------------------------------------
    # Evaluation: greedy raw-policy games, my params vs opponent params.
    # Half the games are played as the first mover, half as the second.
    # ------------------------------------------------------------------
    @functools.partial(jax.pmap, axis_name="d")
    def eval_games(params, opp_params, rng_key):
        b = args.eval_batch
        key, sub = jax.random.split(rng_key)
        states = jax.vmap(env.init)(jax.random.split(sub, b))
        my_player = jnp.arange(b) % 2  # player id our net controls

        def cond(carry):
            states, _, _, _ = carry
            return ~(states.terminated | states.truncated).all()

        def body(carry):
            states, acc, key, ply = carry
            key, k_s = jax.random.split(key)
            my_logits, _ = forward(params, states.observation)
            opp_logits, _ = forward(opp_params, states.observation)
            logits = jnp.where(
                (states.current_player == my_player)[:, None], my_logits, opp_logits
            )
            logits = jnp.where(
                states.legal_action_mask, logits, jnp.finfo(logits.dtype).min
            )
            # First plies sampled for opening diversity (greedy self-play from the
            # fixed empty board would only ever produce two distinct games).
            action = jnp.where(
                ply < 8, jax.random.categorical(k_s, logits, axis=-1),
                jnp.argmax(logits, axis=-1),
            )
            states = jax.vmap(env.step)(states, action)
            # pgx emits the game result only on the terminating step; accumulate.
            acc = acc + states.rewards[jnp.arange(b), my_player]
            return states, acc, key, ply + 1

        acc0 = jnp.zeros((b,), dtype=jnp.float32)
        _, my_reward, _, _ = jax.lax.while_loop(cond, body, (states, acc0, key, 0))
        return my_reward  # (b,) in {-1, 0, 1}

    return selfplay, compute_targets, train_epoch, eval_games, make_optimizer()


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def save_ckpt(path, params, opt_state, it, args):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(
            {
                "params": jax.device_get(jax.tree_util.tree_map(lambda x: x[0], params)),
                "opt_state": jax.device_get(
                    jax.tree_util.tree_map(lambda x: x[0], opt_state)
                ),
                "iteration": it,
                "config": vars(args),
            },
            f,
        )
    os.replace(tmp, path)


def main():
    args = parse_args()
    os.makedirs(args.run_dir, exist_ok=True)
    with open(os.path.join(args.run_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    devices = jax.local_devices()
    num_devices = len(devices)
    print(f"devices: {devices}")

    env = pgx.make(args.env_id)
    net = AZNet(num_actions=env.num_actions, channels=args.channels, num_blocks=args.blocks)

    rng = jax.random.PRNGKey(args.seed)
    rng, k_init = jax.random.split(rng)
    dummy_obs = jax.vmap(env.init)(jax.random.split(k_init, 2)).observation
    variables = net.init(k_init, dummy_obs)
    params = variables["params"]

    selfplay, compute_targets, train_epoch, eval_games, optimizer = make_fns(
        env, net, args, num_devices
    )
    opt_state = optimizer.init(params)

    start_iter = 0
    if args.resume:
        with open(args.resume, "rb") as f:
            ck = pickle.load(f)
        params = ck["params"]
        opt_state = ck["opt_state"]
        start_iter = ck["iteration"]
        print(f"resumed from {args.resume} at iteration {start_iter}")

    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"model parameters: {n_params/1e6:.2f}M")

    params = jax.device_put_replicated(params, devices)
    opt_state = jax.device_put_replicated(opt_state, devices)

    # Frozen snapshot of an earlier self for Elo-style progress tracking.
    anchor_params = jax.tree_util.tree_map(lambda x: x, params)
    anchor_iter = start_iter

    metrics_path = os.path.join(args.run_dir, "metrics.jsonl")
    frames_per_iter = num_devices * args.max_steps * args.selfplay_batch

    for it in range(start_iter, args.iters):
        t0 = time.time()
        rng, k_sp = jax.random.split(rng)
        keys = jax.random.split(k_sp, num_devices)

        data = selfplay(params, keys)
        samples = compute_targets(data)

        # One epoch over this iteration's data, shuffled + trained per device.
        rng, k_perm = jax.random.split(rng)
        params, opt_state, loss, pl, vl = train_epoch(
            params, opt_state, samples, jax.random.split(k_perm, num_devices)
        )
        loss = float(loss.mean())
        pl = float(pl.mean())
        vl = float(vl.mean())
        mask_frac = float(samples.mask.mean())
        finished_games = float(data.done_after.any(axis=1).mean())
        game_len = float(
            jnp.where(data.valid.sum(axis=1) > 0, data.valid.sum(axis=1), 0).mean()
        )
        pass_frac = float(
            (data.is_pass & data.valid).sum() / jnp.maximum(data.valid.sum(), 1)
        )

        rec = {
            "iter": it + 1,
            "time": round(time.time() - t0, 2),
            "loss": round(loss, 4),
            "policy_loss": round(pl, 4),
            "value_loss": round(vl, 4),
            "mask_frac": round(mask_frac, 4),
            "finished_frac": round(finished_games, 4),
            "avg_game_len": round(game_len, 1),
            "pass_frac": round(pass_frac, 4),
            "frames": frames_per_iter,
        }

        # Periodic eval vs frozen anchor.
        if (it + 1) % args.eval_every == 0:
            rng, k_ev = jax.random.split(rng)
            r = eval_games(params, anchor_params, jax.random.split(k_ev, num_devices))
            win = float((r == 1).mean())
            lose = float((r == -1).mean())
            rec["anchor_iter"] = anchor_iter
            rec["win_vs_anchor"] = round(win, 4)
            rec["lose_vs_anchor"] = round(lose, 4)
            # Move the anchor up once we clearly dominate it.
            if win > 0.85:
                anchor_params = jax.tree_util.tree_map(lambda x: x, params)
                anchor_iter = it + 1
                rec["anchor_updated"] = True

        if (it + 1) % args.save_every == 0 or it + 1 == args.iters:
            save_ckpt(
                os.path.join(args.run_dir, f"ckpt_{it+1:06d}.pkl"),
                params, opt_state, it + 1, args,
            )
        save_ckpt(os.path.join(args.run_dir, "latest.pkl"), params, opt_state, it + 1, args)

        with open(metrics_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
