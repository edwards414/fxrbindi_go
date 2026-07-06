"""Match runner: gozero checkpoint vs {checkpoint, pgx baseline, random, GTP engine}.

Batched (on-GPU) opponents:
    python -m gozero.evaluate --ckpt runs/v1/latest.pkl --vs-baseline --games 256
    python -m gozero.evaluate --ckpt A.pkl --vs-ckpt B.pkl --sims 32 --opp-sims 32
    python -m gozero.evaluate --ckpt runs/v1/latest.pkl --vs-random

External engine over GTP (sequential):
    python -m gozero.evaluate --ckpt runs/v1/latest.pkl \
        --vs-gtp "gnugo --mode gtp --boardsize 9 --komi 7.5 --chinese-rules --level 10" \
        --games 20 --sims 256
"""
from __future__ import annotations

import argparse
import subprocess

import jax
import jax.numpy as jnp
import numpy as np
import pgx

from gozero.coords import action_to_gtp, gtp_to_action
from gozero.mcts import batch_of_one, load_ckpt, make_search_fn


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--vs-ckpt", default=None)
    p.add_argument("--vs-baseline", action="store_true")
    p.add_argument("--vs-random", action="store_true")
    p.add_argument("--vs-gtp", default=None, help="opponent GTP command line")
    p.add_argument("--games", type=int, default=256)
    p.add_argument("--sims", type=int, default=32, help="our MCTS sims (0 = raw policy)")
    p.add_argument("--opp-sims", type=int, default=0)
    p.add_argument("--opening-temp-moves", type=int, default=8,
                   help="sample (T=1) for the first N plies for game diversity")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-plies", type=int, default=250, help="GTP-mode adjudication cap")
    p.add_argument("--komi", type=float, default=7.5,
                   help="sent to the GTP opponent; must match pgx env komi (7.5)")
    return p.parse_args()


def make_our_policy(env, net, params, sims):
    search = make_search_fn(env, net, num_simulations=sims)

    def policy(key, states):
        actions, _ = search(params, key, states)
        return actions

    def raw_logits(states):
        logits, _ = jax.jit(lambda o: net.apply({"params": params}, o))(states.observation)
        return jnp.where(states.legal_action_mask, logits, jnp.finfo(logits.dtype).min)

    return policy, raw_logits


def batched_match(env, our_policy, our_logits_fn, opp_logits_fn, games, opening_temp_moves,
                  seed, opp_stochastic=False):
    """Play `games` games; our side alternates player id. Returns (wins, losses, draws).

    opp_stochastic: sample the opponent from its logits every ply (required for a
    true random opponent — argmax over uniform logits is a deterministic
    first-legal-point bot, not random play).
    """
    key = jax.random.PRNGKey(seed)
    key, k_init = jax.random.split(key)
    init = jax.jit(jax.vmap(env.init))
    step = jax.jit(jax.vmap(env.step))
    states = init(jax.random.split(k_init, games))
    my_player = jnp.arange(games) % 2
    # pgx emits the game result only on the terminating step; accumulate.
    acc = jnp.zeros((games,), dtype=jnp.float32)

    ply = 0
    while not bool((states.terminated | states.truncated).all()):
        key, k_pol, k_our, k_opp = jax.random.split(key, 4)
        opp_logits = opp_logits_fn(states)
        if ply < opening_temp_moves:
            our_l = our_logits_fn(states)
            our_actions = jax.random.categorical(k_our, our_l, axis=-1)
            opp_actions = jax.random.categorical(k_opp, opp_logits, axis=-1)
        else:
            our_actions = our_policy(k_pol, states)
            opp_actions = (jax.random.categorical(k_opp, opp_logits, axis=-1)
                           if opp_stochastic else opp_logits.argmax(axis=-1))
        actions = jnp.where(states.current_player == my_player, our_actions, opp_actions)
        # frozen (terminated) games: any action is fine, pgx keeps them frozen,
        # but it must be "legal-shaped"; use argmax of legal mask there.
        fallback = states.legal_action_mask.argmax(axis=-1)
        actions = jnp.where(states.terminated | states.truncated, fallback, actions)
        states = step(states, actions)
        acc = acc + states.rewards[jnp.arange(games), my_player]
        ply += 1

    r = np.asarray(acc)
    return int((r > 0).sum()), int((r < 0).sum()), int((r == 0).sum())


# ---------------------------------------------------------------------------
# GTP opponent (sequential)
# ---------------------------------------------------------------------------
class GtpClient:
    def __init__(self, cmdline: str):
        self.proc = subprocess.Popen(
            cmdline.split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
        )

    def send(self, cmd: str) -> str:
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()
        out = []
        while True:
            line = self.proc.stdout.readline()
            if line == "":
                raise RuntimeError("GTP engine died")
            if line.strip() == "" and out:
                break
            if line.strip():
                out.append(line.strip())
        resp = " ".join(out)
        if resp.startswith("?"):
            raise RuntimeError(f"GTP error: {resp}")
        return resp.lstrip("= ").strip()

    def close(self):
        try:
            self.send("quit")
        except Exception:
            pass
        self.proc.terminate()


def gtp_match(env, net, params, args):
    size = int(env.observation_shape[0])
    search = make_search_fn(env, net, num_simulations=args.sims)
    key = jax.random.PRNGKey(args.seed)
    results = []
    for g in range(args.games):
        we_are_black = g % 2 == 0
        opp = GtpClient(args.vs_gtp)
        opp.send(f"boardsize {size}")
        opp.send("clear_board")
        opp.send(f"komi {args.komi}")  # gnugo defaults to komi 0 otherwise
        key, k = jax.random.split(key)
        state = env.init(k)
        # In pgx go, black moves first; current_player id of black = state.current_player at init.
        black_player = int(state.current_player)
        our_player = black_player if we_are_black else 1 - black_player
        our_color, opp_color = ("black", "white") if we_are_black else ("white", "black")
        plies = 0
        opp_resigned = False
        while not bool(state.terminated | state.truncated) and plies < args.max_plies:
            if int(state.current_player) == our_player:
                key, k = jax.random.split(key)
                actions, _ = search(params, k, batch_of_one(state))
                action = int(actions[0])
                state = env.step(state, jnp.int32(action))
                if bool(state.terminated | state.truncated):
                    break  # e.g. positional-superko loss; don't forward the move
                try:
                    opp.send(f"play {our_color} {action_to_gtp(action, size)}")
                except RuntimeError as e:
                    # opponent ruleset rejected a pgx-legal move (superko edge);
                    # adjudicate from the current pgx position
                    print(f"  opponent rejected our move: {e}; adjudicating", flush=True)
                    break
            else:
                vertex = opp.send(f"genmove {opp_color}")
                if vertex.lower() == "resign":
                    opp_resigned = True
                    break
                action = gtp_to_action(vertex, size)
                if not bool(state.legal_action_mask[action]):
                    # ruleset mismatch corner case: substitute a pass
                    action = size * size
                state = env.step(state, jnp.int32(action))
            plies += 1
        if opp_resigned:
            our_r = 1.0
        elif not bool(state.terminated | state.truncated):
            # adjudicate: both pass from here, Tromp-Taylor scores board as-is
            # (pgx emits the result only on the terminating step, so read after each)
            state = env.step(state, jnp.int32(size * size))
            our_r = float(state.rewards[our_player])
            if not bool(state.terminated | state.truncated):
                state = env.step(state, jnp.int32(size * size))
                our_r = float(state.rewards[our_player])
        else:
            our_r = float(state.rewards[our_player])
        results.append(our_r)
        print(f"game {g+1}/{args.games} as {our_color}: "
              f"{'win' if our_r > 0 else 'loss' if our_r < 0 else 'draw'} ({plies} plies)",
              flush=True)
        opp.close()
    r = np.array(results)
    return int((r > 0).sum()), int((r < 0).sum()), int((r == 0).sum())


def main():
    args = parse_args()
    env, net, params, _ = load_ckpt(args.ckpt)

    if args.vs_gtp:
        w, l, d = gtp_match(env, net, params, args)
    else:
        our_policy, our_logits_fn = make_our_policy(env, net, params, args.sims)
        if args.vs_ckpt:
            env2, net2, params2, _ = load_ckpt(args.vs_ckpt)
            opp_policy, opp_logits_fn_ = make_our_policy(env2, net2, params2, args.opp_sims)
            if args.opp_sims > 0:
                w, l, d = searched_match(env, our_policy, our_logits_fn,
                                         opp_policy, opp_logits_fn_, args)
                n = w + l + d
                print(f"\nresult: {w}W {l}L {d}D / {n}  ({100*w/max(n,1):.1f}% wins)")
                return
            opp_logits_fn = opp_logits_fn_
        elif args.vs_baseline:
            baseline = pgx.make_baseline_model(env.id + "_v0")

            def opp_logits_fn(states):
                logits, _ = baseline(states.observation)
                return jnp.where(states.legal_action_mask, logits,
                                 jnp.finfo(logits.dtype).min)
        elif args.vs_random:
            def opp_logits_fn(states):
                return jnp.where(states.legal_action_mask, 0.0, -1e9)
        else:
            raise SystemExit("pick an opponent: --vs-ckpt/--vs-baseline/--vs-random/--vs-gtp")
        w, l, d = batched_match(env, our_policy, our_logits_fn, opp_logits_fn,
                                args.games, args.opening_temp_moves, args.seed,
                                opp_stochastic=args.vs_random)
    n = w + l + d
    print(f"\nresult: {w}W {l}L {d}D / {n}  ({100*w/max(n,1):.1f}% wins)")


def searched_match(env, our_policy, our_logits_fn, opp_policy, opp_logits_fn, args):
    """Both sides use MCTS; first few plies sampled from raw policies for variety."""
    games = args.games
    key = jax.random.PRNGKey(args.seed)
    key, k_init = jax.random.split(key)
    init = jax.jit(jax.vmap(env.init))
    step = jax.jit(jax.vmap(env.step))
    states = init(jax.random.split(k_init, games))
    my_player = jnp.arange(games) % 2
    acc = jnp.zeros((games,), dtype=jnp.float32)
    ply = 0
    while not bool((states.terminated | states.truncated).all()):
        key, k1, k2, k3, k4 = jax.random.split(key, 5)
        if ply < args.opening_temp_moves:
            our_actions = jax.random.categorical(k3, our_logits_fn(states), axis=-1)
            opp_actions = jax.random.categorical(k4, opp_logits_fn(states), axis=-1)
        else:
            our_actions = our_policy(k1, states)
            opp_actions = opp_policy(k2, states)
        actions = jnp.where(states.current_player == my_player, our_actions, opp_actions)
        fallback = states.legal_action_mask.argmax(axis=-1)
        actions = jnp.where(states.terminated | states.truncated, fallback, actions)
        states = step(states, actions)
        acc = acc + states.rewards[jnp.arange(games), my_player]
        ply += 1
    r = np.asarray(acc)
    return int((r > 0).sum()), int((r < 0).sum()), int((r == 0).sum())


if __name__ == "__main__":
    main()
