"""GTP engine speaking for a trained gozero checkpoint.

Usage:
    python -m gozero.gtp --ckpt runs/v1/latest.pkl --sims 256

Implements enough of the GTP protocol for GUI play (Sabaki, GoGui), engine
matches (gogui-twogtp), and the future app backend.  Board size is fixed to
the checkpoint's environment.
"""
from __future__ import annotations

import argparse
import sys

import jax
import jax.numpy as jnp

from gozero.coords import action_to_gtp, gtp_to_action
from gozero.mcts import batch_of_one, load_ckpt, make_search_fn


class Engine:
    def __init__(self, ckpt_path: str, sims: int, seed: int = 0):
        self.env, self.net, self.params, ck = load_ckpt(ckpt_path)
        self.size = int(self.env.observation_shape[0])
        self.search = make_search_fn(self.env, self.net, num_simulations=sims)
        self.key = jax.random.PRNGKey(seed)
        self.reset()

    def reset(self):
        self.key, k = jax.random.split(self.key)
        self.state = self.env.init(k)
        self.ply = 0

    def play(self, action: int, color: str | None = None):
        if color is not None:
            # pgx enforces strict alternation; black moves first from init
            expected = "b" if self.ply % 2 == 0 else "w"
            if color.lower()[0] != expected:
                raise ValueError(f"expected {expected} to move (no setup/handicap support)")
        if not bool(self.state.legal_action_mask[action]):
            raise ValueError("illegal move")
        self.state = self.env.step(self.state, jnp.int32(action))
        self.ply += 1

    def genmove(self) -> int:
        self.key, k = jax.random.split(self.key)
        actions, values = self.search(self.params, k, batch_of_one(self.state))
        action = int(actions[0])
        self.play(action)
        return action


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--sims", type=int, default=256)
    args = p.parse_args()

    eng = Engine(args.ckpt, args.sims)
    known = [
        "protocol_version", "name", "version", "known_command", "list_commands",
        "boardsize", "clear_board", "komi", "play", "genmove", "quit",
    ]

    cmd_id = ""

    def reply(msg=""):
        sys.stdout.write(f"={cmd_id} {msg}\n\n")
        sys.stdout.flush()

    def fail(msg):
        sys.stdout.write(f"?{cmd_id} {msg}\n\n")
        sys.stdout.flush()

    for line in sys.stdin:
        line = line.split("#")[0].strip()
        if not line:
            continue
        parts = line.split()
        # optional numeric command id (GTP requires echoing it in the response)
        cmd_id = ""
        if parts[0].isdigit():
            cmd_id, parts = parts[0], parts[1:]
        cmd, argv = parts[0].lower(), parts[1:]

        try:
            if cmd == "protocol_version":
                reply("2")
            elif cmd == "name":
                reply("gozero")
            elif cmd == "version":
                reply("0.1")
            elif cmd == "known_command":
                reply("true" if argv and argv[0] in known else "false")
            elif cmd == "list_commands":
                reply("\n".join(known))
            elif cmd == "boardsize":
                if argv and int(argv[0]) == eng.size:
                    reply()
                else:
                    fail(f"unacceptable size (engine is {eng.size}x{eng.size})")
            elif cmd == "clear_board":
                eng.reset()
                reply()
            elif cmd == "komi":
                reply()  # komi fixed by the training environment (7.5)
            elif cmd == "play":
                eng.play(gtp_to_action(argv[1], eng.size), color=argv[0])
                reply()
            elif cmd == "genmove":
                action = eng.genmove()
                reply(action_to_gtp(action, eng.size))
            elif cmd == "quit":
                reply()
                break
            else:
                fail("unknown command")
        except Exception as e:  # noqa: BLE001 - GTP requires an error reply
            fail(str(e))


if __name__ == "__main__":
    main()
