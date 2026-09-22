"""JAX-free stand-in for gozero.server, for UI work and handler tests.

The real server needs JAX + a checkpoint to even import.  This module stubs
those out and swaps in a tiny random-move Go engine, so the *HTTP layer*
(job queue, stamina, undo limit, snapshots, persistence) runs unchanged and
the Flutter app can be exercised on a simulator without a model:

    python3 scripts/fake_engine_server.py --port 8765 --stamina-max 3

The AI plays random legal moves; nothing here says anything about strength.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import threading
import time
import types

# --- stub the heavy imports before gozero.server is loaded --------------------
def _install_stubs() -> None:
    """Only when the real packages are absent (e.g. system python3 without JAX).

    With JAX installed (CI, the deploy host) the real modules are used and
    nothing is patched, so this file is safe to import next to real tests.
    """
    import importlib.util

    if importlib.util.find_spec("jax") is not None:
        return
    if "jax" in sys.modules and getattr(sys.modules["jax"], "_gozero_fake", False):
        return
    jax = types.ModuleType("jax")
    jax._gozero_fake = True
    jax.jit = lambda f: f

    class _Random:
        @staticmethod
        def PRNGKey(seed):
            return int(seed)

        @staticmethod
        def split(key):
            return (int(key) * 2 + 1) & 0x7FFFFFFF, (int(key) * 3 + 7) & 0x7FFFFFFF

    jax.random = _Random()
    jnp = types.ModuleType("jax.numpy")
    jnp.int32 = int
    jax.numpy = jnp
    sys.modules["jax"] = jax
    sys.modules["jax.numpy"] = jnp

    pgx = types.ModuleType("pgx")
    pgx_go = types.ModuleType("pgx.go")
    pgx_go.Go = object
    pgx.go = pgx_go
    sys.modules["pgx"] = pgx
    sys.modules["pgx.go"] = pgx_go

    mcts = types.ModuleType("gozero.mcts")
    mcts.batch_of_one = lambda s: s
    mcts.load_ckpt = lambda path: (_ for _ in ()).throw(RuntimeError("fake engine"))
    mcts.make_search_fn = lambda *a, **k: None
    sys.modules["gozero.mcts"] = mcts


_install_stubs()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gozero.server as server  # noqa: E402  (after stubs on purpose)
from gozero.job_queue import InferenceJobQueue  # noqa: E402
from gozero.stamina import StaminaBank  # noqa: E402


# --- a very small Go: captures + suicide rule, no ko, two passes end the game --
class FakeState:
    def __init__(self, n: int):
        self.n = n
        self.stones = [0] * (n * n)  # 0 empty, 1 black, 2 white
        self.current_player = 0  # 0 = black to move
        self.passes = 0
        self.plies = 0
        self.terminated = False
        self.truncated = False
        self.rewards = [0.0, 0.0]
        self.legal_action_mask = self._legal()

    def copy(self) -> "FakeState":
        s = FakeState.__new__(FakeState)
        s.__dict__.update(self.__dict__)
        s.stones = list(self.stones)
        s.rewards = list(self.rewards)
        return s

    def _group(self, stones, i):
        colour = stones[i]
        n = self.n
        seen, stack, libs = {i}, [i], set()
        while stack:
            p = stack.pop()
            r, c = divmod(p, n)
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if not (0 <= rr < n and 0 <= cc < n):
                    continue
                q = rr * n + cc
                if stones[q] == 0:
                    libs.add(q)
                elif stones[q] == colour and q not in seen:
                    seen.add(q)
                    stack.append(q)
        return seen, libs

    def _place(self, stones, i, colour):
        stones[i] = colour
        n = self.n
        r, c = divmod(i, n)
        opp = 3 - colour
        captured = 0
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < n and 0 <= cc < n:
                q = rr * n + cc
                if stones[q] == opp:
                    grp, libs = self._group(stones, q)
                    if not libs:
                        for p in grp:
                            stones[p] = 0
                        captured += len(grp)
        _, libs = self._group(stones, i)
        return captured, bool(libs)

    def _legal(self):
        n = self.n
        colour = 1 if self.current_player == 0 else 2
        mask = [0] * (n * n + 1)
        if self.terminated or self.truncated:
            return mask
        for i in range(n * n):
            if self.stones[i]:
                continue
            trial = list(self.stones)
            _, alive = self._place(trial, i, colour)
            mask[i] = 1 if alive else 0
        mask[n * n] = 1
        return mask

    def step(self, action: int) -> "FakeState":
        s = self.copy()
        n = s.n
        colour = 1 if s.current_player == 0 else 2
        if action == n * n:
            s.passes += 1
        else:
            s.passes = 0
            s._place(s.stones, action, colour)
        s.plies += 1
        s.current_player = 1 - s.current_player
        if s.passes >= 2:
            s.terminated = True
        elif s.plies >= 2 * n * n:
            s.truncated = True
        s.legal_action_mask = s._legal()
        return s


class FakeEngine(server.Engine):
    """Same registry / snapshot / persistence code as the real Engine."""

    def __init__(self, size: int, jobs: InferenceJobQueue, seed: int = 0):
        self.size = size
        self.iteration = 0
        self.config = {"channels": 0, "blocks": 0}
        self.rng = random.Random(seed)
        h = 2 if size < 13 else 3
        t = size - 1 - h
        self.handicap_actions = [r * size + c for r, c in ((h, t), (t, h), (t, t), (h, h))]
        self.games = {}
        self.games_lock = threading.Lock()
        self.cache_lock = threading.RLock()
        self.jobs = jobs

    def model_info(self) -> dict:
        return {"model": f"fake go_{self.size}x{self.size}", "iteration": 0,
                "board_size": self.size}

    def env_fns(self, komi: float):
        n = self.size
        return (lambda key: FakeState(n), lambda state, action: state.step(int(action)))

    def board(self, game) -> list[int]:
        return list(game.state.stones)

    def state_black_winrate(self, state, ply: int) -> float:
        return 0.5 + 0.02 * ((ply % 5) - 2)

    def ai_move(self, game) -> int:
        n = self.size
        legal = [i for i in range(n * n) if game.state.legal_action_mask[i]]
        action = self.rng.choice(legal) if legal else n * n
        time.sleep(0.05)  # 讓佇列/忙碌指示有東西可看
        self.play(game, action)
        return action


class FakeEnginePool(server.EnginePool):
    def __init__(self, sizes=(19, 9), stamina: StaminaBank | None = None):
        self.jobs = InferenceJobQueue(
            workers=server.Engine.SEARCH_SLOTS,
            max_pending=server.Engine.MAX_QUEUE,
            result_ttl=server.Engine.JOB_TTL,
        )
        self.engines = {n: FakeEngine(n, self.jobs) for n in sizes}
        self.ckpt_paths = {n: f"<fake {n}x{n}>" for n in sizes}
        self.default_size = max(self.engines)
        # 注意 StaminaBank 定義了 __len__，空帳本是 falsy，不能用 `stamina or ...`
        self.stamina = stamina if stamina is not None else StaminaBank()
        self.ip_stamina = StaminaBank(
            max_points=server.IP_STAMINA_MAX,
            regen_seconds=server.IP_STAMINA_REGEN_SECONDS,
        )


def serve(host: str, port: int, pool: FakeEnginePool):
    server.Handler.engines = pool
    return server.Server((host, port), server.Handler)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--stamina-max", type=int, default=server.StaminaBank().max_points)
    p.add_argument("--stamina-regen-seconds", type=int,
                   default=server.StaminaBank().regen_seconds)
    args = p.parse_args()
    pool = FakeEnginePool(stamina=StaminaBank(
        max_points=args.stamina_max, regen_seconds=args.stamina_regen_seconds))
    httpd = serve(args.host, args.port, pool)
    print(f"fake engine (random moves) on http://{args.host}:{args.port}  "
          f"stamina {args.stamina_max} / +1 per {args.stamina_regen_seconds}s  "
          f"undo limit {server.MAX_UNDOS}", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
