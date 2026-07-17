"""HTTP engine server for the Flutter app.

Usage:
    python -m gozero.server --ckpt runs/v1/latest.pkl --port 8765
    python -m gozero.server --ckpt /models/latest.pkl --host 0.0.0.0 --port 8765

The iOS-simulator app talks to http://127.0.0.1:8765 (the simulator shares
the host network).  Threaded: one thread per connection so a slow/stalled
client can't wedge everyone else, but all engine access is serialized with
a lock since JAX search/state mutation isn't safe to run concurrently.

Endpoints (JSON in/out):
    GET  /health                          -> model info
    POST /new    {level, human_color, komi?, handicap?} -> fresh game, AI opens if to move
    POST /move   {game_id, action}        -> human move + AI reply (action 81 = pass)
    POST /undo   {game_id}                -> revert one full round (human+AI plies)
    POST /resign {game_id}                -> human resigns
"""
from __future__ import annotations

import argparse
import atexit
import json
import os
import signal
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import jax
import jax.numpy as jnp
import numpy as np

from pgx.go import Go

from gozero.mcts import batch_of_one, load_ckpt, make_search_fn

LEVELS = {"easy": 0, "normal": 32, "strong": 128}  # MCTS simulations
DEFAULT_KOMI = 7.5  # the pgx training komi; other values are legal but the
                    # value head stays 7.5-calibrated (search still scores
                    # terminal nodes with the game's own komi)
HANDICAPS = (0, 2, 3, 4)


class Game:
    def __init__(self, engine, level: str, human_color: str,
                 komi: float = DEFAULT_KOMI, handicap: int = 0):
        self.level = level
        self.human_color = human_color  # "black" | "white"
        self.komi = komi
        self.handicap = handicap
        # 讓子前綴（黑落星位/白虛手交替）佔掉的手數，undo 不可退進這段
        self.setup_plies = max(0, 2 * handicap - 1)
        self.history: list[int] = []  # actions from the initial position
        key = jax.random.PRNGKey(int(time.time() * 1000) % (2**31))
        self.key, self.init_key = jax.random.split(key)
        self.state = engine.env_fns(komi)[0](self.init_key)
        self.black_player = int(self.state.current_player)  # black moves first
        self.resigned_by: str | None = None
        # 黑勝率軌跡：winrates[i] = 第 i 手後（i=0 為空盤）的模型評估
        self.winrates: list[float] = [engine.state_black_winrate(self.state, 0)]


class Engine:
    MAX_GAMES = 32  # oldest games beyond this are dropped

    def __init__(self, ckpt_path: str):
        self.env, self.net, self.params, ck = load_ckpt(ckpt_path)
        self.size = int(self.env.observation_shape[0])
        self.iteration = ck.get("iteration")
        self.config = ck["config"]
        self.forward = jax.jit(lambda obs: self.net.apply({"params": self.params}, obs))
        # 貼目烙在 env 的 JIT 常數裡：每個 komi 要自己的 env/init/step 與 search fn，
        # 非預設貼目第一手會多等一次編譯，之後走快取
        self._envs: dict[float, object] = {DEFAULT_KOMI: self.env}
        self._env_fns: dict[float, tuple] = {}
        self._searches: dict[tuple[float, str], object] = {}
        # 讓子星位：右上、左下、右下、左上（9 路 hoshi 在三線）
        h = 2 if self.size < 13 else 3
        t = self.size - 1 - h
        self.handicap_actions = [r * self.size + c
                                 for r, c in ((h, t), (t, h), (t, t), (h, h))]
        self.games: dict[str, Game] = {}
        # serializes all engine access across handler threads (JAX calls + games dict)
        self.lock = threading.Lock()
        # trigger compilation up-front so the first app move isn't slow
        g = Game(self, "easy", "black")
        self.env_fns(DEFAULT_KOMI)[1](g.state, jnp.int32(0))
        for name in LEVELS:
            self.search_fn(DEFAULT_KOMI, name)(
                self.params, jax.random.PRNGKey(0), batch_of_one(g.state))
        self.forward(batch_of_one(g.state).observation)

    def _env(self, komi: float):
        if komi not in self._envs:
            self._envs[komi] = Go(size=self.size, komi=komi)
        return self._envs[komi]

    def env_fns(self, komi: float) -> tuple:
        """(jitted init, jitted step) for the given komi.
        eager env.step costs ~200ms/move on CPU; jitted it's ~1ms"""
        if komi not in self._env_fns:
            env = self._env(komi)
            self._env_fns[komi] = (jax.jit(env.init), jax.jit(env.step))
        return self._env_fns[komi]

    def search_fn(self, komi: float, level: str):
        if (komi, level) not in self._searches:
            self._searches[(komi, level)] = make_search_fn(
                self._env(komi), self.net, num_simulations=LEVELS[level])
        return self._searches[(komi, level)]

    # -- board / evaluation helpers -----------------------------------------
    def board(self, game: Game) -> list[int]:
        """0 empty, 1 black, 2 white, from the observation's stone planes."""
        obs = np.asarray(game.state.observation)  # (H, W, C); plane 0 = to-move
        black_to_move = len(game.history) % 2 == 0
        mine, opp = obs[:, :, 0] > 0, obs[:, :, 1] > 0
        black, white = (mine, opp) if black_to_move else (opp, mine)
        return (black.astype(int) + 2 * white.astype(int)).flatten().tolist()

    def state_black_winrate(self, state, ply: int) -> float:
        _, v = self.forward(batch_of_one(state).observation)
        v = float(v[0])  # current player's expected outcome in [-1, 1]
        p = (v + 1.0) / 2.0
        return p if ply % 2 == 0 else 1.0 - p

    def tromp_taylor(self, board: list[int], komi: float) -> float:
        """Black score margin (positive = black leads), area scoring + komi."""
        n = self.size
        b = np.array(board).reshape(n, n)
        counts = {1: int((b == 1).sum()), 2: int((b == 2).sum())}
        seen = np.zeros_like(b, dtype=bool)
        for r in range(n):
            for c in range(n):
                if b[r, c] != 0 or seen[r, c]:
                    continue
                stack, region, borders = [(r, c)], [], set()
                seen[r, c] = True
                while stack:
                    y, x = stack.pop()
                    region.append((y, x))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        yy, xx = y + dy, x + dx
                        if 0 <= yy < n and 0 <= xx < n:
                            if b[yy, xx] == 0 and not seen[yy, xx]:
                                seen[yy, xx] = True
                                stack.append((yy, xx))
                            elif b[yy, xx] != 0:
                                borders.add(int(b[yy, xx]))
                if borders == {1}:
                    counts[1] += len(region)
                elif borders == {2}:
                    counts[2] += len(region)
        return counts[1] - counts[2] - komi

    def snapshot(self, game: Game, game_id: str, ai_move: int | None = None) -> dict:
        board = self.board(game)
        over = bool(game.state.terminated | game.state.truncated) or game.resigned_by
        result = None
        if game.resigned_by:
            winner = "white" if game.resigned_by == "black" else "black"
            result = {"winner": winner, "reason": "resign", "margin": None}
        elif over:
            margin = self.tromp_taylor(board, game.komi)
            r_black = float(game.state.rewards[game.black_player])
            if r_black != 0 and (r_black > 0) != (margin > 0):
                # pgx 的判定（如全同型犯規=立即判負）優先於盤面點目
                result = {"winner": "black" if r_black > 0 else "white",
                          "reason": "rule", "margin": None}
            else:
                result = {
                    "winner": "black" if margin > 0 else "white" if margin < 0 else "draw",
                    "reason": "score",
                    "margin": abs(margin),
                }
        return {
            "game_id": game_id,
            "board": board,
            "size": self.size,
            "to_move": "black" if len(game.history) % 2 == 0 else "white",
            "human_color": game.human_color,
            "moves": len(game.history),
            "history": game.history,
            "last_move": game.history[-1] if game.history else None,
            "ai_move": ai_move,
            "legal": np.asarray(game.state.legal_action_mask).astype(int).tolist(),
            "black_winrate": round(game.winrates[-1], 4),
            "winrates": [round(w, 4) for w in game.winrates],
            "captures": self.capture_counts(game, board),
            "game_over": bool(over),
            "result": result,
            "komi": game.komi,
            "handicap": game.handicap,
            "setup_plies": game.setup_plies,
        }

    def capture_counts(self, game: Game, board: list[int]) -> dict:
        pass_action = self.size * self.size
        black_played = sum(1 for i, a in enumerate(game.history) if i % 2 == 0 and a != pass_action)
        white_played = sum(1 for i, a in enumerate(game.history) if i % 2 == 1 and a != pass_action)
        return {  # stones of each colour removed from the board
            "black": black_played - board.count(1),
            "white": white_played - board.count(2),
        }

    # -- game flow ------------------------------------------------------------
    def play(self, game: Game, action: int):
        if not bool(game.state.legal_action_mask[action]):
            raise ValueError("illegal move")
        game.state = self.env_fns(game.komi)[1](game.state, jnp.int32(action))
        game.history.append(int(action))
        game.winrates.append(self.state_black_winrate(game.state, len(game.history)))

    def apply_handicap(self, game: Game):
        """讓子前綴：黑落星位、白虛手交替。虛手不連續，不會觸發雙虛手終局，
        結束時輪到白方行棋（讓子棋慣例）。"""
        pass_action = self.size * self.size
        for i, a in enumerate(self.handicap_actions[:game.handicap]):
            if i:
                self.play(game, pass_action)
            self.play(game, a)

    def ai_move(self, game: Game) -> int:
        game.key, k = jax.random.split(game.key)
        search = self.search_fn(game.komi, game.level)
        actions, _ = search(self.params, k, batch_of_one(game.state))
        action = int(actions[0])
        self.play(game, action)
        return action

    def replay(self, game: Game, history: list[int]):
        game.state = self.env_fns(game.komi)[0](game.init_key)
        game.history = []
        game.winrates = [self.state_black_winrate(game.state, 0)]
        for a in history:
            self.play(game, a)

    # -- persistence: survive server restarts ---------------------------------
    def save_games(self, path: str):
        with self.lock:
            data = {
                gid: {"level": g.level, "human_color": g.human_color,
                      "history": g.history, "resigned_by": g.resigned_by,
                      "komi": g.komi, "handicap": g.handicap}
                for gid, g in self.games.items()
            }
        with open(path, "w") as f:
            json.dump(data, f)

    def load_games(self, path: str):
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return
        for gid, d in data.items():
            game = Game(self, d["level"], d["human_color"],
                        d.get("komi", DEFAULT_KOMI), d.get("handicap", 0))
            try:
                self.replay(game, d["history"])  # 存檔已含讓子前綴
            except ValueError:
                continue  # stale/corrupt entry; drop it
            game.resigned_by = d["resigned_by"]
            self.games[gid] = game
        if self.games:
            print(f"restored {len(self.games)} game(s)", flush=True)


class Handler(BaseHTTPRequestHandler):
    engine: Engine  # set at startup
    timeout = 30  # cap how long one thread waits on a silent connection

    def log_message(self, fmt, *args):  # quiet
        pass

    def _send(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        e = self.engine
        if self.path.startswith("/state"):
            # read-only resync endpoint: /state?game_id=xxx
            from urllib.parse import parse_qs, urlparse
            gid = parse_qs(urlparse(self.path).query).get("game_id", [""])[0]
            with e.lock:
                game = e.games.get(gid)
                if game is None:
                    return self._send({"error": "unknown game"}, 404)
                return self._send(e.snapshot(game, gid))
        if self.path != "/health":
            return self._send({"error": "not found"}, 404)
        cfg = e.config
        self._send({
            "ok": True,
            "model": f"gozero {cfg['env_id']} {cfg['channels']}ch x {cfg['blocks']}blk",
            "iteration": e.iteration,
            "board_size": e.size,
        })

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        e = self.engine
        try:
            with e.lock:
                if self.path == "/new":
                    level = req.get("level", "normal")
                    human = req.get("human_color", "black")
                    komi = float(req.get("komi", DEFAULT_KOMI))
                    handicap = int(req.get("handicap", 0))
                    if level not in LEVELS or human not in ("black", "white"):
                        return self._send({"error": "bad level/color"}, 400)
                    # 半整數避免 JIT 快取被連續值撐爆；範圍蓋住 9 路全盤
                    if not (komi * 2).is_integer() or not -81 <= komi <= 81:
                        return self._send({"error": "bad komi"}, 400)
                    if handicap not in HANDICAPS:
                        return self._send({"error": "bad handicap"}, 400)
                    game_id = uuid.uuid4().hex[:12]
                    game = Game(e, level, human, komi, handicap)
                    e.games[game_id] = game
                    while len(e.games) > Engine.MAX_GAMES:
                        e.games.pop(next(iter(e.games)))
                    e.apply_handicap(game)
                    to_move = "black" if len(game.history) % 2 == 0 else "white"
                    ai = e.ai_move(game) if to_move != human else None
                    return self._send(e.snapshot(game, game_id, ai_move=ai))

                game = e.games.get(req.get("game_id", ""))
                if game is None:
                    return self._send({"error": "unknown game"}, 404)
                gid = req["game_id"]

                if self.path == "/move":
                    if game.resigned_by or bool(game.state.terminated | game.state.truncated):
                        return self._send({"error": "game over"}, 400)
                    action = int(req["action"])
                    # JAX 索引會 clamp/wrap，越界值必須擋在這裡
                    if not 0 <= action <= e.size * e.size:
                        return self._send({"error": "action out of range"}, 400)
                    e.play(game, action)
                    ai = None
                    if not bool(game.state.terminated | game.state.truncated):
                        ai = e.ai_move(game)
                    return self._send(e.snapshot(game, gid, ai_move=ai))

                if self.path == "/undo":
                    # drop plies until it's the human's turn again (min one round);
                    # never into the handicap setup prefix
                    human_is_black = game.human_color == "black"
                    base = game.setup_plies
                    h = game.history[:]
                    if len(h) <= base:
                        return self._send({"error": "nothing to undo"}, 400)
                    h.pop()
                    while len(h) > base and (len(h) % 2 == 0) != human_is_black:
                        h.pop()
                    game.resigned_by = None
                    e.replay(game, h)
                    # 執白退到空盤時輪到 AI（黑）先行：補回開局手，否則棋局卡死
                    if (len(game.history) % 2 == 0) != human_is_black:
                        e.ai_move(game)
                    return self._send(e.snapshot(game, gid))

                if self.path == "/resign":
                    game.resigned_by = game.human_color
                    return self._send(e.snapshot(game, gid))

                return self._send({"error": "not found"}, 404)
        except ValueError as err:
            return self._send({"error": str(err)}, 400)


class Server(ThreadingHTTPServer):
    daemon_threads = True  # don't block process exit on stuck connections


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--host", default="127.0.0.1",
                   help="interface to bind (use 0.0.0.0 inside Docker)")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--state-file", default=None,
                   help="games survive restarts here (default: <ckpt dir>/app_games.json)")
    args = p.parse_args()
    state_file = args.state_file or os.path.join(os.path.dirname(args.ckpt), "app_games.json")
    print(f"loading {args.ckpt} ...", flush=True)
    Handler.engine = Engine(args.ckpt)
    Handler.engine.load_games(state_file)

    def save_and_exit(signum=None, frame=None):
        Handler.engine.save_games(state_file)
        raise SystemExit(0)

    atexit.register(lambda: Handler.engine.save_games(state_file))
    signal.signal(signal.SIGTERM, save_and_exit)
    signal.signal(signal.SIGINT, save_and_exit)
    print(f"engine ready (iteration {Handler.engine.iteration}), "
          f"serving on http://{args.host}:{args.port}", flush=True)
    Server((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
