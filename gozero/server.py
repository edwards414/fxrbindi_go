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
import traceback
import uuid
from collections import OrderedDict
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
MAX_BODY = 64 * 1024  # 請求主體上限；正常請求不到 1 KB


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
        # 這一局自己的鎖：同一局的請求仍互斥（連點兩下不會壞盤面），
        # 但不同局之間可以真正並行推理。
        self.lock = threading.Lock()
        self.touched = time.time()  # 最後活動時間，淘汰用


class Engine:
    # 上限只是記憶體保險絲（一局約數十 KB），實際靠 TTL 清乾淨。
    # 舊版是 32 局並依「建立順序」淘汰，第 33 個人開局就會刪掉第 1 個人正在下的棋；
    # 現在改成先清逾時的、真的還超量才淘汰「最久沒動作的那局」。
    MAX_GAMES = 2000
    GAME_TTL = 6 * 3600  # 6 小時沒有任何動作才視為棄局
    MAX_KOMI_CACHE = 12  # 每個貼目都要各自 JIT 編譯，快取需有上限

    def __init__(self, ckpt_path: str):
        self.env, self.net, self.params, ck = load_ckpt(ckpt_path)
        self.size = int(self.env.observation_shape[0])
        self.iteration = ck.get("iteration")
        self.config = ck["config"]
        self.forward = jax.jit(lambda obs: self.net.apply({"params": self.params}, obs))
        # 貼目烙在 env 的 JIT 常數裡：每個 komi 要自己的 env/init/step 與 search fn，
        # 非預設貼目第一手會多等一次編譯，之後走快取
        # LRU：貼目可由使用者自訂（-81~81 的半整數），不設上限的話
        # 每個新值都會永久佔住一份編譯產物。DEFAULT_KOMI 永不淘汰。
        self._envs: OrderedDict[float, object] = OrderedDict({DEFAULT_KOMI: self.env})
        self._env_fns: OrderedDict[float, tuple] = OrderedDict()
        self._searches: OrderedDict[tuple[float, str], object] = OrderedDict()
        # 讓子星位：右上、左下、右下、左上（9 路 hoshi 在三線）
        h = 2 if self.size < 13 else 3
        t = self.size - 1 - h
        self.handicap_actions = [r * self.size + c
                                 for r, c in ((h, t), (t, h), (t, t), (h, h))]
        self.games: dict[str, Game] = {}
        # 只保護 games dict 的增刪查，握著的時間是微秒等級。
        # 盤面變更與推理由 Game.lock 保護，所以不同局能並行吃滿多核。
        self.games_lock = threading.Lock()
        # 保護 JIT 快取的填充；_env 會被 env_fns/search_fn 巢狀呼叫，故用 RLock
        self.cache_lock = threading.RLock()
        # trigger compilation up-front so the first app move isn't slow
        g = Game(self, "easy", "black")
        self.env_fns(DEFAULT_KOMI)[1](g.state, jnp.int32(0))
        for name in LEVELS:
            self.search_fn(DEFAULT_KOMI, name)(
                self.params, jax.random.PRNGKey(0), batch_of_one(g.state))
        self.forward(batch_of_one(g.state).observation)

    def _trim_komi_cache(self):
        """淘汰最久沒用到的貼目，預設貼目永久保留。呼叫端須持有 cache_lock。"""
        while len(self._envs) > self.MAX_KOMI_CACHE:
            for komi in self._envs:  # OrderedDict 頭端 = 最久沒用到
                if komi != DEFAULT_KOMI:
                    self._envs.pop(komi)
                    self._env_fns.pop(komi, None)
                    for level in LEVELS:
                        self._searches.pop((komi, level), None)
                    break
            else:
                return  # 只剩預設貼目，沒得淘汰

    def _env(self, komi: float):
        with self.cache_lock:
            if komi in self._envs:
                self._envs.move_to_end(komi)
            else:
                self._envs[komi] = Go(size=self.size, komi=komi)
                self._trim_komi_cache()
            return self._envs[komi]

    def env_fns(self, komi: float) -> tuple:
        """(jitted init, jitted step) for the given komi.
        eager env.step costs ~200ms/move on CPU; jitted it's ~1ms"""
        with self.cache_lock:
            if komi not in self._env_fns:
                env = self._env(komi)
                self._env_fns[komi] = (jax.jit(env.init), jax.jit(env.step))
            return self._env_fns[komi]

    def search_fn(self, komi: float, level: str):
        with self.cache_lock:
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

    # -- game registry --------------------------------------------------------
    def get_game(self, game_id) -> Game | None:
        if not isinstance(game_id, str):
            return None
        with self.games_lock:
            game = self.games.get(game_id)
            if game is not None:
                game.touched = time.time()
            return game

    def register(self, game_id: str, game: Game):
        """收錄新對局，順便清掉逾時的舊局。"""
        now = time.time()
        with self.games_lock:
            for gid in [g for g, x in self.games.items()
                        if now - x.touched > self.GAME_TTL]:
                self.games.pop(gid, None)
            self.games[game_id] = game
            # 真的還超量才淘汰，且挑「最久沒動作」的，不是「最早建立」的——
            # 正在下的棋不會因為別人開新局而消失。
            while len(self.games) > self.MAX_GAMES:
                oldest = min(self.games, key=lambda g: self.games[g].touched)
                self.games.pop(oldest, None)

    # -- persistence: survive server restarts ---------------------------------
    def save_games(self, path: str):
        with self.games_lock:
            games = list(self.games.items())
        data = {}
        for gid, g in games:
            with g.lock:  # 避免讀到寫到一半的 history
                data[gid] = {"level": g.level, "human_color": g.human_color,
                             "history": list(g.history), "resigned_by": g.resigned_by,
                             "komi": g.komi, "handicap": g.handicap,
                             "touched": g.touched}
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
            game.touched = d.get("touched", time.time())
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
        try:
            return self._get()
        except Exception:
            traceback.print_exc()
            return self._send({"error": "internal error"}, 500)

    def _get(self):
        e = self.engine
        if self.path.startswith("/state"):
            # read-only resync endpoint: /state?game_id=xxx
            from urllib.parse import parse_qs, urlparse
            gid = parse_qs(urlparse(self.path).query).get("game_id", [""])[0]
            game = e.get_game(gid)
            if game is None:
                return self._send({"error": "unknown game"}, 404)
            with game.lock:
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
        try:
            return self._post()
        except ValueError as err:
            return self._send({"error": str(err)}, 400)
        except Exception:
            # 沒有這一層，畸形請求會讓 handler thread 拋例外、
            # 連 HTTP 狀態行都不回，client 只看到空回應。
            traceback.print_exc()
            return self._send({"error": "internal error"}, 500)

    def _read_json(self):
        """回傳 (dict, None) 或 (None, 已送出的錯誤回應)。"""
        try:
            n = int(self.headers.get("Content-Length", 0) or 0)
        except ValueError:
            self._send({"error": "bad content-length"}, 400)
            return None, True
        if n < 0 or n > MAX_BODY:
            self._send({"error": "payload too large"}, 413)
            return None, True
        try:
            req = json.loads(self.rfile.read(n) or b"{}")
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send({"error": "malformed json"}, 400)
            return None, True
        if not isinstance(req, dict):
            self._send({"error": "body must be a JSON object"}, 400)
            return None, True
        return req, None

    def _post(self):
        req, sent = self._read_json()
        if sent:
            return
        e = self.engine

        if self.path == "/new":
            level = req.get("level", "normal")
            human = req.get("human_color", "black")
            try:
                komi = float(req.get("komi", DEFAULT_KOMI))
                handicap = int(req.get("handicap", 0))
            except (TypeError, ValueError):
                return self._send({"error": "bad komi/handicap"}, 400)
            if level not in LEVELS or human not in ("black", "white"):
                return self._send({"error": "bad level/color"}, 400)
            # 半整數避免 JIT 快取被連續值撐爆；範圍蓋住 9 路全盤
            if komi != komi or not (komi * 2).is_integer() or not -81 <= komi <= 81:
                return self._send({"error": "bad komi"}, 400)
            if handicap not in HANDICAPS:
                return self._send({"error": "bad handicap"}, 400)
            game_id = uuid.uuid4().hex[:12]
            # 先把開局算完再收錄：這段期間別的執行緒看不到這局，不必上鎖搶。
            game = Game(e, level, human, komi, handicap)
            e.apply_handicap(game)
            to_move = "black" if len(game.history) % 2 == 0 else "white"
            ai = e.ai_move(game) if to_move != human else None
            e.register(game_id, game)
            return self._send(e.snapshot(game, game_id, ai_move=ai))

        gid = req.get("game_id", "")
        game = e.get_game(gid)
        if game is None:
            return self._send({"error": "unknown game"}, 404)

        # 從這裡開始只鎖這一局：別人的對局照常並行推理。
        with game.lock:
            if self.path == "/move":
                if game.resigned_by or bool(game.state.terminated | game.state.truncated):
                    return self._send({"error": "game over"}, 400)
                try:
                    action = int(req["action"])
                except (KeyError, TypeError, ValueError):
                    return self._send({"error": "bad action"}, 400)
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
