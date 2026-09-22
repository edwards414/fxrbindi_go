"""Per-player stamina: a soft cap on how many games one person can start.

Every player starts full (STAMINA_MAX points).  Starting a game costs one
point; one point regenerates every REGEN_SECONDS.  The point of the system is
to keep the number of simultaneous games on a small home server bounded
without accounts: the app sends a random ``player_id`` it generated on first
launch, and old builds that don't send one fall back to their client IP.

Pure Python (no JAX) so it can be unit-tested without a checkpoint.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from typing import Callable

STAMINA_MAX = int(os.environ.get("GOZERO_STAMINA_MAX", 24))
STAMINA_REGEN_SECONDS = int(os.environ.get("GOZERO_STAMINA_REGEN_SECONDS", 3600))
NEW_GAME_COST = 1
# 第二層：同一個對外 IP 不管換多少個 player_id，開局總量也有上限。
# 沒有這層的話，一支腳本每局換一個隨機 id 就能無限開局，體力形同虛設。
# 預設 72 局 / 每 20 分鐘回 1（= 一天 72 局），比單人 24 局寬鬆，
# 同一 NAT 底下三個人同時玩滿也不會被誤擋。
IP_STAMINA_MAX = int(os.environ.get("GOZERO_IP_STAMINA_MAX", 72))
IP_STAMINA_REGEN_SECONDS = int(os.environ.get("GOZERO_IP_STAMINA_REGEN_SECONDS", 1200))
# 帳本筆數上限：攻擊者可以用隨機 player_id 灌爆記憶體，滿了就淘汰最接近回滿的
MAX_PLAYERS = int(os.environ.get("GOZERO_STAMINA_MAX_PLAYERS", 50_000))

PLAYER_ID_RE = re.compile(r"[A-Za-z0-9._:-]{8,64}")


class StaminaExhausted(Exception):
    def __init__(self, view: dict):
        super().__init__("stamina exhausted")
        self.view = view


class StaminaBank:
    """Thread-safe ledger of stamina points keyed by player id.

    Regeneration is lazy: each entry remembers ``points`` and ``since``, the
    timestamp the current regeneration window started.  On every access the
    elapsed whole windows are credited and ``since`` advanced by the same
    amount, so a player who waits 90 minutes gets one point back and keeps the
    30 minutes of progress toward the next one.
    """

    def __init__(
        self,
        max_points: int = STAMINA_MAX,
        regen_seconds: int = STAMINA_REGEN_SECONDS,
        clock: Callable[[], float] = time.time,
        max_players: int = MAX_PLAYERS,
    ):
        self.max_points = max_points
        self.regen_seconds = regen_seconds
        self.clock = clock
        self.max_players = max_players
        self._players: dict[str, dict] = {}
        self._lock = threading.Lock()

    # -- internals ----------------------------------------------------------
    def _refresh(self, player_id: str, now: float) -> dict:
        entry = self._players.get(player_id)
        if entry is None:
            if len(self._players) >= self.max_players:
                self._evict_locked(now)
            entry = {"points": self.max_points, "since": now}
            self._players[player_id] = entry
            return entry
        if entry["points"] >= self.max_points:
            entry["since"] = now
            return entry
        elapsed = now - entry["since"]
        if elapsed < 0:  # 時鐘倒退（NTP 校正）：不倒扣，重新起算
            entry["since"] = now
            return entry
        gained = int(elapsed // self.regen_seconds)
        if gained:
            entry["points"] = min(self.max_points, entry["points"] + gained)
            entry["since"] = (
                now if entry["points"] >= self.max_points
                else entry["since"] + gained * self.regen_seconds
            )
        return entry

    def _evict_locked(self, now: float) -> None:
        """Make room: drop everyone back to full, then the fullest / oldest.

        Eviction only ever *gives* stamina back (a dropped player reappears
        full), so a flood of random ids can cost memory, never someone's game.
        """
        for pid in [p for p in list(self._players)
                    if self._refresh(p, now)["points"] >= self.max_points]:
            self._players.pop(pid, None)
        # 一次多清一批（10%），不然帳本滿了之後每個新 id 都要排序 5 萬筆
        excess = len(self._players) - self.max_players + max(1, self.max_players // 10)
        if excess <= 0:
            return
        victims = sorted(
            self._players,
            key=lambda p: (-self._players[p]["points"], self._players[p]["since"]),
        )[:excess]
        for pid in victims:
            self._players.pop(pid, None)

    def __len__(self) -> int:
        with self._lock:
            return len(self._players)

    def _view(self, entry: dict, now: float) -> dict:
        full = entry["points"] >= self.max_points
        return {
            "points": int(entry["points"]),
            "max": self.max_points,
            "regen_seconds": self.regen_seconds,
            # 距離下一點回復的秒數；滿了就是 None
            "next_in_seconds": None if full else max(
                0, int(self.regen_seconds - (now - entry["since"]))),
        }

    # -- public API ---------------------------------------------------------
    def view(self, player_id: str) -> dict:
        now = self.clock()
        with self._lock:
            return self._view(self._refresh(player_id, now), now)

    def spend(self, player_id: str, cost: int = NEW_GAME_COST) -> dict:
        """Deduct ``cost`` points or raise StaminaExhausted (nothing deducted)."""
        now = self.clock()
        with self._lock:
            entry = self._refresh(player_id, now)
            if entry["points"] < cost:
                raise StaminaExhausted(self._view(entry, now))
            was_full = entry["points"] >= self.max_points
            entry["points"] -= cost
            if was_full:  # 從滿格開始扣的那一刻起算回復
                entry["since"] = now
            return self._view(entry, now)

    def refund(self, player_id: str, amount: int = NEW_GAME_COST) -> None:
        now = self.clock()
        with self._lock:
            entry = self._refresh(player_id, now)
            entry["points"] = min(self.max_points, entry["points"] + amount)

    def prune(self) -> None:
        """Drop players who are back to full; they'd be recreated identical."""
        now = self.clock()
        with self._lock:
            for pid in [p for p in list(self._players)
                        if self._refresh(p, now)["points"] >= self.max_points]:
                self._players.pop(pid, None)

    # -- persistence --------------------------------------------------------
    def save(self, path: str) -> None:
        self.prune()
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with self._lock:
            data = {"version": 1, "players": {
                pid: {"points": e["points"], "since": e["since"]}
                for pid, e in self._players.items()
            }}
        tmp = f"{path}.tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, path)

    def load(self, path: str) -> None:
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return
        players = data.get("players", {}) if isinstance(data, dict) else {}
        with self._lock:
            for pid, e in players.items():
                if len(self._players) >= self.max_players:
                    break
                try:
                    points = int(e["points"])
                    since = float(e["since"])
                except (KeyError, TypeError, ValueError):
                    continue
                self._players[pid] = {
                    "points": max(0, min(self.max_points, points)),
                    "since": since,
                }
        if self._players:
            print(f"restored stamina for {len(self._players)} player(s)", flush=True)


def client_ip_key(headers, client_ip: str) -> str:
    """Bucket key for the caller's public address.

    Behind cloudflared the real address is in CF-Connecting-IP, which
    Cloudflare sets itself (a client-supplied copy is overwritten).  The
    X-Forwarded-For fallback is for other reverse proxies; the origin is only
    reachable through the tunnel, so neither can be forged from outside.
    """
    ip = (headers.get("CF-Connecting-IP")
          or (headers.get("X-Forwarded-For") or "").split(",")[0].strip()
          or client_ip)
    return f"ip:{ip[:64]}"


def player_key(req: dict, headers, client_ip: str) -> str:
    """Identify the player: explicit id from the app, else the client IP.

    Old builds don't send ``player_id``; bucketing them by IP still limits
    the server's total load.
    """
    pid = req.get("player_id")
    if isinstance(pid, str) and PLAYER_ID_RE.fullmatch(pid):
        return f"id:{pid}"
    if pid is not None:
        raise ValueError("bad player_id")
    return client_ip_key(headers, client_ip)
