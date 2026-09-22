"""HTTP-level tests for stamina and the undo limit, run against the real
Handler with the JAX-free fake engine from scripts/fake_engine_server.py."""
import json
import threading
import time
import unittest
import urllib.error
import urllib.request

from gozero.stamina import StaminaBank
from scripts import fake_engine_server as fake

server = fake.server


class FakeClock:
    def __init__(self):
        self.t = 1_000_000.0

    def __call__(self):
        return self.t


class ServerStaminaUndoTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clock = FakeClock()
        cls.bank = StaminaBank(max_points=3, regen_seconds=3600, clock=cls.clock)
        cls.pool = fake.FakeEnginePool(sizes=(9,), stamina=cls.bank)
        cls.httpd = fake.serve("127.0.0.1", 0, cls.pool)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_port}"
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.pool.jobs.shutdown() if hasattr(cls.pool.jobs, "shutdown") else None

    # -- helpers --------------------------------------------------------------
    def _call(self, method, path, body=None, headers=None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            self.base + path, data=data, method=method,
            headers={"Content-Type": "application/json", **(headers or {})})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, json.loads(r.read()), dict(r.headers)
        except urllib.error.HTTPError as err:
            return err.code, json.loads(err.read()), dict(err.headers)

    def _wait(self, job):
        deadline = time.time() + 10
        while job["status"] in ("queued", "running"):
            self.assertLess(time.time(), deadline, "job did not finish")
            time.sleep(0.02)
            _, job, _ = self._call("GET", f"/jobs/{job['job_id']}")
        return job

    def _new(self, player_id=None, request_id=None, headers=None, **extra):
        body = {"level": "easy", "human_color": "black", "board_size": 9, **extra}
        if player_id is not None:
            body["player_id"] = player_id
        if request_id:
            body["request_id"] = request_id
        return self._call("POST", "/new", body, headers)

    def _sync_new(self, player_id, headers=None):
        # legacy (no request_id): handler waits and returns the GameState directly
        status, body, _ = self._new(player_id=player_id, headers=headers)
        return status, body

    # -- tests ----------------------------------------------------------------
    def test_health_advertises_limits(self):
        status, body, _ = self._call("GET", "/health")
        self.assertEqual(status, 200)
        self.assertEqual(body["stamina"]["max"], 3)
        self.assertEqual(body["stamina"]["new_game_cost"], 1)
        self.assertEqual(body["undo_limit"], server.MAX_UNDOS)

    def test_new_game_costs_one_point_and_reports_stamina(self):
        pid = "alice-0001-abcd"
        status, body, _ = self._call("GET", f"/stamina?player_id={pid}")
        self.assertEqual((status, body["points"]), (200, 3))

        status, game = self._sync_new(pid)
        self.assertEqual(status, 200)
        self.assertEqual(game["stamina"]["points"], 2)
        self.assertEqual(game["stamina"]["next_in_seconds"], 3600)
        self.assertEqual(game["undos_left"], server.MAX_UNDOS)
        self.assertEqual(game["undo_limit"], server.MAX_UNDOS)

        _, body, _ = self._call("GET", f"/stamina?player_id={pid}")
        self.assertEqual(body["points"], 2)

    def test_exhausted_player_gets_429_with_countdown(self):
        pid = "bob-0001-abcd"
        for expected in (2, 1, 0):
            status, game = self._sync_new(pid)
            self.assertEqual(status, 200)
            self.assertEqual(game["stamina"]["points"], expected)
        status, body, headers = self._new(player_id=pid)
        self.assertEqual(status, 429)
        self.assertEqual(body["error"], "stamina exhausted")
        self.assertEqual(body["stamina"]["points"], 0)
        self.assertEqual(body["stamina"]["next_in_seconds"], 3600)
        self.assertEqual(headers.get("Retry-After"), "3600")
        # one hour later exactly one game is allowed again
        self.clock.t += 3600
        status, game = self._sync_new(pid)
        self.assertEqual(status, 200)
        self.assertEqual(game["stamina"]["points"], 0)
        status, _, _ = self._new(player_id=pid)
        self.assertEqual(status, 429)

    def test_queued_new_game_deducts_once_even_when_retried(self):
        pid = "carol-0001-abcd"
        rid = "retry-0001-abcdef"
        status, job, _ = self._new(player_id=pid, request_id=rid)
        self.assertEqual(status, 202)
        first = self._wait(job)
        self.assertEqual(first["status"], "completed")
        self.assertEqual(first["result"]["stamina"]["points"], 2)
        # same Idempotency key again: same job, no second deduction
        status, again, _ = self._new(player_id=pid, request_id=rid)
        self.assertIn(status, (200, 202))
        self.assertEqual(self._wait(again)["result"]["game_id"], first["result"]["game_id"])
        _, body, _ = self._call("GET", f"/stamina?player_id={pid}")
        self.assertEqual(body["points"], 2)

    def test_legacy_client_is_bucketed_by_cloudflare_ip(self):
        headers = {"CF-Connecting-IP": "203.0.113.9"}
        for _ in range(3):
            status, _ = self._sync_new(None, headers=headers)
            self.assertEqual(status, 200)
        status, body, _ = self._new(headers=headers)
        self.assertEqual(status, 429)
        # a different address is a different bucket
        status, _ = self._sync_new(None, headers={"CF-Connecting-IP": "203.0.113.10"})
        self.assertEqual(status, 200)

    def test_ip_budget_stops_player_id_rotation(self):
        # 每局換一個 player_id 的腳本：IP 桶要擋下來
        self.pool.ip_stamina = StaminaBank(max_points=4, regen_seconds=3600, clock=self.clock)
        headers = {"CF-Connecting-IP": "198.51.100.77"}
        for i in range(4):
            status, _ = self._sync_new(f"rotate-{i:04d}-abcd", headers=headers)
            self.assertEqual(status, 200)
        status, body, _ = self._new(player_id="rotate-9999-abcd", headers=headers)
        self.assertEqual(status, 429)
        self.assertEqual(body["stamina"]["points"], 0)
        self.assertEqual(body["stamina"]["next_in_seconds"], 3600)
        # 別的 IP 不受影響
        status, _ = self._sync_new("rotate-0000-zzzz", headers={"CF-Connecting-IP": "198.51.100.78"})
        self.assertEqual(status, 200)
        self.pool.ip_stamina = StaminaBank(clock=self.clock)

    def test_connection_cap_refuses_extra_connections(self):
        import socket

        old = server.Server.MAX_CONNECTIONS
        server.Server.MAX_CONNECTIONS = 2
        idle = []
        try:
            for _ in range(2):  # 兩條只連不講話的 slowloris 連線佔滿名額
                s = socket.create_connection(("127.0.0.1", self.httpd.server_port), timeout=5)
                idle.append(s)
            time.sleep(0.3)
            with self.assertRaises((urllib.error.URLError, ConnectionError, OSError)):
                urllib.request.urlopen(self.base + "/health", timeout=3)
        finally:
            server.Server.MAX_CONNECTIONS = old
            for s in idle:
                s.close()
        time.sleep(0.3)
        status, body, _ = self._call("GET", "/health")  # 名額釋放後恢復
        self.assertEqual(status, 200)

    def test_bad_player_id_rejected(self):
        status, body, _ = self._new(player_id="x")
        self.assertEqual(status, 400)
        status, body, _ = self._call("GET", "/stamina?player_id=x")
        self.assertEqual(status, 400)

    def test_undo_limited_per_game(self):
        pid = "dave-0001-abcd"
        status, game = self._sync_new(pid)
        self.assertEqual(status, 200)
        gid = game["game_id"]
        moves = game["moves"]
        for i in range(server.MAX_UNDOS):
            legal = next(a for a, ok in enumerate(game["legal"][:-1]) if ok)
            status, game, _ = self._call(
                "POST", "/move", {"game_id": gid, "action": legal, "expected_moves": moves})
            self.assertEqual(status, 200)
            moves = game["moves"]
            status, game, _ = self._call(
                "POST", "/undo", {"game_id": gid, "expected_moves": moves})
            self.assertEqual(status, 200, game)
            moves = game["moves"]
            self.assertEqual(game["undos_used"], i + 1)
            self.assertEqual(game["undos_left"], server.MAX_UNDOS - i - 1)
        legal = next(a for a, ok in enumerate(game["legal"][:-1]) if ok)
        status, game, _ = self._call(
            "POST", "/move", {"game_id": gid, "action": legal, "expected_moves": moves})
        self.assertEqual(status, 200)
        status, body, _ = self._call(
            "POST", "/undo", {"game_id": gid, "expected_moves": game["moves"]})
        self.assertEqual(status, 400)
        self.assertEqual(body["error"], "undo limit reached")
        # the board is untouched by the refused undo
        _, state, _ = self._call("GET", f"/state?game_id={gid}")
        self.assertEqual(state["moves"], game["moves"])
        self.assertEqual(state["undos_left"], 0)

    def test_undo_count_survives_save_and_load(self):
        import os
        import tempfile

        engine = self.pool.engines[9]
        pid = "erin-0001-abcd"
        status, game = self._sync_new(pid)
        gid = game["game_id"]
        legal = next(a for a, ok in enumerate(game["legal"][:-1]) if ok)
        _, game, _ = self._call(
            "POST", "/move", {"game_id": gid, "action": legal, "expected_moves": game["moves"]})
        _, game, _ = self._call(
            "POST", "/undo", {"game_id": gid, "expected_moves": game["moves"]})
        self.assertEqual(game["undos_used"], 1)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "games.json")
            engine.save_games(path)
            other = fake.FakeEngine(9, self.pool.jobs)
            other.load_games(path)
            self.assertEqual(other.games[gid].undos_used, 1)


if __name__ == "__main__":
    unittest.main()
