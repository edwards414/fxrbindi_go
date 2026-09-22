import json
import os
import tempfile
import unittest

from gozero.stamina import StaminaBank, StaminaExhausted, player_key


class FakeClock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


class StaminaBankTest(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.bank = StaminaBank(max_points=24, regen_seconds=3600, clock=self.clock)

    def test_new_player_starts_full(self):
        v = self.bank.view("id:alice")
        self.assertEqual(v["points"], 24)
        self.assertEqual(v["max"], 24)
        self.assertIsNone(v["next_in_seconds"])

    def test_spend_deducts_and_starts_regen_clock(self):
        v = self.bank.spend("id:alice")
        self.assertEqual(v["points"], 23)
        self.assertEqual(v["next_in_seconds"], 3600)
        self.clock.advance(1800)
        self.assertEqual(self.bank.view("id:alice")["next_in_seconds"], 1800)

    def test_one_point_per_hour_with_carry(self):
        for _ in range(5):
            self.bank.spend("id:alice")
        self.assertEqual(self.bank.view("id:alice")["points"], 19)
        self.clock.advance(3600 * 2 + 600)  # 2h10m → +2, 10 min carried
        v = self.bank.view("id:alice")
        self.assertEqual(v["points"], 21)
        self.assertEqual(v["next_in_seconds"], 3000)

    def test_regen_caps_at_max(self):
        self.bank.spend("id:alice")
        self.clock.advance(3600 * 100)
        v = self.bank.view("id:alice")
        self.assertEqual(v["points"], 24)
        self.assertIsNone(v["next_in_seconds"])

    def test_exhausted_raises_without_deducting(self):
        for _ in range(24):
            self.bank.spend("id:alice")
        with self.assertRaises(StaminaExhausted) as ctx:
            self.bank.spend("id:alice")
        self.assertEqual(ctx.exception.view["points"], 0)
        self.assertEqual(ctx.exception.view["next_in_seconds"], 3600)
        self.clock.advance(3600)
        self.assertEqual(self.bank.spend("id:alice")["points"], 0)

    def test_refund_on_failed_start(self):
        self.bank.spend("id:alice")
        self.bank.refund("id:alice")
        self.assertEqual(self.bank.view("id:alice")["points"], 24)

    def test_players_are_independent(self):
        self.bank.spend("id:alice")
        self.assertEqual(self.bank.view("ip:1.2.3.4")["points"], 24)

    def test_clock_going_backwards_does_not_penalise(self):
        self.bank.spend("id:alice")
        self.clock.advance(-7200)
        v = self.bank.view("id:alice")
        self.assertEqual(v["points"], 23)
        self.assertEqual(v["next_in_seconds"], 3600)

    def test_save_load_round_trip_drops_full_players(self):
        self.bank.spend("id:alice")
        self.bank.spend("id:alice")
        self.bank.view("id:bob")  # full → not persisted
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "stamina.json")
            self.bank.save(path)
            with open(path) as f:
                data = json.load(f)
            self.assertEqual(set(data["players"]), {"id:alice"})
            self.clock.advance(3600)
            other = StaminaBank(max_points=24, regen_seconds=3600, clock=self.clock)
            other.load(path)
            v = other.view("id:alice")
            self.assertEqual(v["points"], 23)
            self.assertEqual(v["next_in_seconds"], 3600)

    def test_load_missing_or_corrupt_file_is_noop(self):
        self.bank.load("/nonexistent/stamina.json")
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "stamina.json")
            with open(path, "w") as f:
                f.write("{not json")
            self.bank.load(path)
        self.assertEqual(self.bank.view("id:alice")["points"], 24)


class StaminaBankBoundsTest(unittest.TestCase):
    def test_ledger_is_bounded_and_eviction_never_costs_points(self):
        clock = FakeClock()
        bank = StaminaBank(max_points=5, regen_seconds=60, clock=clock, max_players=20)
        for i in range(20):
            bank.spend(f"id:spam-{i:04d}")  # 20 個都在 4/5
        self.assertEqual(len(bank), 20)
        bank.spend("id:victim-0001")  # 第 21 個觸發淘汰（一次清一批）
        self.assertLessEqual(len(bank), 20)
        # 被淘汰的人再出現是滿格，不會被扣到負分或卡在 0
        clock.advance(0)
        for i in range(20):
            self.assertGreaterEqual(bank.view(f"id:spam-{i:04d}")["points"], 4)

    def test_eviction_prefers_players_closest_to_full(self):
        clock = FakeClock()
        bank = StaminaBank(max_points=5, regen_seconds=60, clock=clock, max_players=10)
        for _ in range(5):
            bank.spend("id:heavy-user-1")  # 0/5：最該保留的紀錄
        for i in range(9):
            bank.spend(f"id:light-{i:03d}")  # 4/5
        bank.spend("id:newcomer-1")
        self.assertEqual(bank.view("id:heavy-user-1")["points"], 0)

    def test_load_respects_cap(self):
        import json
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "s.json")
            with open(path, "w") as f:
                json.dump({"version": 1, "players": {
                    f"id:p{i}": {"points": 1, "since": 0} for i in range(100)}}, f)
            bank = StaminaBank(max_points=5, regen_seconds=60, clock=FakeClock(0), max_players=10)
            bank.load(path)
            self.assertEqual(len(bank), 10)


class PlayerKeyTest(unittest.TestCase):
    def test_explicit_player_id(self):
        key = player_key({"player_id": "abcdef0123456789"}, {}, "10.0.0.1")
        self.assertEqual(key, "id:abcdef0123456789")

    def test_bad_player_id_rejected(self):
        with self.assertRaises(ValueError):
            player_key({"player_id": "x"}, {}, "10.0.0.1")
        with self.assertRaises(ValueError):
            player_key({"player_id": 42}, {}, "10.0.0.1")

    def test_legacy_client_falls_back_to_cloudflare_ip(self):
        headers = {"CF-Connecting-IP": "203.0.113.7", "X-Forwarded-For": "9.9.9.9"}
        self.assertEqual(player_key({}, headers, "172.18.0.2"), "ip:203.0.113.7")

    def test_legacy_client_falls_back_to_forwarded_then_socket(self):
        self.assertEqual(
            player_key({}, {"X-Forwarded-For": "198.51.100.4, 10.0.0.9"}, "172.18.0.2"),
            "ip:198.51.100.4",
        )
        self.assertEqual(player_key({}, {}, "172.18.0.2"), "ip:172.18.0.2")


if __name__ == "__main__":
    unittest.main()
