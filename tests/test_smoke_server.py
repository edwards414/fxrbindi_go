import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from scripts.smoke_test_server import (
    SmokeError,
    _validate_game,
    _validate_health,
    _validate_queue,
    smoke_test,
)


def queue_payload():
    return {
        "workers": 4,
        "active": 0,
        "queued": 0,
        "queued_standard": 0,
        "queued_premium": 0,
        "queue_capacity": 64,
        "average_job_seconds": 1.25,
        "estimated_tail_wait_seconds": 0,
    }


def game_payload(size=19):
    area = size * size
    return {
        "game_id": "012345abcdef",
        "board": [0] * (area - 1) + [1],
        "size": size,
        "to_move": "white",
        "human_color": "white",
        "moves": 1,
        "history": [area - 1],
        "last_move": area - 1,
        "ai_move": area - 1,
        "legal": [1] * (area + 1),
        "black_winrate": 0.51,
        "winrates": [0.5, 0.51],
        "captures": {"black": 0, "white": 0},
        "game_over": False,
        "result": None,
        "komi": 7.5,
        "handicap": 0,
        "setup_plies": 0,
    }


class FakeGoZeroHandler(BaseHTTPRequestHandler):
    job_id = "a" * 32
    board_size = 19

    def log_message(self, _format, *_args):
        pass

    def send_json(self, payload, status=200, headers=None):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            return self.send_json(
                {
                    "ok": True,
                    "model": "gozero go_19x19 192ch x 12blk",
                    "iteration": 1000,
                    "board_size": 19,
                    "board_sizes": [9, 19],
                    "models": [
                        {
                            "model": "gozero go_9x9 192ch x 12blk",
                            "iteration": 4628,
                            "board_size": 9,
                        },
                        {
                            "model": "gozero go_19x19 192ch x 12blk",
                            "iteration": 1000,
                            "board_size": 19,
                        },
                    ],
                    "queue": queue_payload(),
                }
            )
        if self.path == "/queue":
            return self.send_json(queue_payload())
        if self.path == f"/jobs/{self.job_id}":
            return self.send_json(
                {
                    "job_id": self.job_id,
                    "status": "completed",
                    "result": game_payload(type(self).board_size),
                }
            )
        if self.path.startswith("/state?game_id="):
            state = game_payload(type(self).board_size)
            state["ai_move"] = None
            return self.send_json(state)
        return self.send_json({"error": "not found"}, 404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        if self.path == "/new":
            if payload.get("human_color") != "white" or not self.headers.get(
                "Idempotency-Key"
            ):
                return self.send_json({"error": "bad new request"}, 400)
            type(self).board_size = payload.get("board_size")
            return self.send_json(
                {"job_id": self.job_id, "status": "queued", "queue_position": 1},
                202,
                {"location": f"/jobs/{self.job_id}"},
            )
        if self.path == "/resign":
            resigned = game_payload(type(self).board_size)
            resigned["game_over"] = True
            resigned["result"] = {
                "winner": "black",
                "reason": "resign",
                "margin": None,
            }
            return self.send_json(resigned)
        return self.send_json({"error": "not found"}, 404)


class ServerSmokeValidationTest(unittest.TestCase):
    def test_full_http_smoke_flow(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), FakeGoZeroHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            result = smoke_test(
                f"http://127.0.0.1:{server.server_port}",
                expected_model="gozero go_19x19 192ch x 12blk",
                expected_iteration=1000,
                expected_board_size=19,
                timeout=2,
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(2)
        self.assertEqual(result["iteration"], 1000)
        self.assertEqual(result["ai_move"], 360)

    def test_full_http_smoke_flow_9x9(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), FakeGoZeroHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            result = smoke_test(
                f"http://127.0.0.1:{server.server_port}",
                expected_model="gozero go_9x9 192ch x 12blk",
                expected_iteration=4628,
                expected_board_size=9,
                timeout=2,
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(2)
        self.assertEqual(result["iteration"], 4628)
        self.assertEqual(result["board_size"], 9)
        self.assertEqual(result["ai_move"], 80)

    def test_accepts_expected_health_queue_and_game(self):
        health = {
            "ok": True,
            "model": "gozero go_19x19 192ch x 12blk",
            "iteration": 1000,
            "board_size": 19,
            "queue": queue_payload(),
        }
        _validate_health(
            health,
            expected_model="gozero go_19x19 192ch x 12blk",
            expected_iteration=1000,
            expected_board_size=19,
        )
        _validate_game(
            game_payload(),
            expected_board_size=19,
            expected_human_color="white",
            require_ai_move=True,
        )

    def test_selects_9x9_identity_from_dual_model_health(self):
        health = {
            "ok": True,
            "model": "gozero go_19x19 192ch x 12blk",
            "iteration": 1000,
            "board_size": 19,
            "board_sizes": [9, 19],
            "models": [
                {
                    "model": "gozero go_9x9 192ch x 12blk",
                    "iteration": 4628,
                    "board_size": 9,
                },
                {
                    "model": "gozero go_19x19 192ch x 12blk",
                    "iteration": 1000,
                    "board_size": 19,
                },
            ],
            "queue": queue_payload(),
        }
        _validate_health(
            health,
            expected_model="gozero go_9x9 192ch x 12blk",
            expected_iteration=4628,
            expected_board_size=9,
        )

    def test_rejects_wrong_board_size(self):
        game = game_payload()
        game["size"] = 9
        with self.assertRaisesRegex(SmokeError, "board size"):
            _validate_game(
                game,
                expected_board_size=19,
                expected_human_color="white",
                require_ai_move=True,
            )

    def test_rejects_missing_ai_move(self):
        game = game_payload()
        game["ai_move"] = None
        with self.assertRaisesRegex(SmokeError, "AI did not produce"):
            _validate_game(
                game,
                expected_board_size=19,
                expected_human_color="white",
                require_ai_move=True,
            )

    def test_rejects_inconsistent_queue_lanes(self):
        queue = queue_payload()
        queue["queued"] = 2
        with self.assertRaisesRegex(SmokeError, "lane counts"):
            _validate_queue(queue)


if __name__ == "__main__":
    unittest.main()
